#define _GNU_SOURCE


#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "internal.h"

typedef struct AllocationRequest {
    ShadowSpillRuntime *runtime;
    ShadowSpillBackendStream stream;
    ShadowSpillStatus status;
    ShadowSpillAllocation allocation;
} AllocationRequest;

static void *allocate_from_thread(void *pointer) {
    AllocationRequest *request = pointer;
    request->status = shadowspill_memory_pool_allocate(request->runtime, 0U,
        128U,
        1U,
        request->stream,
        &request->allocation
    );
    return NULL;
}

static int a_blocked_allocator_resumes_when_the_free_lands(void) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig mock_config = {
        .event_delay_nanoseconds = 2000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(&mock, 128U, 1U, 1U, 10000U, &topology
    );
    if (shadowspill_runtime_create(&topology.runtime, &runtime) !=
        SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    ShadowSpillBackendStream first_stream = {{0U, 0U}};
    ShadowSpillBackendStream second_stream = {{0U, 0U}};
    ShadowSpillAllocation first = {0};
    if (mock.create_stream(mock.state, &first_stream) != 0 ||
        mock.create_stream(mock.state, &second_stream) != 0 ||
        shadowspill_memory_pool_allocate(runtime, 0U, 128U, 1U, first_stream, &first) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_record_stream(runtime, 0U, first.allocation_id, second_stream
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_free(runtime, 0U, first.allocation_id, first_stream) !=
            SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    AllocationRequest request = {
        .runtime = runtime,
        .stream = first_stream,
        .status = SHADOWSPILL_STATUS_INVALID_STATE,
    };
    pthread_t thread;
    const int create_status = pthread_create(
        &thread, NULL, allocate_from_thread, &request
    );
    const int join_status = create_status == 0
        ? pthread_join(thread, NULL) : -1;
    const ShadowSpillStatus free_status =
        request.status == SHADOWSPILL_STATUS_OK
        ? shadowspill_memory_pool_free(
              runtime,
              0U,
              request.allocation.allocation_id,
              first_stream
          )
        : request.status;
    const ShadowSpillStatus wait_status =
        free_status == SHADOWSPILL_STATUS_OK
        ? shadowspill_runtime_wait_idle(runtime) : free_status;
    if (create_status != 0 || join_status != 0 ||
        request.status != SHADOWSPILL_STATUS_OK ||
        request.allocation.pointer == NULL ||
        free_status != SHADOWSPILL_STATUS_OK ||
        wait_status != SHADOWSPILL_STATUS_OK) {
        fprintf(
            stderr,
            "blocking statuses: create=%d join=%d allocate=%u pointer=%p "
            "free=%u wait=%u\n",
            create_status,
            join_status,
            (unsigned)request.status,
            request.allocation.pointer,
            (unsigned)free_status,
            (unsigned)wait_status
        );
        return -1;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_STATUS_OK ||
        statistics.free_bytes != 128U ||
        statistics.largest_free_range_bytes != 128U) {
        return -1;
    }
    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_STATUS_OK ||
        mock.destroy_stream(mock.state, first_stream) != 0 ||
        mock.destroy_stream(mock.state, second_stream) != 0) {
        return -1;
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_backend_destroy(&mock);
    return 0;
}

typedef struct DrainingRequest {
    ShadowSpillRuntime *runtime;
    ShadowSpillBackendStream stream;
    ShadowSpillStatus status;
    ShadowSpillAllocation allocation;
} DrainingRequest;

static void *allocate_into_a_full_pool(void *pointer) {
    DrainingRequest *request = pointer;
    request->status = shadowspill_memory_pool_allocate(
        request->runtime, 0U, 128U, 1U, request->stream, &request->allocation
    );
    return NULL;
}

/*
 * A wait ends when the capacity it was promised stops coming.
 *
 * An allocator that cannot fit waits only because something is on its way to
 * freeing room. What it watches, though, is the pool's capacity epoch, and
 * the epoch moves only when a range is actually freed. Work can drain
 * without freeing one -- and then the epoch never moves again, and a wait on
 * it alone is a wait for an event that has already happened. Neither of the
 * remaining exits helps: an allocation failure is latched by the caller, not
 * the runtime, and the worker is not stopping.
 *
 * So this drains the release source without freeing anything and requires
 * the waiter to answer. The answer is NO_PROGRESS: the pool is full and
 * nothing is coming. Before the wait re-read its release source this hung
 * forever at full tilt on one core, and a bounded join is what tells the two
 * apart.
 */
static int a_wait_ends_when_its_release_source_drains(void) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig mock_config = {
        .event_delay_nanoseconds = 2000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(&mock, 128U, 1U, 1U, 10000U, &topology);
    if (shadowspill_runtime_create(&topology.runtime, &runtime) !=
        SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillAllocation held = {0};
    if (mock.create_stream(mock.state, &stream) != 0 ||
        shadowspill_memory_pool_allocate(
            runtime, 0U, 128U, 1U, stream, &held
        ) != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    /* The pool by index: its release accounting is what a waiting allocator
       watches, and no public call reaches it. */
    if (runtime->pools == NULL || runtime->pool_count == 0U) {
        return -1;
    }
    ShadowSpillMemoryPool *pool = &runtime->pools[0];
    /* Something is on its way, so the next request waits rather than giving
       up. Nothing will free a range, which is the case that hung. */
    (void)atomic_fetch_add_explicit(
        &pool->pending_retirements, 1U, memory_order_release
    );

    DrainingRequest request = {
        .runtime = runtime,
        .stream = stream,
        .status = SHADOWSPILL_STATUS_INVALID_STATE,
    };
    pthread_t thread;
    if (pthread_create(&thread, NULL, allocate_into_a_full_pool, &request) != 0) {
        return -1;
    }
    /* Wait for it to reach the wait, so the drain below is what releases it
       rather than a race with entering. */
    for (uint32_t spins = 0U; spins < 100000U; ++spins) {
        if (pool->blocked_allocators != 0U) {
            break;
        }
        struct timespec pause = {0, 1000000L};
        nanosleep(&pause, NULL);
    }
    const int reached_the_wait = pool->blocked_allocators != 0U;
    /* It drains, and no range is freed: the epoch does not move. */
    (void)atomic_store_explicit(
        &pool->pending_retirements, 0U, memory_order_release
    );

    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += 10;
    const int join_status = pthread_timedjoin_np(thread, NULL, &deadline);
    if (join_status != 0) {
        fprintf(
            stderr,
            "the allocator never answered: it is still waiting for capacity "
            "that stopped coming\n"
        );
        return -1;
    }
    if (!reached_the_wait ||
        request.status != SHADOWSPILL_STATUS_NO_PROGRESS) {
        fprintf(
            stderr,
            "draining release source: reached_wait=%d status=%u "
            "(expected NO_PROGRESS=%u)\n",
            reached_the_wait,
            (unsigned)request.status,
            (unsigned)SHADOWSPILL_STATUS_NO_PROGRESS
        );
        return -1;
    }
    (void)shadowspill_memory_pool_free(runtime, 0U, held.allocation_id, stream);
    (void)shadowspill_runtime_close(runtime);
    (void)mock.destroy_stream(mock.state, stream);
    shadowspill_runtime_destroy(runtime);
    shadowspill_backend_destroy(&mock);
    return 0;
}

int main(void) {
    if (a_blocked_allocator_resumes_when_the_free_lands() != 0 ||
        a_wait_ends_when_its_release_source_drains() != 0) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
