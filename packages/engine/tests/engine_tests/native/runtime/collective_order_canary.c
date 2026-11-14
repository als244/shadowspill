/* Task dispatch and transformed transfers must have one host submission order.
 * This checks the existing boundary handshake without requiring NCCL or CUDA. */
#define _POSIX_C_SOURCE 200809L
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "runtime_test.h"

typedef struct Log {
    _Atomic uint32_t count;
    uint32_t entries[8];
} Log;

typedef struct Transfer {
    Log *log;
    uint32_t id;
    uint8_t delayed_direction;
} Transfer;

static void record(Log *log, uint32_t value) {
    const uint32_t index = atomic_fetch_add_explicit(
        &log->count, 1U, memory_order_relaxed
    );
    if (index < 8U) log->entries[index] = value;
}

static void context_noop(void *context) { (void)context; }

static int enqueue(
    void *context, const ShadowSpillBackend *backend,
    ShadowSpillBackendStream stream, uint8_t fetch,
    void *destination, const void *source,
    ShadowSpillTransferCopy copy, void *copy_context
) {
    Transfer *transfer = context;
    if (fetch == transfer->delayed_direction) {
        const struct timespec delay = {.tv_nsec = 2000000L};
        (void)nanosleep(&delay, NULL);
    }
    record(transfer->log, transfer->id * 10U + (fetch ? 0U : 1U));
    (void)backend; (void)stream;
    return copy(copy_context, destination, source, 8U);
}

static int ordered_submission(uint8_t delayed_direction) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig config = {0};
    if (shadowspill_mock_backend_create(&config, &mock) != 0) return -1;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = 0U;
    ShadowSpillBackendSignals signals = 0U;
    ShadowSpillBackendEvent done = 0U;
    uint64_t *gate = NULL;
    Log log = {0};
    Transfer contexts[2] = {
        {&log, 1U, delayed_direction}, {&log, 2U, delayed_direction},
    };
    int failed = shadowspill_test_create_runtime(
        &mock, 64U, 32U, 8U, 1000U, &runtime
    ) != SHADOWSPILL_STATUS_OK ||
        mock.create_stream(mock.state, &compute) != 0 ||
        mock.allocate_signals(mock.state, 1U, &signals, &gate) != 0 ||
        mock.create_event(mock.state, &done, 0U) != 0;
    for (uint32_t i = 0; i < 2U && !failed; ++i) {
        const ShadowSpillObjectTransfer transfer = {
            .spill_size_bytes = 8U, .collective_directions = 3U,
            .context = &contexts[i], .retain = context_noop,
            .release = context_noop, .enqueue = enqueue,
        };
        const ShadowSpillObjectDescription object = {
            .object_id = i + 1U, .size_bytes = 16U, .initial_pool_id = 1U,
            .retain_spill_copy = 1U, .initially_resident = 1U,
            .transfer = &transfer,
        };
        failed = shadowspill_register_object(runtime, &object) != SHADOWSPILL_STATUS_OK;
    }
    const ShadowSpillRuntimeAction initial = {
        .object_id = 1U, .kind = SHADOWSPILL_RUNTIME_FETCH,
    };
    /* Deliberately list the fetch first: collective evictions must be submitted
       before fetches in this boundary, regardless of device completion timing. */
    const ShadowSpillRuntimeAction actions[2] = {
        {.object_id = 2U, .kind = SHADOWSPILL_RUNTIME_FETCH},
        {.object_id = 1U, .kind = SHADOWSPILL_RUNTIME_EVICT},
    };
    const uint64_t input = 1U;
    const ShadowSpillObjectUpdate update = {.object_id = input, .version_delta = 1U};
    const ShadowSpillTaskDescription task = {
        .task_id = 1U, .input_object_ids = &input, .input_count = 1U,
        .updates = &update, .update_count = 1U,
        .actions = actions, .action_count = 2U,
    };
    ShadowSpillObjectBinding binding = {0};
    failed = failed ||
        shadowspill_test_admit_task(runtime, &task) != SHADOWSPILL_STATUS_OK ||
        shadowspill_test_submit_actions(runtime, 0U, compute, &initial, 1U) !=
            SHADOWSPILL_STATUS_OK ||
        atomic_load_explicit(&log.count, memory_order_relaxed) != 1U ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_STATUS_OK ||
        shadowspill_test_before_task(runtime, 1U, compute, &binding, 1U) !=
            SHADOWSPILL_STATUS_OK ||
        mock.wait_value(mock.state, compute, signals, 0U, 1U) != 0;
    if (!failed) {
        /* Stands for the explicit collective callable in the dispatch thread. */
        record(&log, 90U);
        failed = shadowspill_test_after_task(runtime, 1U, compute) !=
            SHADOWSPILL_STATUS_OK;
    }
    if (!failed) {
        record(&log, 91U);
        const uint32_t expected[] = {10U, 90U, 11U, 20U, 91U};
        failed = atomic_load_explicit(&log.count, memory_order_relaxed) != 5U;
        for (uint32_t i = 0; i < 5U; ++i) {
            failed = failed || log.entries[i] != expected[i];
        }
        int complete = 1;
        failed = failed || mock.record_event(mock.state, done, compute) != 0 ||
            mock.query_event(mock.state, done, &complete) != 0 || complete;
        /* The gate is still shut: acknowledgement means submitted, not finished.
           Independent GPU work may overlap the transfers and explicit task. */
    }
    if (gate != NULL) atomic_store_explicit((_Atomic uint64_t *)gate, 1U, memory_order_release);
    if (runtime != NULL) {
        failed = shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_STATUS_OK || failed;
        shadowspill_test_destroy_runtime(runtime);
    }
    if (done != 0U) mock.destroy_event(mock.state, done);
    if (compute != 0U) mock.destroy_stream(mock.state, compute);
    if (signals != 0U) mock.free_signals(mock.state, signals);
    shadowspill_backend_destroy(&mock);
    return failed ? -1 : 0;
}

int main(void) {
    for (uint8_t direction = 0U; direction < 2U; ++direction) {
        if (ordered_submission(direction) != 0) {
            fprintf(stderr, "collective submission order failed (delay=%u)\n", direction);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
