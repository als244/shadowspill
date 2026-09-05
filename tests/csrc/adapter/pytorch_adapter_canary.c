/*
 * Drives the PyTorch adapter's C entry points against the mock backend,
 * loaded by path the way the frontend loads a provider.
 *
 * No accelerator and no libtorch are involved: the storage operators, the
 * C++ exception path and the message it carries are the Python canaries' to
 * cover. What this certifies is everything else -- every call the adapter
 * refactor moves, in the order a process makes them -- so a split that
 * changes behaviour fails here, on a CPU, before any device sees it.
 *
 * Every stream handle passed is 0: the backend's default stream, the one
 * stream a caller with no streams of its own can name.
 */
#include <shadowspill/pytorch_adapter.h>
#include <shadowspill/runtime.h>

#include "runtime_test.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MIB(count) ((uint64_t)(count) << 20U)
#define DEVICE_BUDGET_BYTES MIB(64)
#define PROVIDER_HEADROOM_BYTES MIB(4)
#define DEFAULT_STREAM ((void *)0)

#define REQUIRE(condition, message)                                          \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "pytorch adapter canary: %s\n", message);         \
            return -1;                                                         \
        }                                                                      \
    } while (0)

enum {
    DEVICE_POOL = 0,
    SPILL_POOL = 1,
    FETCH_ROUTE = 0,
    EVICT_ROUTE = 1,
    OBJECT_ID = 7,
};

/* Nothing is bound yet, and every call says so the same way. */
static int before_bootstrap(void) {
    uintptr_t handle = 1U;
    ShadowSpillPytorchAdapterStatistics statistics = {0};
    REQUIRE(
        shadowspill_pytorch_runtime_handle(&handle) ==
                SHADOWSPILL_STATUS_CLOSED &&
            handle == 0U,
        "runtime_handle before bootstrap must say CLOSED"
    );
    REQUIRE(
        shadowspill_pytorch_allocator_statistics(&statistics) ==
            SHADOWSPILL_STATUS_CLOSED,
        "statistics before bootstrap must say CLOSED"
    );
    REQUIRE(
        shadowspill_pytorch_allocator_close() == SHADOWSPILL_STATUS_CLOSED,
        "close before bootstrap must say CLOSED"
    );
    return 0;
}

static int bootstrap(const char *backend_library) {
    const ShadowSpillPytorchPoolConfig pools[2] = {
        {.pool_id = DEVICE_POOL, .kind = SHADOWSPILL_POOL_DEVICE},
        {
            .pool_id = SPILL_POOL,
            .kind = SHADOWSPILL_POOL_PINNED_HOST,
            .capacity_bytes = MIB(16),
        },
    };
    const ShadowSpillPytorchRouteConfig routes[2] = {
        {
            .route_id = FETCH_ROUTE,
            .source_pool_id = SPILL_POOL,
            .destination_pool_id = DEVICE_POOL,
            .name = "fetch",
        },
        {
            .route_id = EVICT_ROUTE,
            .source_pool_id = DEVICE_POOL,
            .destination_pool_id = SPILL_POOL,
            .name = "evict",
        },
    };
    ShadowSpillPytorchAdapterConfig config = {
        .abi_version = 0U,
        .device_ordinal = 0,
        .device_budget_bytes = DEVICE_BUDGET_BYTES,
        .provider_headroom_bytes = PROVIDER_HEADROOM_BYTES,
        .allocator_pool_id = DEVICE_POOL,
        .pools = pools,
        .pool_count = 2U,
        .routes = routes,
        .route_count = 2U,
        .worker_poll_nanoseconds = 100000U,
        .backend_library = backend_library,
    };
    REQUIRE(
        shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT,
        "a wrong contract version must be refused"
    );
    config.abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION;
    REQUIRE(
        shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_OK,
        "bootstrap failed"
    );
    REQUIRE(
        shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_INVALID_STATE,
        "a second bootstrap must be refused"
    );
    ShadowSpillPytorchAdapterCapabilities capabilities = {0};
    REQUIRE(
        shadowspill_pytorch_adapter_capabilities(&capabilities) ==
                SHADOWSPILL_STATUS_OK &&
            capabilities.abi_version ==
                SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION &&
            capabilities.runtime_abi_version == SHADOWSPILL_ABI_VERSION &&
            capabilities.backend_abi_version ==
                SHADOWSPILL_BACKEND_ABI_VERSION,
        "capabilities do not name the three contracts"
    );
    return 0;
}

/* The physical-memory ledger: opened by bootstrap, checked, then sealed. */
static int ledger(void) {
    ShadowSpillPytorchPhysicalAdmission admission = {0};
    REQUIRE(
        shadowspill_pytorch_physical_admission(&admission) ==
            SHADOWSPILL_STATUS_OK,
        "physical admission is unreadable"
    );
    REQUIRE(
        admission.device_budget_bytes == DEVICE_BUDGET_BYTES &&
            admission.provider_headroom_bytes == PROVIDER_HEADROOM_BYTES &&
            admission.allocator_pool_bytes ==
                DEVICE_BUDGET_BYTES - PROVIDER_HEADROOM_BYTES &&
            admission.pool_count == 2U &&
            admission.allocator_pool_id == DEVICE_POOL &&
            admission.device_total_bytes == (UINT64_C(1) << 40U),
        "physical admission does not describe the bootstrap"
    );
    ShadowSpillBackendPhysicalMemory memory = {0};
    REQUIRE(
        shadowspill_pytorch_physical_memory(&memory) ==
                SHADOWSPILL_STATUS_OK &&
            memory.device_total_bytes == admission.device_total_bytes,
        "physical memory is unreadable"
    );
    REQUIRE(
        shadowspill_pytorch_check_physical_budget() == SHADOWSPILL_STATUS_OK,
        "the budget check failed with nothing in use"
    );
    REQUIRE(
        shadowspill_pytorch_seal_physical_budget(
            PROVIDER_HEADROOM_BYTES, 8U
        ) == SHADOWSPILL_STATUS_OK,
        "sealing within the reserved headroom failed"
    );
    ShadowSpillPytorchAdapterStatistics statistics = {0};
    REQUIRE(
        shadowspill_pytorch_allocator_statistics(&statistics) ==
                SHADOWSPILL_STATUS_OK &&
            statistics.physical_budget_sealed == 1U &&
            statistics.physical_checks == 3U,
        "the ledger did not count bootstrap, the check and the seal"
    );
    return 0;
}

/* The mock supplies no profiler; the adapter reports that as CLOSED today. */
static int profiler(void) {
    REQUIRE(
        shadowspill_pytorch_profile_range_begin("shadowspill.canary") == 0U,
        "a range began while annotations were disabled"
    );
    shadowspill_pytorch_profile_range_end(0U);
    REQUIRE(
        shadowspill_pytorch_profiler_annotations_set(1U) ==
            SHADOWSPILL_STATUS_CLOSED,
        "annotations on a backend without a profiler must be refused"
    );
    return 0;
}

/* The three callbacks PyTorch's pluggable allocator makes, and the query. */
static int allocator(ShadowSpillRuntime *runtime) {
    REQUIRE(
        shadowspill_pytorch_backend_malloc(0, 0, DEFAULT_STREAM) == NULL,
        "a zero-byte request must return NULL"
    );
    void *const block =
        shadowspill_pytorch_backend_malloc((ptrdiff_t)MIB(1), 0, DEFAULT_STREAM);
    REQUIRE(block != NULL, "a 1 MiB request failed");
    ShadowSpillAllocation allocation = {0};
    REQUIRE(
        shadowspill_pytorch_allocation_for_pointer(
            (uint64_t)(uintptr_t)block, &allocation
        ) == SHADOWSPILL_STATUS_OK &&
            allocation.pointer == block &&
            allocation.pool_id == DEVICE_POOL &&
            allocation.requested_bytes == MIB(1),
        "the pointer query does not describe the block"
    );
    shadowspill_pytorch_backend_record_stream(block, DEFAULT_STREAM);
    shadowspill_pytorch_backend_free(block, (size_t)MIB(1), 0, DEFAULT_STREAM);
    REQUIRE(
        shadowspill_runtime_wait_idle(runtime) == SHADOWSPILL_STATUS_OK,
        "wait_idle after the free failed"
    );
    ShadowSpillPytorchAdapterStatistics statistics = {0};
    REQUIRE(
        shadowspill_pytorch_allocator_statistics(&statistics) ==
                SHADOWSPILL_STATUS_OK &&
            statistics.allocation_callbacks == 2U &&
            statistics.zero_size_allocation_callbacks == 1U &&
            statistics.free_callbacks == 1U &&
            statistics.record_stream_callbacks == 1U &&
            statistics.pointer_lookup_failures == 0U &&
            statistics.callback_failures == 0U,
        "the callback counters do not match the calls made"
    );
    return 0;
}

/* A spill-resident object, its lease validated by address and size. */
static int objects(ShadowSpillRuntime *runtime) {
    unsigned char payload[4096];
    unsigned char back[4096] = {0};
    for (uint32_t index = 0U; index < sizeof(payload); ++index) {
        payload[index] = (unsigned char)(index * 7U);
    }
    const ShadowSpillObjectDescription object = {
        .object_id = OBJECT_ID,
        .size_bytes = sizeof(payload),
        .initial_pool_id = SPILL_POOL,
        .retain_spill_copy = 1U,
        .initially_resident = 1U,
    };
    REQUIRE(
        shadowspill_register_object(runtime, &object) == SHADOWSPILL_STATUS_OK &&
            shadowspill_write_object(
                runtime, OBJECT_ID, SPILL_POOL, payload, sizeof(payload)
            ) == SHADOWSPILL_STATUS_OK,
        "registering a spill-resident object failed"
    );
    ShadowSpillObjectLocationSnapshot location = {0};
    REQUIRE(
        shadowspill_object_location_snapshot(
            runtime, OBJECT_ID, SPILL_POOL, &location
        ) == SHADOWSPILL_STATUS_OK &&
            location.has_lease && location.current &&
            location.pointer != NULL,
        "the registered object has no current spill lease"
    );
    const uint64_t address = (uint64_t)(uintptr_t)location.pointer;
    REQUIRE(
        shadowspill_pytorch_validate_object_binding(
            SPILL_POOL, OBJECT_ID, address, sizeof(payload)
        ) == SHADOWSPILL_STATUS_OK,
        "the lease's own address failed validation"
    );
    REQUIRE(
        shadowspill_pytorch_validate_object_binding(
            SPILL_POOL, OBJECT_ID, address + 1U, sizeof(payload)
        ) == SHADOWSPILL_STATUS_INVALID_STATE,
        "a wrong address passed validation"
    );
    REQUIRE(
        shadowspill_pytorch_validate_object_binding(
            SPILL_POOL, OBJECT_ID, address, sizeof(payload) - 1U
        ) == SHADOWSPILL_STATUS_INVALID_STATE,
        "a wrong size passed validation"
    );
    REQUIRE(
        shadowspill_read_object(
            runtime, OBJECT_ID, SPILL_POOL, back, sizeof(back)
        ) == SHADOWSPILL_STATUS_OK &&
            memcmp(payload, back, sizeof(back)) == 0,
        "the object does not read back what was written"
    );
    const ShadowSpillObjectDescription placeholder = {
        .object_id = OBJECT_ID + 1U,
        .size_bytes = 512U,
    };
    REQUIRE(
        shadowspill_register_object(runtime, &placeholder) ==
            SHADOWSPILL_STATUS_OK,
        "registering a placeholder failed"
    );
    REQUIRE(
        shadowspill_unregister_object(runtime, OBJECT_ID + 1U) ==
            SHADOWSPILL_STATUS_OK,
        "unregistering the placeholder failed"
    );
    return 0;
}

/* Allocation scopes outside any task, ended and aborted. */
static int scopes(ShadowSpillRuntime *runtime) {
    /* The frontend's profiling scopes sit above 1 << 62. */
    const uint64_t scope_id = (UINT64_C(1) << 62U) | 3U;
    REQUIRE(
        shadowspill_pytorch_allocation_scope_begin(scope_id) ==
            SHADOWSPILL_STATUS_OK,
        "opening an allocation scope failed"
    );
    REQUIRE(
        shadowspill_pytorch_allocation_scope_begin(scope_id) ==
            SHADOWSPILL_STATUS_INVALID_STATE,
        "a nested scope must be refused"
    );
    void *const probe = shadowspill_pytorch_backend_malloc(4096, 0, DEFAULT_STREAM);
    REQUIRE(probe != NULL, "a request inside a scope failed");
    shadowspill_pytorch_backend_free(probe, 4096U, 0, DEFAULT_STREAM);
    REQUIRE(
        shadowspill_pytorch_allocation_scope_end(scope_id, 0U) ==
            SHADOWSPILL_STATUS_OK,
        "closing the scope failed"
    );
    REQUIRE(
        shadowspill_pytorch_allocation_scope_begin(scope_id + 1U) ==
            SHADOWSPILL_STATUS_OK,
        "opening a second scope failed"
    );
    shadowspill_pytorch_allocation_scope_abort();
    REQUIRE(
        shadowspill_runtime_wait_idle(runtime) == SHADOWSPILL_STATUS_OK,
        "wait_idle after the scopes failed"
    );
    return 0;
}

/* The task boundary: before, an allocation, after; then abort. */
static int tasks(ShadowSpillRuntime *runtime) {
    const ShadowSpillTaskDescription task = {
        .task_id = 1U,
        .trace_label = "execution_000001.canary",
    };
    REQUIRE(
        shadowspill_test_admit_task(runtime, &task) == SHADOWSPILL_STATUS_OK,
        "admitting a task with no inputs failed"
    );
    const uintptr_t handle =
        (uintptr_t)shadowspill_test_task_handle(runtime, task.task_id);
    REQUIRE(handle != 0U, "the admitted task has no handle");
    const ShadowSpillObjectBinding *bindings = NULL;
    uint32_t binding_count = 99U;
    REQUIRE(
        shadowspill_pytorch_before_task_handle(
            handle, 0U, &bindings, &binding_count
        ) == SHADOWSPILL_STATUS_OK &&
            binding_count == 0U,
        "before_task failed for a task with no inputs"
    );
    REQUIRE(
        shadowspill_pytorch_before_task_handle(
            handle, 0U, &bindings, &binding_count
        ) == SHADOWSPILL_STATUS_INVALID_STATE,
        "a nested before_task must be refused"
    );
    void *const scratch = shadowspill_pytorch_backend_malloc(1024, 0, DEFAULT_STREAM);
    REQUIRE(scratch != NULL, "a request inside a task failed");
    shadowspill_pytorch_backend_free(scratch, 1024U, 0, DEFAULT_STREAM);
    REQUIRE(
        shadowspill_pytorch_after_task_handle(handle, 0U) ==
            SHADOWSPILL_STATUS_OK,
        "after_task failed"
    );
    REQUIRE(
        shadowspill_pytorch_before_task_handle(
            handle, 0U, &bindings, &binding_count
        ) == SHADOWSPILL_STATUS_OK,
        "the task could not run a second time"
    );
    REQUIRE(
        shadowspill_pytorch_abort_task_handle(handle) == SHADOWSPILL_STATUS_OK,
        "abort_task failed"
    );
    REQUIRE(
        shadowspill_runtime_wait_idle(runtime) == SHADOWSPILL_STATUS_OK,
        "wait_idle after the task failed"
    );
    return 0;
}

/* The pre-task placement batch, then an acquisition handed to the caller. */
static int placement_and_acquisition(ShadowSpillRuntime *runtime) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 1);
    REQUIRE(record != NULL, "no test plan");
    REQUIRE(
        shadowspill_test_bind_object(record, OBJECT_ID) == SHADOWSPILL_STATUS_OK,
        "binding the object to the plan failed"
    );
    const ShadowSpillRuntimeAction fetch = {
        .object_id = OBJECT_ID,
        .kind = SHADOWSPILL_RUNTIME_FETCH,
    };
    const ShadowSpillActionBatchHandle *batch = NULL;
    REQUIRE(
        shadowspill_plan_admit_action_batch(
            record->plan, UINT64_C(1) << 60U, &fetch, 1U, &batch
        ) == SHADOWSPILL_STATUS_OK &&
            batch != NULL,
        "admitting the placement batch failed"
    );
    REQUIRE(
        shadowspill_pytorch_submit_action_batch_handle(0U, 0U) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT,
        "a null batch must be refused"
    );
    REQUIRE(
        shadowspill_pytorch_submit_action_batch_handle((uintptr_t)batch, 0U) ==
            SHADOWSPILL_STATUS_OK,
        "submitting the placement batch failed"
    );
    REQUIRE(
        shadowspill_runtime_wait_idle(runtime) == SHADOWSPILL_STATUS_OK,
        "wait_idle after the batch failed"
    );
    const uint64_t object_id = OBJECT_ID;
    const ShadowSpillObjectAcquisitionHandle *acquisition = NULL;
    REQUIRE(
        shadowspill_plan_admit_object_acquisition(
            record->plan, &object_id, 1U, &acquisition
        ) == SHADOWSPILL_STATUS_OK &&
            acquisition != NULL,
        "admitting the acquisition failed"
    );
    ShadowSpillObjectBinding binding = {0};
    REQUIRE(
        shadowspill_pytorch_acquire_objects_handle(
            (uintptr_t)acquisition, 0U, &binding, 1U
        ) == SHADOWSPILL_STATUS_OK &&
            binding.object_id == OBJECT_ID && binding.pointer != NULL,
        "acquiring the object for the default stream failed"
    );
    ShadowSpillAllocation caller = {0};
    REQUIRE(
        shadowspill_pytorch_transfer_acquired_object_to_caller(
            (uintptr_t)acquisition,
            0U,
            0U,
            (uint64_t)(uintptr_t)binding.pointer,
            binding.generation,
            binding.allocation_id,
            &caller
        ) == SHADOWSPILL_STATUS_OK &&
            caller.pointer == binding.pointer &&
            caller.allocation_id == binding.allocation_id,
        "handing the acquired object to the caller failed"
    );
    REQUIRE(
        shadowspill_pytorch_release_caller_allocation(
            caller.allocation_id, 0U
        ) == SHADOWSPILL_STATUS_OK,
        "releasing the caller's allocation failed"
    );
    REQUIRE(
        shadowspill_runtime_wait_idle(runtime) == SHADOWSPILL_STATUS_OK,
        "wait_idle after the handoff failed"
    );
    return 0;
}

/* Transfer calibration through the adapter, and the profiles it leaves. */
static int calibration(ShadowSpillRuntime *runtime) {
    const ShadowSpillTransferRouteKey keys[2] = {
        {.source_pool_id = SPILL_POOL, .destination_pool_id = DEVICE_POOL},
        {.source_pool_id = DEVICE_POOL, .destination_pool_id = SPILL_POOL},
    };
    const ShadowSpillTransferCalibrationConfig config = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .small_copy_bytes = 4096U,
        .large_copy_bytes = MIB(1),
        .warmup_copies = 1U,
        .measured_copies = 2U,
        .provenance = SHADOWSPILL_TRANSFER_PROFILE_RECALIBRATION,
    };
    REQUIRE(
        shadowspill_runtime_calibrate_transfer_capabilities(
            runtime, &config, keys, 2U
        ) == SHADOWSPILL_STATUS_OK,
        "transfer calibration failed"
    );
    /* The runtime keeps one profile per ordered pool pair: a 2 x 2 matrix,
     * indexed source * pool_count + destination. */
    uint32_t count = 0U;
    uint64_t generation = 0U;
    REQUIRE(
        shadowspill_runtime_transfer_profiles(
            runtime, NULL, 0U, &count, &generation
        ) == SHADOWSPILL_STATUS_OK &&
            count == 4U,
        "the profile count is not the pool-pair matrix"
    );
    ShadowSpillTransferProfile profiles[4] = {{0}};
    REQUIRE(
        shadowspill_runtime_transfer_profiles(
            runtime, profiles, 4U, &count, &generation
        ) == SHADOWSPILL_STATUS_OK &&
            count == 4U && generation == 1U,
        "the profiles are unreadable after one calibration"
    );
    const ShadowSpillTransferProfile *fetch = &profiles[SPILL_POOL * 2 + DEVICE_POOL];
    const ShadowSpillTransferProfile *evict = &profiles[DEVICE_POOL * 2 + SPILL_POOL];
    REQUIRE(
        fetch->calibrated && evict->calibrated &&
            fetch->bandwidth_bytes_per_second != 0U &&
            evict->bandwidth_bytes_per_second != 0U,
        "the two routes are not calibrated"
    );
    return 0;
}

/* A refusal the adapter makes itself, kept as the first failure. */
static int failure(void) {
    shadowspill_pytorch_backend_free(
        (void *)(uintptr_t)0x10U, 16U, 1, DEFAULT_STREAM
    );
    ShadowSpillPytorchAdapterFailure latched = {0};
    REQUIRE(
        shadowspill_pytorch_allocator_failure(&latched) ==
                SHADOWSPILL_STATUS_INVALID_ARGUMENT &&
            latched.status == SHADOWSPILL_STATUS_INVALID_ARGUMENT &&
            latched.device_ordinal == 1 && latched.address == 0x10U &&
            latched.requested_bytes == 16U,
        "the wrong-device free was not latched as INVALID_ARGUMENT"
    );
    REQUIRE(
        shadowspill_pytorch_recover_no_progress() != SHADOWSPILL_STATUS_OK,
        "recovery must be refused when the latched failure is not NO_PROGRESS"
    );
    REQUIRE(
        shadowspill_pytorch_seal_physical_budget(DEVICE_BUDGET_BYTES, 8U) ==
            SHADOWSPILL_STATUS_PLAN_VIOLATION,
        "sealing with headroom above the reservation must be refused"
    );
    REQUIRE(
        shadowspill_pytorch_allocator_failure(&latched) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT,
        "the first failure was overwritten by a later one"
    );
    ShadowSpillPytorchAdapterStatistics statistics = {0};
    REQUIRE(
        shadowspill_pytorch_allocator_statistics(&statistics) ==
                SHADOWSPILL_STATUS_OK &&
            statistics.callback_failures == 1U,
        "the callback failure was not counted"
    );
    return 0;
}

/* Close is deterministic and idempotent, and everything after it says so. */
static int close_adapter(ShadowSpillRuntime *runtime) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 0);
    REQUIRE(record != NULL, "the test plan is gone");
    REQUIRE(
        shadowspill_plan_close(record->plan) == SHADOWSPILL_STATUS_OK,
        "closing the test plan failed"
    );
    shadowspill_plan_destroy(record->plan);
    record->plan = NULL;
    record->runtime = NULL;
    record->task_count = 0U;
    REQUIRE(
        shadowspill_unregister_object(runtime, OBJECT_ID) ==
            SHADOWSPILL_STATUS_OK,
        "unregistering the object failed"
    );
    REQUIRE(
        shadowspill_runtime_wait_idle(runtime) == SHADOWSPILL_STATUS_OK,
        "wait_idle before close failed"
    );
    REQUIRE(
        shadowspill_pytorch_allocator_close() == SHADOWSPILL_STATUS_OK,
        "close failed"
    );
    REQUIRE(
        shadowspill_pytorch_allocator_close() == SHADOWSPILL_STATUS_OK,
        "a second close must be idempotent"
    );
    uintptr_t handle = 1U;
    REQUIRE(
        shadowspill_pytorch_runtime_handle(&handle) ==
                SHADOWSPILL_STATUS_CLOSED &&
            handle == 0U,
        "runtime_handle after close must say CLOSED"
    );
    ShadowSpillPytorchAdapterStatistics statistics = {0};
    REQUIRE(
        shadowspill_pytorch_allocator_statistics(&statistics) ==
            SHADOWSPILL_STATUS_CLOSED,
        "statistics after close must say CLOSED"
    );
    REQUIRE(
        shadowspill_pytorch_validate_object_binding(
            SPILL_POOL, OBJECT_ID, 0U, 0U
        ) == SHADOWSPILL_STATUS_CLOSED,
        "validation after close must say CLOSED"
    );
    ShadowSpillBackendPhysicalMemory memory = {0};
    REQUIRE(
        shadowspill_pytorch_physical_memory(&memory) ==
            SHADOWSPILL_STATUS_CLOSED,
        "physical memory after close must say CLOSED"
    );
    /* A callback after close returns without touching the released runtime. */
    shadowspill_pytorch_backend_free(
        (void *)(uintptr_t)0x10U, 16U, 0, DEFAULT_STREAM
    );
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(
            stderr,
            "usage: pytorch_adapter_canary <mock backend shared object>\n"
        );
        return 1;
    }
    if (before_bootstrap() != 0 || bootstrap(argv[1]) != 0) {
        return 1;
    }
    uintptr_t handle = 0U;
    if (shadowspill_pytorch_runtime_handle(&handle) != SHADOWSPILL_STATUS_OK ||
        handle == 0U) {
        fprintf(stderr, "pytorch adapter canary: no runtime after bootstrap\n");
        return 1;
    }
    ShadowSpillRuntime *const runtime = (ShadowSpillRuntime *)handle;
    if (ledger() != 0 || profiler() != 0 || allocator(runtime) != 0 ||
        objects(runtime) != 0 || scopes(runtime) != 0 || tasks(runtime) != 0 ||
        placement_and_acquisition(runtime) != 0 || calibration(runtime) != 0 ||
        failure() != 0 ||
        close_adapter(runtime) != 0) {
        return 1;
    }
    return 0;
}
