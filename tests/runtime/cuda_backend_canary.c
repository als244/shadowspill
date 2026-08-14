#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend_cuda.h>
#include <shadowspill/runtime.h>

#define PAYLOAD_BYTES (4U << 20U)
#define FAIL(stage_)                                                         \
    do {                                                                     \
        fprintf(stderr, "cuda backend canary failed at %s\n", (stage_));   \
        return EXIT_FAILURE;                                                 \
    } while (0)

int main(void) {
    const ShadowSpillCudaBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_CUDA_BACKEND_ABI_VERSION,
        .device_ordinal = 0,
    };
    ShadowSpillCudaBackend *cuda = NULL;
    if (shadowspill_cuda_backend_create(&backend_config, &cuda) != 0) {
        FAIL("backend creation");
    }
    ShadowSpillCudaBackendCapabilities capabilities = {0};
    ShadowSpillBackend backend = shadowspill_cuda_backend_vtable(cuda);
    if (shadowspill_cuda_backend_capabilities(cuda, &capabilities) != 0 ||
        capabilities.abi_version != SHADOWSPILL_CUDA_BACKEND_ABI_VERSION ||
        capabilities.total_device_bytes == 0U ||
        capabilities.recommended_minimum_alignment == 0U ||
        !capabilities.process_memory_accounting) {
        FAIL("capability validation");
    }
    ShadowSpillCudaPhysicalMemory context_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &context_memory) != 0 ||
        context_memory.abi_version != SHADOWSPILL_CUDA_BACKEND_ABI_VERSION ||
        context_memory.process_bytes == 0U ||
        context_memory.device_used_bytes < context_memory.process_bytes ||
        context_memory.device_total_bytes < capabilities.total_device_bytes) {
        FAIL("initial physical-memory accounting");
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 8U << 20U,
        .spill_pool_bytes = 8U << 20U,
        .minimum_alignment = capabilities.recommended_minimum_alignment,
        .worker_poll_nanoseconds = 10000U,
        .backend = backend,
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        backend.create_stream(
            backend.context, SHADOWSPILL_TRANSFER_FETCH, &compute
        ) != 0 ||
        shadowspill_cuda_backend_seal_event_pool(cuda, 8U) != 0) {
        FAIL("runtime or stream initialization");
    }
    ShadowSpillCudaPhysicalMemory admitted_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &admitted_memory) != 0 ||
        admitted_memory.process_bytes <
            context_memory.process_bytes + runtime_config.execution_pool_bytes) {
        FAIL("admitted physical-memory accounting");
    }
    unsigned char *original = malloc(PAYLOAD_BYTES);
    unsigned char *restored = calloc(PAYLOAD_BYTES, 1U);
    if (original == NULL || restored == NULL) {
        FAIL("host payload allocation");
    }
    for (uint64_t index = 0U; index < PAYLOAD_BYTES; ++index) {
        original[index] = (unsigned char)((index * 17U + 29U) & 0xffU);
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = PAYLOAD_BYTES,
        .initial_version = 1U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    if (shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_write_spill_object(
            runtime, object.object_id, original, PAYLOAD_BYTES
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 0U, compute, NULL, 0U, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK) {
        FAIL("initial object registration or fetch submission");
    }
    const uint64_t input_object_id = object.object_id;
    ShadowSpillObjectBinding binding = {0};
    if (shadowspill_before_task(
            runtime,
            1U,
            compute,
            &input_object_id,
            1U,
            &binding,
            1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        binding.pointer == NULL || binding.authoritative_version != 1U) {
        FAIL("fetch readiness");
    }
    const ShadowSpillRuntimeAction offload = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillRuntimeStatus after_status = shadowspill_after_task(
        runtime, 1U, compute, NULL, 0U, &offload, 1U
    );
    const ShadowSpillRuntimeStatus idle_status = after_status ==
            SHADOWSPILL_RUNTIME_OK
        ? shadowspill_runtime_wait_idle(runtime)
        : after_status;
    const ShadowSpillRuntimeStatus read_status = idle_status ==
            SHADOWSPILL_RUNTIME_OK
        ? shadowspill_read_spill_object(
            runtime, object.object_id, restored, PAYLOAD_BYTES
        )
        : idle_status;
    if (after_status != SHADOWSPILL_RUNTIME_OK ||
        idle_status != SHADOWSPILL_RUNTIME_OK ||
        read_status != SHADOWSPILL_RUNTIME_OK ||
        memcmp(original, restored, PAYLOAD_BYTES) != 0) {
        ShadowSpillRuntimeFailure failure = {0};
        (void)shadowspill_runtime_failure(runtime, &failure);
        fprintf(
            stderr,
            "after=%u idle=%u read=%u failure=%u task=%llu object=%llu "
            "allocation=%llu\n",
            (unsigned)after_status,
            (unsigned)idle_status,
            (unsigned)read_status,
            (unsigned)failure.status,
            (unsigned long long)failure.task_id,
            (unsigned long long)failure.object_id,
            (unsigned long long)failure.allocation_id
        );
        fprintf(
            stderr,
            "cuda_error=%u nvml_error=%u\n",
            shadowspill_cuda_backend_last_error(cuda),
            shadowspill_cuda_backend_last_nvml_error(cuda)
        );
        FAIL("eviction completion or data validation");
    }
    ShadowSpillRuntimeStatistics runtime_statistics = {0};
    ShadowSpillCudaBackendStatistics cuda_statistics = {0};
    if (shadowspill_runtime_statistics(runtime, &runtime_statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        runtime_statistics.fetch_transfers != 1U ||
        runtime_statistics.evict_transfers != 1U ||
        runtime_statistics.bytes_fetched != PAYLOAD_BYTES ||
        runtime_statistics.bytes_evicted != PAYLOAD_BYTES) {
        FAIL("runtime statistics");
    }
    shadowspill_cuda_backend_statistics(cuda, &cuda_statistics);
    if (cuda_statistics.device_allocations != 1U ||
        cuda_statistics.pinned_host_allocations != 1U ||
        cuda_statistics.fetch_copies != 1U ||
        cuda_statistics.evict_copies != 1U ||
        cuda_statistics.bytes_fetched != PAYLOAD_BYTES ||
        cuda_statistics.bytes_evicted != PAYLOAD_BYTES) {
        FAIL("backend transfer statistics");
    }
    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_RUNTIME_OK ||
        backend.destroy_stream(backend.context, compute) != 0) {
        FAIL("runtime close");
    }
    shadowspill_cuda_backend_statistics(cuda, &cuda_statistics);
    if (cuda_statistics.device_frees != 1U ||
        cuda_statistics.pinned_host_frees != 1U ||
        cuda_statistics.streams_created != 3U ||
        cuda_statistics.streams_destroyed != 3U ||
        cuda_statistics.events_created != cuda_statistics.events_destroyed ||
        !cuda_statistics.event_pool_sealed ||
        cuda_statistics.event_pool_capacity < 8U ||
        cuda_statistics.event_pool_driver_creates !=
            cuda_statistics.event_pool_capacity ||
        cuda_statistics.event_pool_in_use != 0U ||
        cuda_statistics.event_pool_peak_in_use == 0U ||
        cuda_statistics.event_pool_growth_rejections != 0U ||
        shadowspill_cuda_backend_last_error(cuda) != 0U ||
        shadowspill_cuda_backend_last_nvml_error(cuda) != 0U) {
        FAIL("backend teardown statistics");
    }
    free(original);
    free(restored);
    shadowspill_runtime_destroy(runtime);
    shadowspill_cuda_backend_destroy(cuda);
    return EXIT_SUCCESS;
}
