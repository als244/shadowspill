#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend_cuda.h>
#include <shadowspill/runtime.h>

#define PAYLOAD_BYTES (4U << 20U)

int main(void) {
    const ShadowSpillCudaBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_CUDA_BACKEND_ABI_VERSION,
        .device_ordinal = 0,
    };
    ShadowSpillCudaBackend *cuda = NULL;
    if (shadowspill_cuda_backend_create(&backend_config, &cuda) != 0) {
        return EXIT_FAILURE;
    }
    ShadowSpillCudaBackendCapabilities capabilities = {0};
    ShadowSpillBackend backend = shadowspill_cuda_backend_vtable(cuda);
    if (shadowspill_cuda_backend_capabilities(cuda, &capabilities) != 0 ||
        capabilities.abi_version != SHADOWSPILL_CUDA_BACKEND_ABI_VERSION ||
        capabilities.total_device_bytes == 0U ||
        capabilities.recommended_minimum_alignment == 0U ||
        !capabilities.process_memory_accounting) {
        return EXIT_FAILURE;
    }
    ShadowSpillCudaPhysicalMemory context_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &context_memory) != 0 ||
        context_memory.abi_version != SHADOWSPILL_CUDA_BACKEND_ABI_VERSION ||
        context_memory.process_bytes == 0U ||
        context_memory.device_used_bytes < context_memory.process_bytes ||
        context_memory.device_total_bytes < capabilities.total_device_bytes) {
        return EXIT_FAILURE;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .device_slab_bytes = 8U << 20U,
        .host_arena_bytes = 8U << 20U,
        .minimum_alignment = capabilities.recommended_minimum_alignment,
        .progress_poll_nanoseconds = 10000U,
        .backend = backend,
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        backend.create_stream(
            backend.context, SHADOWSPILL_TRANSFER_TO_DEVICE, &compute
        ) != 0) {
        return EXIT_FAILURE;
    }
    ShadowSpillCudaPhysicalMemory admitted_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &admitted_memory) != 0 ||
        admitted_memory.process_bytes <
            context_memory.process_bytes + runtime_config.device_slab_bytes) {
        return EXIT_FAILURE;
    }
    unsigned char *original = malloc(PAYLOAD_BYTES);
    unsigned char *restored = calloc(PAYLOAD_BYTES, 1U);
    if (original == NULL || restored == NULL) {
        return EXIT_FAILURE;
    }
    for (uint64_t index = 0U; index < PAYLOAD_BYTES; ++index) {
        original[index] = (unsigned char)((index * 17U + 29U) & 0xffU);
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = PAYLOAD_BYTES,
        .initial_version = 1U,
        .retain_host_backing = 1U,
        .initially_host_resident = 1U,
    };
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    if (shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_write_host_object(
            runtime, object.object_id, original, PAYLOAD_BYTES
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 0U, compute, NULL, 0U, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
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
        return EXIT_FAILURE;
    }
    const ShadowSpillRuntimeAction offload = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    if (shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, &offload, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_read_host_object(
            runtime, object.object_id, restored, PAYLOAD_BYTES
        ) != SHADOWSPILL_RUNTIME_OK ||
        memcmp(original, restored, PAYLOAD_BYTES) != 0) {
        return EXIT_FAILURE;
    }
    ShadowSpillRuntimeStatistics runtime_statistics = {0};
    ShadowSpillCudaBackendStatistics cuda_statistics = {0};
    if (shadowspill_runtime_statistics(runtime, &runtime_statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        runtime_statistics.transfers_to_device != 1U ||
        runtime_statistics.transfers_to_host != 1U ||
        runtime_statistics.bytes_to_device != PAYLOAD_BYTES ||
        runtime_statistics.bytes_to_host != PAYLOAD_BYTES) {
        return EXIT_FAILURE;
    }
    shadowspill_cuda_backend_statistics(cuda, &cuda_statistics);
    if (cuda_statistics.device_allocations != 1U ||
        cuda_statistics.pinned_host_allocations != 1U ||
        cuda_statistics.copies_to_device != 1U ||
        cuda_statistics.copies_to_host != 1U ||
        cuda_statistics.bytes_to_device != PAYLOAD_BYTES ||
        cuda_statistics.bytes_to_host != PAYLOAD_BYTES) {
        return EXIT_FAILURE;
    }
    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_RUNTIME_OK ||
        backend.destroy_stream(backend.context, compute) != 0) {
        return EXIT_FAILURE;
    }
    shadowspill_cuda_backend_statistics(cuda, &cuda_statistics);
    if (cuda_statistics.device_frees != 1U ||
        cuda_statistics.pinned_host_frees != 1U ||
        cuda_statistics.streams_created != 3U ||
        cuda_statistics.streams_destroyed != 3U ||
        cuda_statistics.events_created != cuda_statistics.events_destroyed ||
        shadowspill_cuda_backend_last_error(cuda) != 0U ||
        shadowspill_cuda_backend_last_nvml_error(cuda) != 0U) {
        return EXIT_FAILURE;
    }
    free(original);
    free(restored);
    shadowspill_runtime_destroy(runtime);
    shadowspill_cuda_backend_destroy(cuda);
    return EXIT_SUCCESS;
}
