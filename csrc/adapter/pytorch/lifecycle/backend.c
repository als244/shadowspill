#include "internal.h"

#include <dlfcn.h>

ShadowSpillStatus shadowspill_pytorch_backend_load(
    const char *path,
    int32_t device_ordinal,
    ShadowSpillPytorchLoadedBackend *loaded
) {
    void *const library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    union {
        void *object;
        ShadowSpillBackendCreate create;
    } create = {.object = dlsym(library, SHADOWSPILL_BACKEND_CREATE_SYMBOL)};
    union {
        void *object;
        ShadowSpillBackendDestroy destroy;
    } destroy = {.object = dlsym(library, SHADOWSPILL_BACKEND_DESTROY_SYMBOL)};
    if (create.object == NULL || destroy.object == NULL) {
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    const ShadowSpillBackendConfig config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .device_ordinal = device_ordinal,
    };
    *loaded = (ShadowSpillPytorchLoadedBackend){
        .library = library,
        .destroy = destroy.destroy,
    };
    if (create.create(&config, &loaded->table) != 0 ||
        !shadowspill_backend_is_valid(&loaded->table)) {
        shadowspill_pytorch_backend_unload(loaded);
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    return SHADOWSPILL_STATUS_OK;
}

void shadowspill_pytorch_backend_unload(
    ShadowSpillPytorchLoadedBackend *loaded
) {
    if (loaded->table.state != NULL && loaded->destroy != NULL) {
        loaded->destroy(&loaded->table);
    }
    if (loaded->library != NULL) {
        (void)dlclose(loaded->library);
    }
    *loaded = (ShadowSpillPytorchLoadedBackend){0};
}
