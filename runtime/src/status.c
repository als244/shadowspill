#include "internal.h"

uint32_t shadowspill_runtime_abi_version(void) {
    return SHADOWSPILL_RUNTIME_ABI_VERSION;
}

const char *shadowspill_runtime_status_string(ShadowSpillRuntimeStatus status) {
    switch (status) {
        case SHADOWSPILL_RUNTIME_OK:
            return "ok";
        case SHADOWSPILL_RUNTIME_INVALID_ARGUMENT:
            return "invalid runtime argument";
        case SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE:
            return "runtime allocation failure";
        case SHADOWSPILL_RUNTIME_OUT_OF_MEMORY:
            return "device slab out of memory";
        case SHADOWSPILL_RUNTIME_NO_PROGRESS:
            return "allocation cannot make progress";
        case SHADOWSPILL_RUNTIME_INVALID_STATE:
            return "invalid runtime state";
        case SHADOWSPILL_RUNTIME_PLAN_VIOLATION:
            return "execution plan violation";
        case SHADOWSPILL_RUNTIME_BACKEND_FAILURE:
            return "backend failure";
        case SHADOWSPILL_RUNTIME_WORKER_FAILURE:
            return "worker thread failure";
        case SHADOWSPILL_RUNTIME_CLOSED:
            return "runtime is closed";
    }
    return "unknown runtime status";
}
