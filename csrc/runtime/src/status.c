#include "internal.h"

uint32_t shadowspill_runtime_abi_version(void) {
    return SHADOWSPILL_RUNTIME_ABI_VERSION;
}

const char *shadowspill_failure_reason_string(
    ShadowSpillFailureReason reason
) {
    switch (reason) {
        case SHADOWSPILL_FAILURE_REASON_UNSPECIFIED:
            return "unspecified";
        case SHADOWSPILL_FAILURE_REASON_HOST_ALLOCATION_REFUSED:
            return "the process allocator refused memory for an internal "
                   "record (anonymous memory; neither pool)";
        case SHADOWSPILL_FAILURE_REASON_RECORD_CAPACITY_EXHAUSTED:
            return "a sealed bookkeeping table had no free record; the "
                   "reserve is too small for this workload";
        case SHADOWSPILL_FAILURE_REASON_LEASE_RELEASE_REJECTED:
            return "the lease could not be released; it was not linked to "
                   "the pool it names, was already free, or was mid-handoff";
        case SHADOWSPILL_FAILURE_REASON_RESERVATION_CANCEL_REJECTED:
            return "a successor's claim on a predecessor's range could not "
                   "be cancelled";
        case SHADOWSPILL_FAILURE_REASON_RANGE_RETURN_REJECTED:
            return "freed bytes could not be returned to the range allocator";
        case SHADOWSPILL_FAILURE_REASON_POOL_EXHAUSTED:
            return "no range is large enough and nothing remains to release "
                   "for one";
    }
    return "unknown reason";
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
        case SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ENVELOPE_EXCEEDED:
            return "task allocation envelope exceeded";
        case SHADOWSPILL_RUNTIME_TASK_ALLOCATION_CONTRACT_MISMATCH:
            return "task allocation contract mismatch";
    }
    return "unknown runtime status";
}
