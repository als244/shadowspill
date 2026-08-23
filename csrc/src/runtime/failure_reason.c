#include "internal.h"

const char *shadowspill_failure_reason_string(
    ShadowSpillFailureReason reason
) {
    switch (reason) {
        case SHADOWSPILL_FAILURE_REASON_UNSPECIFIED:
            return "unspecified";
        case SHADOWSPILL_FAILURE_REASON_PROCESS_ALLOCATION_REFUSED:
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
        case SHADOWSPILL_FAILURE_REASON_EVENT_RELEASE_REJECTED:
            return "a backend event could not be released back to its pool";
        case SHADOWSPILL_FAILURE_REASON_USE_RECORD_RETURN_REJECTED:
            return "stream-use records could not be returned to the pool "
                   "that lent them";
        case SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED:
            return "the backend refused a stream or event operation";
        case SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED:
            return "an object was not in the residency, version or lease "
                   "state the plan requires here";
        case SHADOWSPILL_FAILURE_REASON_RETIREMENT_PUBLICATION_REJECTED:
            return "a task's retirement event could not be published, so its "
                   "allocations have no completion to retire against";
        case SHADOWSPILL_FAILURE_REASON_RETIREMENT_ENQUEUE_REJECTED:
            return "a retirement could not be queued for the worker";
        case SHADOWSPILL_FAILURE_REASON_TASK_BOUNDARY_REJECTED:
            return "the task boundary did not complete; the status names the "
                   "step that failed";
        case SHADOWSPILL_FAILURE_REASON_TASK_ALLOCATION_REJECTED:
            return "a planned task allocation could not be placed at its "
                   "fixed offset";
    }
    return "unknown reason";
}
