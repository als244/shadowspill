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
    }
    return "unknown reason";
}
