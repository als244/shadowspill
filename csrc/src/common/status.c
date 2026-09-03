#include <shadowspill/shadowspill.h>

const char *shadowspill_status_string(ShadowSpillStatus status) {
    switch (status) {
        case SHADOWSPILL_STATUS_OK:
            return "ok";
        case SHADOWSPILL_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case SHADOWSPILL_STATUS_INTERNAL_FAILURE:
            return "internal failure";

        case SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE:
            return "no feasible candidate";
        case SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR:
            return "planner internal error";
        case SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE:
            return "analytically infeasible";

        case SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY:
            return "initial residency exceeds device capacity";
        case SHADOWSPILL_STATUS_INITIAL_SPILL_CAPACITY:
            return "initial residency exceeds spill capacity";
        case SHADOWSPILL_STATUS_TASK_INPUT_DEADLOCK:
            return "task input deadlock";
        case SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY:
            return "task exceeds device capacity";
        case SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY:
            return "fetch exceeds device capacity";
        case SHADOWSPILL_STATUS_EVICT_SPILL_CAPACITY:
            return "evict exceeds spill capacity";
        case SHADOWSPILL_STATUS_TRANSFER_DEADLOCK:
            return "transfer deadlock";
        case SHADOWSPILL_STATUS_INVALID_RELEASE:
            return "invalid release";
        case SHADOWSPILL_STATUS_RELEASE_TRANSFER_CONFLICT:
            return "release conflicts with a transfer";
        case SHADOWSPILL_STATUS_INVALID_EVICT:
            return "invalid evict";
        case SHADOWSPILL_STATUS_INVALID_FETCH:
            return "invalid fetch";
        case SHADOWSPILL_STATUS_FINAL_RESIDENCY:
            return "final residency unsatisfied";
        case SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR:
            return "simulator internal error";

        case SHADOWSPILL_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case SHADOWSPILL_STATUS_NO_PROGRESS:
            return "out of memory with nothing left to release";
        case SHADOWSPILL_STATUS_INVALID_STATE:
            return "invalid state";
        case SHADOWSPILL_STATUS_PLAN_VIOLATION:
            return "execution plan violation";
        case SHADOWSPILL_STATUS_BACKEND_FAILURE:
            return "backend failure";
        case SHADOWSPILL_STATUS_WORKER_FAILURE:
            return "worker thread failure";
        case SHADOWSPILL_STATUS_CLOSED:
            return "runtime is closed";
        case SHADOWSPILL_STATUS_TASK_ALLOCATION_ENVELOPE_EXCEEDED:
            return "task allocation envelope exceeded";
        case SHADOWSPILL_STATUS_TASK_ALLOCATION_CONTRACT_MISMATCH:
            return "task allocation contract mismatch";

        case SHADOWSPILL_STATUS_REPLAY_INFEASIBLE:
            return "replay is infeasible";
        case SHADOWSPILL_STATUS_INVALID_OPERATIONS:
            return "invalid operation sequence";
    }
    return "unknown status";
}

uint32_t shadowspill_abi_version(void) {
    return SHADOWSPILL_ABI_VERSION;
}
