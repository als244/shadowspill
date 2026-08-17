#include <shadowspill/admission_replay.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int causal_successor_replay(void) {
    const ShadowSpillAdmissionReplayOperation operations[] = {
        {0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 96U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE, 0U},
        {1U, 0U, 0U, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT, 0U},
        {2U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 64U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_RESERVE, 0U},
        {3U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED, 0U},
        {4U, 0U, 0U, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT, 0U},
        {5U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_RELEASE, 0U},
    };
    ShadowSpillAdmissionReplayDecision decisions[6] = {{0}};
    ShadowSpillAdmissionReuseDependency dependencies[2] = {{0}};
    ShadowSpillAdmissionReplayResult result = {
        .decisions = decisions,
        .decision_capacity = 6U,
        .dependencies = dependencies,
        .dependency_capacity = 2U,
    };
    const ShadowSpillAdmissionReplayProgram program = {
        .abi_version = SHADOWSPILL_ADMISSION_REPLAY_ABI_VERSION,
        .capacity_bytes = 128U,
        .minimum_alignment = 1U,
        .lease_count = 2U,
        .dependency_count = 1U,
        .operations = operations,
        .operation_count = 6U,
    };
    const ShadowSpillAdmissionReplayStatus status =
        shadowspill_admission_replay_run(&program, &result);
    return status != SHADOWSPILL_ADMISSION_REPLAY_OK ||
        result.status != SHADOWSPILL_ADMISSION_REPLAY_OK ||
        result.decision_count != 6U || result.dependency_result_count != 1U ||
        result.peak_allocated_bytes != 96U ||
        result.peak_reserved_bytes != 96U ||
        result.final_allocated_bytes != 0U ||
        result.final_reserved_bytes != 0U ||
        result.final_largest_free_range_bytes != 128U ||
        decisions[2].predecessor_lease_id != 0U ||
        decisions[2].physical_bytes_delta != 0 ||
        decisions[2].resulting_state !=
            SHADOWSPILL_ADMISSION_REPLAY_LEASE_SUCCESSOR_RESERVED ||
        decisions[3].offset != 0U || decisions[3].charged_bytes != 96U ||
        decisions[3].resulting_state !=
            SHADOWSPILL_ADMISSION_REPLAY_LEASE_IN_USE ||
        dependencies[0].predecessor_lease_id != 0U ||
        dependencies[0].successor_lease_id != 1U ||
        dependencies[0].dependency_id != 0U ||
        dependencies[0].consumer_operation_index != 3U;
}

static int promised_dependency_replay(void) {
    const ShadowSpillAdmissionReplayOperation operations[] = {
        {0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 128U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE, 0U},
        {1U, 0U, 0U, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT, 1U},
        {2U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 32U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_RESERVE, 0U},
        {3U, 0U, 0U, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_PUBLISH_DEPENDENCY, 0U},
        {4U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED, 0U},
        {5U, 0U, 0U, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT, 0U},
        {6U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_RELEASE, 0U},
    };
    ShadowSpillAdmissionReplayDecision decisions[7] = {{0}};
    ShadowSpillAdmissionReuseDependency dependency = {0};
    ShadowSpillAdmissionReplayResult result = {
        .decisions = decisions,
        .decision_capacity = 7U,
        .dependencies = &dependency,
        .dependency_capacity = 1U,
    };
    const ShadowSpillAdmissionReplayProgram program = {
        .abi_version = SHADOWSPILL_ADMISSION_REPLAY_ABI_VERSION,
        .capacity_bytes = 128U,
        .minimum_alignment = 1U,
        .lease_count = 2U,
        .dependency_count = 1U,
        .operations = operations,
        .operation_count = 7U,
    };
    return shadowspill_admission_replay_run(&program, &result) !=
            SHADOWSPILL_ADMISSION_REPLAY_OK ||
        result.dependency_result_count != 1U ||
        dependency.dependency_id != 0U ||
        result.final_allocated_bytes != 0U;
}

static int infeasible_replay_reports_geometry(void) {
    const ShadowSpillAdmissionReplayOperation operations[] = {
        {0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 96U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE, 0U},
        {1U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 64U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE, 0U},
    };
    ShadowSpillAdmissionReplayDecision decisions[2] = {{0}};
    ShadowSpillAdmissionReplayResult result = {
        .decisions = decisions,
        .decision_capacity = 2U,
    };
    const ShadowSpillAdmissionReplayProgram program = {
        .abi_version = SHADOWSPILL_ADMISSION_REPLAY_ABI_VERSION,
        .capacity_bytes = 128U,
        .minimum_alignment = 1U,
        .lease_count = 2U,
        .operations = operations,
        .operation_count = 2U,
    };
    return shadowspill_admission_replay_run(&program, &result) !=
            SHADOWSPILL_ADMISSION_REPLAY_INFEASIBLE ||
        result.error_operation_index != 1U || result.error_lease_id != 1U ||
        result.error_requested_bytes != 64U || result.error_free_bytes != 32U ||
        result.error_largest_free_range_bytes != 32U;
}

static int reusable_workspace_preserves_decisions(void) {
    const ShadowSpillAdmissionReplayOperation operations[] = {
        {0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 96U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE, 0U},
        {1U, 0U, 0U, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT, 1U},
        {2U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 64U, 1U,
         SHADOWSPILL_ADMISSION_REPLAY_RESERVE, 0U},
        {3U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED, 0U},
        {4U, 0U, 0U, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT, 0U},
        {5U, 1U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U, 0U,
         SHADOWSPILL_ADMISSION_REPLAY_RELEASE, 0U},
    };
    const ShadowSpillAdmissionReplayProgram program = {
        .abi_version = SHADOWSPILL_ADMISSION_REPLAY_ABI_VERSION,
        .capacity_bytes = 128U,
        .minimum_alignment = 1U,
        .lease_count = 2U,
        .dependency_count = 1U,
        .operations = operations,
        .operation_count = 6U,
    };
    ShadowSpillAdmissionReplayWorkspace *workspace = NULL;
    if (shadowspill_admission_replay_workspace_create(
            2U, 1U, &workspace
        ) != SHADOWSPILL_ADMISSION_REPLAY_OK) {
        return 1;
    }
    uint64_t expected_digest = 0U;
    int failed = 0;
    for (uint32_t repetition = 0U; repetition < 1000U; ++repetition) {
        ShadowSpillAdmissionReplayDecision decisions[6] = {{0}};
        ShadowSpillAdmissionReuseDependency dependency = {0};
        ShadowSpillAdmissionReplayResult result = {
            .decisions = decisions,
            .decision_capacity = 6U,
            .dependencies = &dependency,
            .dependency_capacity = 1U,
        };
        if (shadowspill_admission_replay_run_reusing(
                &program, &result, workspace
            ) != SHADOWSPILL_ADMISSION_REPLAY_OK ||
            result.final_allocated_bytes != 0U ||
            result.dependency_result_count != 1U ||
            (repetition != 0U && result.decision_digest != expected_digest)) {
            failed = 1;
            break;
        }
        expected_digest = result.decision_digest;
    }
    shadowspill_admission_replay_workspace_destroy(workspace);
    return failed;
}

int main(void) {
    if (shadowspill_admission_replay_abi_version() !=
        SHADOWSPILL_ADMISSION_REPLAY_ABI_VERSION) {
        fprintf(stderr, "AdmissionReplay canary: ABI mismatch\n");
        return 1;
    }
    if (causal_successor_replay() != 0) {
        fprintf(stderr, "AdmissionReplay canary: causal successor failed\n");
        return EXIT_FAILURE;
    }
    if (promised_dependency_replay() != 0) {
        fprintf(stderr, "AdmissionReplay canary: promised dependency failed\n");
        return EXIT_FAILURE;
    }
    if (infeasible_replay_reports_geometry() != 0) {
        fprintf(stderr, "AdmissionReplay canary: infeasible geometry failed\n");
        return EXIT_FAILURE;
    }
    if (reusable_workspace_preserves_decisions() != 0) {
        fprintf(stderr, "AdmissionReplay canary: reusable workspace failed\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
