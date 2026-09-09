#include <stdint.h>
#include <stdlib.h>

#include <shadowspill/planner.h>
#include <shadowspill/pressurefit/pressurefit.h>

int main(void) {
    const uint32_t task_device[] = {0U};
    const uint8_t task_kind[] = {0U};
    const uint32_t task_lane[] = {0U};
    const uint32_t empty_offsets[] = {0U, 0U};

    if (shadowspill_abi_version() != SHADOWSPILL_ABI_VERSION) {
        return EXIT_FAILURE;
    }

    const uint64_t problem_alias_size[] = {32U};
    const ShadowSpillSimulationDevice problem_device = {
        .capacity_bytes = 64U,
        .fetch_bandwidth_bytes_per_second = 1U,
        .evict_bandwidth_bytes_per_second = 1U,
    };
    const uint32_t problem_alias_device[] = {0U};
    const uint8_t problem_retain_spill[] = {1U};
    const uint32_t problem_priority[] = {0U};
    const uint64_t problem_alias_version[] = {0U};
    const uint64_t problem_task_runtime[] = {10U};
    const uint64_t problem_task_workspace[] = {0U};
    const uint32_t problem_input_offsets[] = {0U, 1U};
    const uint32_t problem_input_aliases[] = {0U};
    const uint32_t problem_initial_aliases[] = {0U};
    const uint8_t problem_initial_locations[] = {
        SHADOWSPILL_MEMORY_DEVICE,
    };
    const ShadowSpillSimulationProgram problem_simulation = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .device_count = 1U,
        .alias_count = 1U,
        .task_count = 1U,
        .initial_count = 1U,
        .input_count = 1U,
        .spill_capacity_bytes = 64U,
        .devices = &problem_device,
        .alias_device = problem_alias_device,
        .alias_size_bytes = problem_alias_size,
        .alias_initial_version = problem_alias_version,
        .alias_retain_spill_copy = problem_retain_spill,
        .task_device = task_device,
        .task_resource_kind = task_kind,
        .task_resource_lane = task_lane,
        .task_runtime_ns = problem_task_runtime,
        .task_workspace_bytes = problem_task_workspace,
        .dependency_offsets = empty_offsets,
        .input_offsets = problem_input_offsets,
        .input_aliases = problem_input_aliases,
        .output_offsets = empty_offsets,
        .mutation_offsets = empty_offsets,
        .initial_aliases = problem_initial_aliases,
        .initial_locations = problem_initial_locations,
    };
    const char *problem_alias_names[] = {"alias"};
    const char *problem_task_names[] = {"task"};
    const uint8_t problem_strategies[] = {SHADOWSPILL_PRESSUREFIT_RESIDENCY_TIGHT_STALL};
    const uint8_t problem_rules[] = {SHADOWSPILL_PRESSUREFIT_FETCH_LATEST_SAFE};
    /* Plain emission only. All three axes are lists, and the candidate count
     * is their product, so leaving this one empty asks for no candidates. */
    const uint8_t problem_modes[] = {0U};
    const ShadowSpillPressureFitOptions problem_options = {
        .coalescing_modes = problem_modes,
        .coalescing_mode_count = 1U,
        .residency_strategies = problem_strategies,
        .residency_strategy_count = 1U,
        .fetch_rules = problem_rules,
        .fetch_rule_count = 1U,
        .max_repair_attempts = 1U,
    };
    const ShadowSpillIndexedProblem program_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .context = {
            .simulation = &problem_simulation,
            .alias_json_names = problem_alias_names,
            .task_json_names = problem_task_names,
        },
        .device_priority = problem_priority,
    };
    ShadowSpillPressureFitPreflightResult preflight = {0};
    if (shadowspill_pressurefit_preflight(
            &program_problem,
            &preflight
        ) != SHADOWSPILL_STATUS_OK ||
        preflight.status != SHADOWSPILL_STATUS_OK ||
        preflight.failure_kind != SHADOWSPILL_PRESSUREFIT_PREFLIGHT_NONE) {
        return EXIT_FAILURE;
    }
    const uint64_t excessive_workspace[] = {65U};
    ShadowSpillSimulationProgram oversized_simulation = problem_simulation;
    oversized_simulation.task_workspace_bytes = excessive_workspace;
    ShadowSpillIndexedProblem oversized_problem = program_problem;
    oversized_problem.context.simulation = &oversized_simulation;
    if (shadowspill_pressurefit_preflight(
            &oversized_problem,
            &preflight
        ) != SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE ||
        preflight.status != SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE ||
        preflight.failure_kind != SHADOWSPILL_PRESSUREFIT_PREFLIGHT_WORKSPACE_CAPACITY ||
        preflight.error_boundary != 0 || preflight.required_bytes != 65U ||
        preflight.capacity_bytes != 64U) {
        return EXIT_FAILURE;
    }
    ShadowSpillPressureFitResult program_problem_result = {0};
    if (shadowspill_pressurefit_search(
            &program_problem,
            1U,
            &problem_options,
            &program_problem_result
        ) != SHADOWSPILL_STATUS_OK ||
        program_problem_result.candidate_count != 1U ||
        program_problem_result.selected_candidate_index != 0U ||
        program_problem_result.work.simulation_calls != 1U ||
        program_problem_result.work.schedule_emissions != 1U ||
        program_problem_result.candidates[0].work.simulation_calls != 1U ||
        program_problem_result.candidates[0].work.schedule_emissions != 1U ||
        program_problem_result.selected_schedule.action_count != 1U ||
        program_problem_result.selected_schedule.action_kinds[0] !=
            SHADOWSPILL_MEMORY_RELEASE) {
        shadowspill_pressurefit_result_destroy(&program_problem_result);
        return EXIT_FAILURE;
    }
    shadowspill_pressurefit_result_destroy(&program_problem_result);
    return EXIT_SUCCESS;
}
