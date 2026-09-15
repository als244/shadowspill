/* The worker pool, and the (problem, candidate) tasks it hands out. */
#include "internal.h"

static uint32_t problem_of_task(
    const ProgramSearch *search, uint32_t task, uint32_t *candidate
) {
    for (uint32_t index = 0U; index < search->problem_count; ++index) {
        const SearchedProblem *problem = &search->problems[index];
        if (task < problem->first_task + problem->candidate_count) {
            *candidate = task - problem->first_task;
            return index;
        }
    }
    *candidate = 0U;
    return SHADOWSPILL_PLANNER_NO_INDEX;
}

/* Give this worker a workspace sized for `index`, reusing the one it has when
 * it is already for that problem. */
static int worker_workspace_for(SearchWorker *worker, uint32_t index) {
    if (worker->workspace_problem == index) {
        return 0;
    }
    if (worker->workspace_problem != SHADOWSPILL_PLANNER_NO_INDEX) {
        shadowspill_candidate_candidate_workspace_destroy(&worker->workspace);
        worker->workspace_problem = SHADOWSPILL_PLANNER_NO_INDEX;
    }
    if (shadowspill_candidate_candidate_workspace_create(
            worker->search->problems[index].problem, &worker->workspace
        ) != 0) {
        return -1;
    }
    worker->workspace_problem = index;
    return 0;
}

/* The plan to beat, as the winner slot names it: not a candidate, and
 * never displaced by one that merely ties it. */

/* Record a plan as this problem's answer when it beats what is held. Ties
 * fall to the earlier candidate, so the answer does not depend on which
 * worker finished first; the plan to beat came before every candidate. */
static int offer_problem_winner(
    SearchedProblem *problem,
    uint32_t candidate,
    uint64_t makespan_ns,
    const ShadowSpillScheduleStorage *schedule
) {
    while (atomic_flag_test_and_set_explicit(&problem->guard, memory_order_acquire)) {
        /* Held only for a schedule copy, and taken only by a worker that has
         * already beaten the record, so spinning beats descheduling. */
        shadowspill_thread_yield();
    }
    int failed = 0;
    if (problem->selected_candidate == SHADOWSPILL_PLANNER_NO_INDEX ||
        makespan_ns < problem->selected_makespan_ns ||
        (makespan_ns == problem->selected_makespan_ns &&
         problem->selected_candidate != INCUMBENT_CANDIDATE &&
         candidate < problem->selected_candidate)) {
        failed = shadowspill_schedule_storage_copy(
            &problem->selected, schedule
        ) != 0;
        if (!failed) {
            problem->selected_candidate = candidate;
            problem->selected_makespan_ns = makespan_ns;
        }
    }
    atomic_flag_clear_explicit(&problem->guard, memory_order_release);
    return failed ? -1 : 0;
}

/* Stamp what a task cost and when it ended. Every exit that answered goes
 * through here, so a candidate that stopped early still lands on the
 * timeline and still reports the work it did before it stopped. An exit
 * that failed internally does not: it ends the whole search, and there is
 * no candidate left to describe. */
static void finish_task(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    const ProgramSearch *search,
    CandidateWorkspace *workspace,
    ShadowSpillPressureFitWorkDiagnostics before,
    uint64_t started
) {
    diagnostic->work = shadowspill_candidate_work_delta(shadowspill_candidate_workspace_work(workspace), before);
    shadowspill_candidate_section_close_total(&diagnostic->work.sections, started);
    /* Both ends are stamped here, at the end, because evaluating a candidate
     * initializes the diagnostic again and would erase a start written
     * before it. */
    diagnostic->started_ns = started - search->origin_ns;
    diagnostic->finished_ns = shadowspill_monotonic_ns() - search->origin_ns;
}

/* Evaluate one candidate. Returns -1 only for an internal failure; a
 * candidate that simply has no answer returns 0. */
static int worker_evaluate_task(SearchWorker *worker, uint32_t task) {
    ProgramSearch *search = worker->search;
    uint32_t candidate = 0U;
    const uint32_t index = problem_of_task(search, task, &candidate);
    if (index == SHADOWSPILL_PLANNER_NO_INDEX) {
        return -1;
    }
    if (worker_workspace_for(worker, index) != 0) {
        return -1;
    }
    SearchedProblem *problem = &search->problems[index];
    CandidateWorkspace *workspace = &worker->workspace;
    const ShadowSpillPressureFitOptions *options = search->options;

    const uint32_t modes = options->coalescing_mode_count;
    const uint32_t rules = options->fetch_rule_count;
    const uint8_t mode = options->coalescing_modes[candidate % modes];
    const uint8_t rule = options->fetch_rules[(candidate / modes) % rules];
    const uint8_t strategy =
        options->residency_strategies[candidate / (modes * rules)];

    ShadowSpillPressureFitCandidateDiagnostic *diagnostic =
        &problem->result->candidates[candidate];
    shadowspill_candidate_initialize_diagnostic(diagnostic, strategy, rule, mode);

    /* Everything from here on is this candidate's, the base reduction
     * included: it is work a worker does for this task and nobody else's. */
    const ShadowSpillPressureFitWorkDiagnostics before = shadowspill_candidate_workspace_work(workspace);
    const uint64_t started = shadowspill_monotonic_ns();

    /* This strategy's base residency, from this worker's own memo. */
    const uint64_t pressure_cells =
        (uint64_t)problem->problem->residency->device_count *
        problem->problem->residency->boundary_count;
    memset(
        workspace->extra_pressure,
        0,
        (size_t)pressure_cells * sizeof(*workspace->extra_pressure)
    );
    ShadowSpillPressureFitResidencyOptions reduce_options;
    shadowspill_candidate_residency_options(workspace, strategy, &reduce_options);
    ShadowSpillPressureFitResidencyResult base;
    const Section reduce = shadowspill_candidate_section_open(&workspace->sections.reduce_ns);
    const ShadowSpillStatus base_status = shadowspill_candidate_reduce_residency(
        problem->problem,
        workspace,
        &reduce_options,
        strategy,
        workspace->base_resident,
        workspace->base_breaks,
        &base
    );
    shadowspill_candidate_section_close(reduce);
    workspace->base_residency = workspace->current_residency;
    if (base_status == SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE) {
        shadowspill_candidate_copy_analytic_error(diagnostic, &base);
        finish_task(diagnostic, search, workspace, before, started);
        return 0;
    }
    if (base_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }

    const int valid = shadowspill_candidate_evaluate_candidate(
        problem->problem,
        &problem->facts,
        options,
        workspace,
        strategy,
        rule,
        mode,
        diagnostic
    );
    finish_task(diagnostic, search, workspace, before, started);
    if (valid < 0) {
        return -1;
    }
    if (valid > 0) {
        return offer_problem_winner(
            problem, candidate, diagnostic->makespan_ns, &workspace->schedule
        );
    }
    return 0;
}

/*
 * Measure the plan to beat at this capacity, before any candidate runs.
 *
 * It is a plan for this resolved program that already runs elsewhere -- at
 * a smaller capacity, say -- and it is measured exactly as a candidate's
 * plan is: simulated, admitted, and placed against the pool. A plan that
 * places becomes the problem's answer until a candidate does strictly
 * better, and is offered to the shared record so that, in the default
 * mode, every candidate measures against it from the start. A plan that
 * does not place here is reported and not used; nothing about the search
 * changes for it. Runs on the calling thread, so its workspace is worker
 * zero's, which the first task resizes as it would anyway.
 */
int shadowspill_candidate_evaluate_incumbent(SearchWorker *worker, uint32_t index) {
    ProgramSearch *search = worker->search;
    SearchedProblem *problem = &search->problems[index];
    ShadowSpillPressureFitResult *result = problem->result;
    const ShadowSpillIndexedSchedule *incumbent = problem->problem->incumbent;
    if (incumbent == NULL) {
        return 0;
    }
    result->incumbent_given = 1U;
    result->incumbent_status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_INTERNAL_ERROR;
    if (worker_workspace_for(worker, index) != 0) {
        return -1;
    }
    CandidateWorkspace *workspace = &worker->workspace;
    if (shadowspill_schedule_storage_assign(&workspace->schedule, incumbent) != 0) {
        return -1;
    }
    ShadowSpillSimulationResult simulation;
    ShadowSpillStatus admission_status = SHADOWSPILL_STATUS_OK;
    ShadowSpillAdmissionReplayResult replay = {0};
    if (shadowspill_candidate_simulate_schedule(
            problem->problem,
            &workspace->schedule.value,
            &workspace->simulation,
            &workspace->admission,
            &workspace->first_violation,
            &simulation,
            &admission_status,
            &replay
        ) != 0) {
        return -1;
    }
    if (admission_status == SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
        result->incumbent_status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_ADMISSION_INFEASIBLE;
        return 0;
    }
    if (simulation.status != SHADOWSPILL_STATUS_OK || simulation.makespan_ns == 0U) {
        result->incumbent_status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_SIMULATION_INFEASIBLE;
        return 0;
    }
    result->incumbent_makespan_ns = simulation.makespan_ns;
    const ShadowSpillAdmissionFacts *placement = problem->problem->context.placement;
    if (placement != NULL) {
        uint64_t required_bytes = 0U;
        if (shadowspill_candidate_place_plan(problem->problem, workspace, &simulation, &required_bytes) != 0) {
            result->incumbent_status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_UNPLACEABLE;
            return 0;
        }
        result->incumbent_required_bytes = required_bytes;
        if (required_bytes > placement->pool_capacity_bytes) {
            result->incumbent_status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_UNPLACEABLE;
            return 0;
        }
        ShadowSpillPressureFitBestPlacedRecord record = {
            .makespan_ns = simulation.makespan_ns,
            .object_capacity_bytes = placement->object_capacity_bytes,
        };
        shadowspill_schedule_digest(
            &problem->problem->context,
            &workspace->schedule.value,
            record.schedule_digest
        );
        (void)shadowspill_best_placed_offer(
            search->options->best_placed, &record, &workspace->schedule
        );
    }
    result->incumbent_status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_VALID;
    return offer_problem_winner(
        problem, INCUMBENT_CANDIDATE, simulation.makespan_ns, &workspace->schedule
    );
}

void *shadowspill_candidate_worker_main(void *argument) {
    SearchWorker *worker = argument;
    shadowspill_name_current_thread("shadowspill.pln");
    while (worker->failed == 0) {
        const uint32_t task = atomic_fetch_add_explicit(
            &worker->search->next_task, 1U, memory_order_relaxed
        );
        if (task >= worker->search->total_tasks) {
            break;
        }
        if (worker_evaluate_task(worker, task) != 0) {
            worker->failed = 1;
        }
    }
    return NULL;
}

/* How many threads to evaluate with. Scheduling rather than search, so this
 * is free to consider the machine: it changes neither which plans are legal
 * nor how they simulate. It does change how many candidates the shared
 * record lets a search skip, which is why per-candidate counters move with
 * it. Never more threads than there is work to give them. */
uint32_t shadowspill_candidate_worker_count_for(
    const ShadowSpillPressureFitOptions *options, uint32_t tasks
) {
    if (tasks <= 1U || options->workers == 1U) {
        return 1U;
    }
    const uint32_t wanted = options->workers != 0U
        ? options->workers
        : shadowspill_logical_cpu_count();
    return wanted < tasks ? wanted : tasks;
}

/* Release everything the search allocated, whatever it managed to finish. */
void shadowspill_candidate_program_search_destroy(
    ProgramSearch *search, SearchWorker *workers, uint32_t worker_count
) {
    for (uint32_t index = 0U; index < worker_count; ++index) {
        if (workers[index].workspace_problem != SHADOWSPILL_PLANNER_NO_INDEX) {
            shadowspill_candidate_candidate_workspace_destroy(&workers[index].workspace);
        }
    }
    free(workers);
    for (uint32_t index = 0U; index < search->problem_count; ++index) {
        SearchedProblem *problem = &search->problems[index];
        if (problem->ready) {
            shadowspill_schedule_facts_destroy(&problem->facts);
            shadowspill_schedule_storage_destroy(&problem->selected);
        }
        if (problem->derived) {
            shadowspill_residency_sparse_lists_destroy(&problem->lists);
        }
    }
    free(search->problems);
}

