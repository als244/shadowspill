/* The entry: what a resolved problem must look like, and the search
 * over several of them. */
#include "internal.h"

static int multiply_u32(uint32_t left, uint32_t right, uint32_t *result) {
    uint64_t value = (uint64_t)left * right;
    if (value > UINT32_MAX) {
        return -1;
    }
    *result = (uint32_t)value;
    return 0;
}

static int strategy_valid(uint8_t strategy) {
    return strategy <= SHADOWSPILL_PRESSUREFIT_RESIDENCY_RELAXED_STALL;
}

static int rule_valid(uint8_t rule) {
    return rule <= SHADOWSPILL_PRESSUREFIT_FETCH_DEMAND;
}

/* The plan to beat indexes this problem's aliases and tasks, and names
 * only kinds and locations the schedule vocabulary has. */
static int incumbent_valid(const ShadowSpillPressureFitProblem *problem) {
    const ShadowSpillIndexedSchedule *plan = problem->incumbent;
    const uint32_t aliases = problem->residency->alias_count;
    const uint32_t tasks = problem->context.simulation->task_count;
    if ((plan->action_count != 0U &&
         (plan->action_trigger_tasks == NULL || plan->action_aliases == NULL ||
          plan->action_kinds == NULL)) ||
        (plan->initial_count != 0U &&
         (plan->initial_aliases == NULL || plan->initial_locations == NULL)) ||
        (plan->final_count != 0U &&
         (plan->final_aliases == NULL || plan->final_locations == NULL))) {
        return 0;
    }
    for (uint32_t index = 0U; index < plan->action_count; ++index) {
        if (plan->action_trigger_tasks[index] >= tasks ||
            plan->action_aliases[index] >= aliases ||
            plan->action_kinds[index] > SHADOWSPILL_MEMORY_WRITE_BACK) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < plan->initial_count; ++index) {
        if (plan->initial_aliases[index] >= aliases ||
            plan->initial_locations[index] > 1U) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < plan->final_count; ++index) {
        if (plan->final_aliases[index] >= aliases ||
            plan->final_locations[index] > 1U) {
            return 0;
        }
    }
    return 1;
}

static int problem_valid(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillPressureFitOptions *options
) {
    if (problem == NULL || options == NULL || problem->residency == NULL ||
        problem->context.simulation == NULL || problem->seed_resident == NULL ||
        problem->seed_breaks == NULL || problem->context.alias_json_names == NULL ||
        problem->context.task_json_names == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->residency->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->context.simulation->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->context.simulation->task_count == 0U ||
        options->residency_strategies == NULL ||
        options->residency_strategy_count == 0U ||
        options->fetch_rules == NULL || options->fetch_rule_count == 0U ||
        options->coalescing_modes == NULL ||
        options->coalescing_mode_count == 0U) {
        return 0;
    }
    for (uint32_t index = 0U; index < options->coalescing_mode_count; ++index) {
        if (options->coalescing_modes[index] > 1U) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < options->residency_strategy_count;
         ++index) {
        if (!strategy_valid(options->residency_strategies[index])) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < options->fetch_rule_count; ++index) {
        if (!rule_valid(options->fetch_rules[index])) {
            return 0;
        }
    }
    for (uint32_t alias = 0U; alias < problem->residency->alias_count; ++alias) {
        if (problem->context.alias_json_names[alias] == NULL) {
            return 0;
        }
    }
    for (uint32_t task = 0U; task < problem->context.simulation->task_count; ++task) {
        if (problem->context.task_json_names[task] == NULL) {
            return 0;
        }
    }
    return problem->incumbent == NULL || incumbent_valid(problem);
}

void shadowspill_pressurefit_result_destroy(
    ShadowSpillPressureFitResult *result
) {
    if (result == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < result->candidate_count; ++index) {
        free(result->candidates[index].steps);
        free(result->candidates[index].cut_aliases);
    }
    free(result->candidates);
    free(result->selected_schedule.action_trigger_tasks);
    free(result->selected_schedule.action_aliases);
    free(result->selected_schedule.action_kinds);
    free(result->selected_schedule.initial_aliases);
    free(result->selected_schedule.initial_locations);
    free(result->selected_schedule.final_aliases);
    free(result->selected_schedule.final_locations);
    free(result->resident_slice_bytes);
    free(result->alias_evict_eligible);
    memset(result, 0, sizeof(*result));
}

/*
 * Evaluating several resolved programs on one set of worker threads.
 *
 * The unit of work is a candidate of a problem -- one residency strategy, one
 * fetch rule, one coalescing mode -- and every such unit of every problem
 * competes for the same workers. Worker count and problem count are
 * independent: asking for eight workers gets eight threads whether there is
 * one problem or five, which is the whole reason this takes a list. ("Pool"
 * is deliberately not used here -- in this codebase a pool is memory, the
 * execution pool or the spill pool, and these are threads.)
 *
 * Three things are shared, and nothing else is:
 *
 * - **The task counter**, an atomic index. A worker takes the next one and
 *   owns it outright; the diagnostic it writes is that task's own slot, so
 *   diagnostics need no lock.
 * - **Each problem's winner**, behind a small spin lock. A worker takes it
 *   only when it has beaten what is recorded, which is rare, and holds it
 *   only for a schedule copy.
 * - **The placement record**, which does its own locking and is the one
 *   thing deliberately shared across problems: a plan placed under any of
 *   them bounds the search under all of them.
 *
 * Everything else a worker touches is its own workspace, including the three
 * memo tables. They are scratch for the search that worker is doing, so they
 * need no synchronisation and cannot race. A workspace is sized for one
 * problem, so a worker that moves to a different problem rebuilds it; tasks
 * are handed out problem by problem, so that is rare.
 */
/*
 * A search has two drivers and one thing they share.
 *
 *   ProgramSearch     drives the whole Program: every resolved problem, the
 *                     task counter workers pull from, and the options
 *   CandidateSearch   drives one candidate: emit -> simulate -> repair
 *   SearchedProblem   drives nothing. It is one resolved problem while it is
 *                     being searched: the facts its candidates share and the
 *                     problem itself (read-only for the whole search), its
 *                     slice of the task range, which is how one flat counter
 *                     serves several problems, and the best plan any worker
 *                     has placed for it, which is the only part that changes
 *                     and the only part that needs a lock.
 */
/*
 * One SearchedProblem per resolved program: the sparse residency lists it
 * needs, the schedule facts it plans against, and the slot its candidates
 * write into. The caller owns the search, so a failure here leaves what this
 * reached for the caller to destroy.
 */
static ShadowSpillStatus prepare_problems(
    ProgramSearch *search,
    const ShadowSpillPressureFitProblem *problems,
    uint32_t problem_count,
    uint32_t per_problem,
    ShadowSpillPressureFitResult *results
) {
    for (uint32_t index = 0U; index < problem_count; ++index) {
        SearchedProblem *problem = &search->problems[index];
        problem->problem = &problems[index];
        if (problems[index].residency->anchor_offsets == NULL) {
            const ShadowSpillPressureFitResidencyProblem *residency = problems[index].residency;
            if (shadowspill_residency_sparse_lists_build(
                    residency->anchors,
                    residency->latest_access_task,
                    residency->output_reservations,
                    residency->alias_count,
                    residency->boundary_count,
                    &problem->lists
                ) != 0) {
                return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            }
            problem->derived = 1;
            problem->owned_residency = *residency;
            problem->owned_residency.anchor_offsets = problem->lists.anchor_offsets;
            problem->owned_residency.anchor_positions = problem->lists.anchor_positions;
            problem->owned_residency.anchor_tasks = problem->lists.anchor_tasks;
            problem->owned_residency.reserved_offsets = problem->lists.reserved_offsets;
            problem->owned_residency.reserved_positions = problem->lists.reserved_positions;
            problem->owned_problem = problems[index];
            problem->owned_problem.residency = &problem->owned_residency;
            problem->problem = &problem->owned_problem;
        }
        problem->result = &results[index];
        problem->first_task = search->total_tasks;
        problem->candidate_count = per_problem;
        problem->selected_candidate = SHADOWSPILL_PLANNER_NO_INDEX;
        search->total_tasks += per_problem;
        results[index].candidates =
            calloc(per_problem, sizeof(*results[index].candidates));
        if (results[index].candidates == NULL ||
            shadowspill_schedule_facts_create(problem->problem, &problem->facts) != 0 ||
            shadowspill_schedule_storage_create(
                problem->problem->residency->alias_count, &problem->selected
            ) != 0) {
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        results[index].candidate_count = per_problem;
        problem->ready = 1;
    }

    return SHADOWSPILL_STATUS_OK;
}

/* What each problem's candidates found, and what finding it cost. */
static void adopt_winners(
    ProgramSearch *search,
    uint32_t problem_count,
    ShadowSpillPressureFitResult *results,
    int *failed
) {
    /* Adopt each problem's winner, and sum what its candidates did. */
    for (uint32_t index = 0U; index < problem_count; ++index) {
        SearchedProblem *problem = &search->problems[index];
        ShadowSpillPressureFitResult *result = &results[index];
        for (uint32_t slot = 0U; slot < problem->candidate_count; ++slot) {
            const ShadowSpillPressureFitCandidateDiagnostic *candidate =
                &result->candidates[slot];
            shadowspill_candidate_add_repairs(&result->repairs, &candidate->repairs);
            result->work = shadowspill_candidate_add_work(result->work, candidate->work);
            /* The problem spans its candidates. A candidate no worker
             * reached has both stamps zero and is skipped, so an untouched
             * problem keeps the zero span it started with. */
            if (candidate->finished_ns == 0U) {
                continue;
            }
            if (result->finished_ns == 0U ||
                candidate->started_ns < result->started_ns) {
                result->started_ns = candidate->started_ns;
            }
            if (candidate->finished_ns > result->finished_ns) {
                result->finished_ns = candidate->finished_ns;
            }
        }
        if (*failed) {
            result->status = SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
            continue;
        }
        if (problem->selected_candidate == SHADOWSPILL_PLANNER_NO_INDEX) {
            result->status = SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE;
            continue;
        }
        if (problem->selected_candidate == INCUMBENT_CANDIDATE) {
            result->selected_candidate_index = SHADOWSPILL_PLANNER_NO_INDEX;
            result->incumbent_selected = 1U;
        } else {
            result->selected_candidate_index = problem->selected_candidate;
        }
        result->selected_makespan_ns = problem->selected_makespan_ns;
        if (shadowspill_candidate_adopt_selected_schedule(result, &problem->selected) != 0) {
            result->status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            *failed = 1;
            continue;
        }
        result->status = SHADOWSPILL_STATUS_OK;
    }

}

ShadowSpillStatus shadowspill_pressurefit_evaluate_resolved(
    const ShadowSpillPressureFitProblem *problems,
    uint32_t problem_count,
    const ShadowSpillPressureFitOptions *options,
    ShadowSpillPressureFitResult *results
) {
    if (problems == NULL || results == NULL || problem_count == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0U; index < problem_count; ++index) {
        memset(&results[index], 0, sizeof(results[index]));
        results[index].selected_candidate_index = SHADOWSPILL_PLANNER_NO_INDEX;
        results[index].status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        if (!problem_valid(&problems[index], options)) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
    }
    uint32_t per_problem = 0U;
    if (multiply_u32(
            options->residency_strategy_count,
            options->fetch_rule_count,
            &per_problem
        ) != 0 ||
        multiply_u32(per_problem, options->coalescing_mode_count, &per_problem) != 0) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t started = shadowspill_monotonic_ns();
    ProgramSearch search = {
        .options = options,
        .problem_count = problem_count,
        .origin_ns = started,
    };
    search.problems = calloc(problem_count, sizeof(*search.problems));
    if (search.problems == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    atomic_init(&search.next_task, 0U);

    const ShadowSpillStatus prepared = prepare_problems(
        &search, problems, problem_count, per_problem, results
    );
    if (prepared != SHADOWSPILL_STATUS_OK) {
        shadowspill_candidate_program_search_destroy(&search, NULL, 0U);
        return prepared;
    }
    const uint32_t worker_count = shadowspill_candidate_worker_count_for(options, search.total_tasks);
    SearchWorker *workers = calloc(worker_count, sizeof(*workers));
    if (workers == NULL) {
        shadowspill_candidate_program_search_destroy(&search, NULL, 0U);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    for (uint32_t index = 0U; index < worker_count; ++index) {
        workers[index].search = &search;
        workers[index].workspace_problem = SHADOWSPILL_PLANNER_NO_INDEX;
    }
    /* Every plan to beat is measured before the first candidate starts, so
     * the bound it sets is there for all of them alike. */
    for (uint32_t index = 0U; index < problem_count; ++index) {
        if (shadowspill_candidate_evaluate_incumbent(&workers[0], index) != 0) {
            shadowspill_candidate_program_search_destroy(&search, workers, worker_count);
            return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
        }
    }

    /* The calling thread is one of the workers, so a single-worker run needs
     * no thread at all and the common case starts one fewer. */
    pthread_t *threads = NULL;
    uint32_t started_threads = 0U;
    if (worker_count > 1U) {
        threads = calloc(worker_count - 1U, sizeof(*threads));
        if (threads == NULL) {
            shadowspill_candidate_program_search_destroy(&search, workers, worker_count);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        for (uint32_t index = 1U; index < worker_count; ++index) {
            if (pthread_create(
                    &threads[index - 1U], NULL, shadowspill_candidate_worker_main, &workers[index]
                ) != 0) {
                break;
            }
            ++started_threads;
        }
    }
    shadowspill_candidate_worker_main(&workers[0]);
    for (uint32_t index = 0U; index < started_threads; ++index) {
        (void)pthread_join(threads[index], NULL);
    }
    free(threads);

    int failed = 0;
    for (uint32_t index = 0U; index < worker_count; ++index) {
        failed |= workers[index].failed;
    }

    adopt_winners(&search, problem_count, results, &failed);
    /* A problem's sections are the sum of its candidates', including their
     * totals and residuals, so the identity total == named + residual still
     * holds -- as an accounting identity over work done, which is what it has
     * to be once several workers run at once. Wall time is not that sum and
     * is not reported here: the whole point of the workers is that the call
     * finishes sooner than the work it did. */
    (void)started;
    shadowspill_candidate_program_search_destroy(&search, workers, worker_count);
    if (failed) {
        return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
    }
    for (uint32_t index = 0U; index < problem_count; ++index) {
        if (results[index].status == SHADOWSPILL_STATUS_OK) {
            return SHADOWSPILL_STATUS_OK;
        }
    }
    return SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE;
}
