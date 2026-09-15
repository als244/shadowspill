/* The stages one candidate walks, and the loop that walks them. */
#include "internal.h"

static void search_begin(
    CandidateSearch *search,
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillPressureFitOptions *options,
    CandidateWorkspace *workspace,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    memset(search, 0, sizeof(*search));
    search->problem = problem;
    /* The emitter measures against the capacity the plan being built kept,
     * which is the same array the reducer adds to its occupancy. Without
     * this the emitter packs against a capacity the plan does not have, and
     * refining capacity never shrinks the layout it produces. */
    search->facts = *facts;
    search->facts.extra_pressure = workspace->extra_pressure;
    search->options = options;
    search->workspace = workspace;
    search->diagnostic = diagnostic;
    search->strategy = strategy;
    search->rule = rule;
    search->coalesced = coalesced;
    search->placing = problem->context.placement != NULL;
    search->cells = (uint64_t)problem->residency->alias_count *
        problem->residency->boundary_count;
    search->pressure_cells = (uint64_t)problem->residency->device_count *
        problem->residency->boundary_count;
    search->need_emit = 1;
    search->last_error_task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    search->last_error_time_ns = 0U;
    search->failure_repeats = 0U;
    search->plan_capacity_bytes = problem->context.placement == NULL
        ? 0U
        : problem->context.placement->object_capacity_bytes;
    shadowspill_candidate_residency_options(workspace, strategy, &search->reduce_options);

    shadowspill_candidate_initialize_diagnostic(diagnostic, strategy, rule, coalesced);
    memset(
        workspace->extra_pressure,
        0,
        (size_t)search->pressure_cells * sizeof(*workspace->extra_pressure)
    );
    workspace->plan_capacity_given_back = 0U;
    workspace->cut_scratch_count = 0U;
    memcpy(workspace->resident, workspace->base_resident, workspace->packed_cell_count);
    memcpy(workspace->breaks, workspace->base_breaks, workspace->packed_cell_count);
    workspace->fetch_constraint_count = 0U;
    workspace->current_residency = workspace->base_residency;
}

/* Report a plan as this candidate's answer. */
static void set_answer(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    uint64_t makespan_ns,
    const uint8_t *digest
) {
    diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_VALID;
    diagnostic->makespan_ns = makespan_ns;
    memcpy(diagnostic->schedule_digest, digest, SHADOWSPILL_PLANNER_DIGEST_BYTES);
}

/* Answer with a plan the search set aside. The caller reads the answer from
 * the live schedule, so the kept plan has to move back into it. */
static int answer_with_kept(CandidateSearch *search, uint64_t makespan_ns) {
    set_answer(search->diagnostic, makespan_ns, search->best_digest);
    if (shadowspill_schedule_storage_copy(
            &search->workspace->schedule, &search->workspace->best
        ) != 0) {
        return -1;
    }
    return 1;
}

/* Stop without an answer of this candidate's own -- unless it placed a plan,
 * in which case it has one whatever became of it afterwards. The shared
 * record already names that plan, so this candidate has to report it:
 * otherwise the record holds a makespan whose plan nobody kept, and the
 * search would answer with a plan it cannot produce. */
static int answer_or_stop(CandidateSearch *search) {
    if (search->placing && search->placed_makespan_ns != 0U) {
        return answer_with_kept(search, search->placed_makespan_ns);
    }
    return 0;
}

static StageOutcome search_done(CandidateSearch *search, int answer) {
    search->answer = answer;
    return STAGE_DONE;
}

/* Add a step to the trajectory, when the caller asked for one. */
static int record_search_step(CandidateSearch *search) {
    if (search->options->record_reduction_steps == 0U) {
        return 0;
    }
    const uint32_t cuts_before = search->diagnostic->cut_count;
    if (shadowspill_candidate_drain_cuts(search->diagnostic, search->workspace) != 0) {
        return -1;
    }
    return shadowspill_candidate_record_step(
        search->diagnostic,
        (ShadowSpillPressureFitReductionStep){
            .makespan_ns = search->simulation.makespan_ns,
            .capacity_bytes = search->plan_capacity_bytes,
            .cut_offset = cuts_before,
            .cut_count = search->diagnostic->cut_count - cuts_before,
            .repairs = (uint32_t)shadowspill_candidate_repair_total(&search->diagnostic->repairs),
            .simulation_status = search->simulation.status,
            .capacity_violations = search->simulation.capacity_violation_count,
            .flags = search->simulation.status == SHADOWSPILL_STATUS_OK
                ? SHADOWSPILL_STEP_SIMULATED
                : 0U,
        }
    );
}

static void mark_search_step(
    CandidateSearch *search, uint32_t flags, uint64_t required_bytes
) {
    if (search->options->record_reduction_steps != 0U) {
        shadowspill_candidate_mark_last_step(search->diagnostic, flags, required_bytes);
    }
}

/*
 * Diagnostic-only reduction tracing, enabled by SHADOWSPILL_PRESSUREFIT_REDUCTION_TRACE
 * and never active in normal planning.
 *
 * Emitted after every simulation rather than only after a failing one,
 * because a reduction that succeeds is exactly the interesting case: whether
 * makespan falls monotonically as a candidate reduces, or rises and later
 * recovers. The resolved program is identified by the problem it was compiled
 * from, since a policy alone is shared across every resolved program and
 * grouping by it merges them.
 */
static void trace_reduction(const CandidateSearch *search) {
    static _Thread_local int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("SHADOWSPILL_PRESSUREFIT_REDUCTION_TRACE") != NULL;
    }
    if (!enabled) {
        return;
    }
    fprintf(
        stderr,
        "reduction-trace resolved=%llu strategy=%u rule=%u coalesced=%u "
        "step=%llu status=%d makespan=%llu shortfalls=%u actions=%u\n",
        (unsigned long long)(uintptr_t)search->problem,
        search->strategy,
        search->rule,
        search->coalesced,
        (unsigned long long)shadowspill_candidate_repair_total(&search->diagnostic->repairs),
        (int)search->simulation_status,
        (unsigned long long)search->simulation.makespan_ns,
        search->simulation.capacity_violation_count,
        search->workspace->schedule.value.action_count
    );
}

/*
 * Diagnostic-only repair tracing, enabled by SHADOWSPILL_REPAIR_TRACE and
 * never active in normal planning.
 *
 * `makespan` and the transfer totals are what a dominance bound would be
 * built from, logged per repair so a reader can see whether any of them rises
 * fast enough to cross the plan in hand before the candidate converges
 * anyway. `makespan` is only meaningful when status is OK: the simulator
 * assigns it on the success path alone.
 */
static void trace_repair(const CandidateSearch *search) {
    static _Thread_local int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("SHADOWSPILL_REPAIR_TRACE") != NULL;
    }
    if (!enabled) {
        return;
    }
    const ShadowSpillIndexedSchedule *schedule = &search->workspace->schedule.value;
    uint64_t fetch_bytes = 0U;
    uint64_t evict_bytes = 0U;
    for (uint32_t index = 0U; index < schedule->action_count; ++index) {
        const uint64_t bytes =
            search->problem->residency->alias_size_bytes[schedule->action_aliases[index]];
        if (schedule->action_kinds[index] == SHADOWSPILL_MEMORY_FETCH) {
            fetch_bytes += bytes;
        } else if (schedule->action_kinds[index] == SHADOWSPILL_MEMORY_EVICT) {
            evict_bytes += bytes;
        }
    }
    fprintf(
        stderr,
        "repair-trace strategy=%u rule=%u coalesced=%u attempt=%llu status=%d "
        "makespan=%llu fetch_bytes=%llu evict_bytes=%llu actions=%u time=%llu "
        "used=%llu requested=%llu capacity=%llu\n",
        search->strategy,
        search->rule,
        search->coalesced,
        (unsigned long long)shadowspill_candidate_repair_total(&search->diagnostic->repairs),
        (int)search->simulation_status,
        (unsigned long long)search->simulation.makespan_ns,
        (unsigned long long)fetch_bytes,
        (unsigned long long)evict_bytes,
        schedule->action_count,
        (unsigned long long)search->simulation.error_time_ns,
        (unsigned long long)search->simulation.error_used_bytes,
        (unsigned long long)search->simulation.error_requested_bytes,
        (unsigned long long)search->simulation.error_capacity_bytes
    );
}

/* Turn the current residency into an ordered schedule. */
static StageOutcome search_emit(CandidateSearch *search) {
    CandidateWorkspace *workspace = search->workspace;
    if (search->rule == SHADOWSPILL_PRESSUREFIT_FETCH_INTERVAL_ENTRY &&
        shadowspill_extend_interval_entries(
            &search->facts, workspace->resident, workspace->breaks
        ) != 0) {
        return search_done(search, -1);
    }
    if (shadowspill_candidate_emit_cached(
            &search->facts,
            workspace,
            workspace->resident,
            workspace->breaks,
            search->rule,
            search->coalesced,
            search->reduce_options.fetch_headroom
        ) != 0) {
        return search_done(search, -1);
    }
    const int constrained = shadowspill_apply_fetch_trigger_constraints(
        &search->facts,
        workspace->fetch_constraints,
        workspace->fetch_constraint_count,
        &workspace->schedule
    );
    if (constrained < 0) {
        return search_done(search, -1);
    }
    if (constrained > 0) {
        search->diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_ADMISSION_INFEASIBLE;
        return search_done(search, answer_or_stop(search));
    }
    search->need_emit = 0;
    return STAGE_NEXT;
}

/* Replay the schedule for a makespan, admitting it into the pool on the way. */
static StageOutcome search_simulate(CandidateSearch *search) {
    search->admission_status = SHADOWSPILL_STATUS_OK;
    memset(&search->admission_result, 0, sizeof(search->admission_result));
    memset(&search->admission_annotation, 0, sizeof(search->admission_annotation));
    search->simulation_entry = NULL;
    if (shadowspill_candidate_simulate_cached(
            search->problem,
            search->workspace,
            &search->simulation,
            &search->admission_status,
            &search->admission_result,
            &search->admission_annotation,
            &search->simulation_entry
        ) != 0) {
        return search_done(search, -1);
    }
    search->simulation_status = (ShadowSpillStatus)search->simulation.status;
    if (record_search_step(search) != 0) {
        return search_done(search, -1);
    }
    trace_reduction(search);
    return STAGE_NEXT;
}

/* Record a trigger move and count it, or stop if it was already tried:
 * repeating a move the schedule already carries would loop. */
static StageOutcome record_admission_move(
    CandidateSearch *search,
    ShadowSpillFetchTriggerConstraint constraint,
    uint64_t *attempts
) {
    const int recorded = shadowspill_candidate_record_fetch_constraint(search->workspace, constraint);
    if (recorded < 0) {
        return search_done(search, -1);
    }
    if (recorded > 0) {
        shadowspill_candidate_copy_admission_error(
            search->problem,
            &search->workspace->schedule.value,
            &search->admission_result,
            search->admission_annotation,
            search->diagnostic
        );
        return search_done(search, answer_or_stop(search));
    }
    ++*attempts;
    return STAGE_REPEAT;
}

/* Admission refused the schedule: move the fetch that overran, or make
 * room for it and reduce again. */
static StageOutcome search_repair_admission(CandidateSearch *search) {
    if (search->admission_status != SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
        return STAGE_NEXT;
    }
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    if (shadowspill_candidate_may_repair_again(search->options, diagnostic)) {
        ShadowSpillFetchTriggerConstraint constraint = {0};
        int moved = shadowspill_candidate_advance_admission_fetch(
            &search->facts,
            search->admission_annotation,
            &search->workspace->schedule,
            &constraint
        );
        if (moved < 0) {
            return search_done(search, -1);
        }
        if (moved > 0) {
            return record_admission_move(
                search,
                constraint,
                &diagnostic->repairs.admission_fetch_advance_attempts
            );
        }
        moved = shadowspill_candidate_delay_admission_fetch(
            &search->facts,
            &search->admission_result,
            search->admission_annotation,
            &search->workspace->schedule,
            &constraint
        );
        if (moved < 0) {
            return search_done(search, -1);
        }
        if (moved > 0) {
            return record_admission_move(
                search,
                constraint,
                &diagnostic->repairs.admission_fetch_delay_attempts
            );
        }
        const int pressed = shadowspill_candidate_add_admission_repair_pressure(
            search->problem,
            search->workspace,
            &search->reduce_options,
            &search->admission_result,
            search->admission_annotation,
            &search->workspace->schedule.value
        );
        if (pressed < 0) {
            return search_done(search, -1);
        }
        if (pressed > 0) {
            ++diagnostic->repairs.admission_pressure_boundary_attempts;
            const int reduced = shadowspill_candidate_reduce_repaired_candidate(
                search->problem,
                search->workspace,
                &search->reduce_options,
                search->strategy,
                diagnostic
            );
            if (reduced < 0) {
                return search_done(search, -1);
            }
            if (reduced == 0) {
                return search_done(search, answer_or_stop(search));
            }
            search->need_emit = 1;
            return STAGE_REPEAT;
        }
    }
    shadowspill_candidate_copy_admission_error(
        search->problem,
        &search->workspace->schedule.value,
        &search->admission_result,
        search->admission_annotation,
        diagnostic
    );
    if (!shadowspill_candidate_may_repair_again(search->options, diagnostic)) {
        diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_REPAIR_EXHAUSTED;
    }
    return search_done(search, answer_or_stop(search));
}

/* Name the schedule. Two candidates that reduce to the same plan get the same
 * name, which is how the search recognises a plan it has already measured. */
/* The schedule's name, computed the first time a stage needs it: a plan that
 * neither improves on the candidate's best nor reaches placement is never
 * named, and most plans are such. */
static const uint8_t *schedule_name(CandidateSearch *search) {
    if (search->simulation_entry->digest_valid == 0U) {
        shadowspill_schedule_digest(
            &search->problem->context,
            &search->workspace->schedule.value,
            search->simulation_entry->digest
        );
        search->simulation_entry->digest_valid = 1U;
    }
    return search->simulation_entry->digest;
}

static StageOutcome search_digest(CandidateSearch *search) {
    search->improves = search->best_makespan_ns == 0U ||
        search->simulation.makespan_ns < search->best_makespan_ns;
    return STAGE_NEXT;
}

/* Keep a plan whose layout fit, offering it to the shared record. */
static StageOutcome search_keep_placed(CandidateSearch *search) {
    ShadowSpillPressureFitBestPlacedRecord record = {
        .makespan_ns = search->simulation.makespan_ns,
        .object_capacity_bytes = search->plan_capacity_bytes,
        .capacity_given_back_bytes = search->workspace->plan_capacity_given_back,
        .residency_strategy = search->strategy,
        .fetch_rule = search->rule,
        .coalesced = search->coalesced,
    };
    memcpy(
        record.schedule_digest,
        schedule_name(search),
        sizeof(record.schedule_digest)
    );
    /* Re-compared under the lock: another candidate may have placed something
     * better while this was being measured, so being admitted is not being
     * best. */
    (void)shadowspill_best_placed_offer(
        search->options->best_placed, &record, &search->workspace->schedule
    );
    ++search->diagnostic->placements_admitted;
    mark_search_step(search, SHADOWSPILL_STEP_PLACED, 0U);
    if (search->placed_makespan_ns != 0U &&
        search->simulation.makespan_ns >= search->placed_makespan_ns) {
        return STAGE_NEXT;
    }
    mark_search_step(search, SHADOWSPILL_STEP_BEST, 0U);
    search->diagnostic->repairs_at_best =
        (uint32_t)shadowspill_candidate_repair_total(&search->diagnostic->repairs);
    search->placed_makespan_ns = search->simulation.makespan_ns;
    if (shadowspill_schedule_storage_copy(
            &search->workspace->best, &search->workspace->schedule
        ) != 0) {
        return search_done(search, -1);
    }
    memcpy(search->best_digest, schedule_name(search), sizeof(search->best_digest));
    return STAGE_NEXT;
}

/*
 * The pool cannot place this plan, which is a fact about the plan and not
 * about the search: it gives back exactly what it overran and reduces again.
 * Uniform pressure is how a plan expresses a smaller capacity to the reducer.
 *
 * The extent does not fall byte for byte with the capacity -- on one measured
 * point a 1 GiB reduction moved it 2.2 GB -- so handing back the whole
 * overage can overshoot the capacity that would have fit, which a bounded
 * step avoids at the cost of more rounds.
 */
static StageOutcome search_refine_capacity(
    CandidateSearch *search, uint64_t required_bytes, uint64_t pool_bytes
) {
    CandidateWorkspace *workspace = search->workspace;
    const uint64_t shortfall = required_bytes - pool_bytes;
    const uint64_t step = search->options->capacity_refinement_bytes;
    const uint64_t overage = (step == 0U || shortfall < step) ? shortfall : step;
    ++search->diagnostic->capacity_refinements;
    search->plan_capacity_bytes = search->plan_capacity_bytes > overage
        ? search->plan_capacity_bytes - overage
        : 0U;
    for (uint64_t cell = 0U; cell < search->pressure_cells; ++cell) {
        workspace->extra_pressure[cell] += overage;
    }
    workspace->plan_capacity_given_back += overage;
    /* A new capacity round starts its own count of repeated failures. */
    search->last_error_task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    search->last_error_time_ns = 0U;
    search->failure_repeats = 0U;
    mark_search_step(search, SHADOWSPILL_STEP_REFINED, 0U);
    /* Plan again at the smaller capacity rather than pressing further on what
     * this capacity produced. */
    memcpy(workspace->resident, workspace->base_resident, workspace->packed_cell_count);
    memcpy(workspace->breaks, workspace->base_breaks, workspace->packed_cell_count);
    workspace->fetch_constraint_count = 0U;
    if (shadowspill_candidate_reduce_repaired_candidate(
            search->problem,
            workspace,
            &search->reduce_options,
            search->strategy,
            search->diagnostic
        ) > 0) {
        search->need_emit = 1;
        return STAGE_REPEAT;
    }
    /* Nothing left to reduce. The plan in hand is all this round produced, so
     * settle with it rather than starting another round that would rebuild
     * the same thing. */
    return STAGE_NEXT;
}

/* A plan no better than what this candidate already placed cannot become
 * its answer, whatever the mode. */
static int candidate_refuses(const CandidateSearch *search) {
    return search->placed_makespan_ns != 0U &&
        search->simulation.makespan_ns >= search->placed_makespan_ns;
}

/* Whether the shared best-placed record refuses this makespan. Lock-free
 * and possibly stale, which costs at most a measurement that would have
 * been skipped. */
static int shared_refuses(const CandidateSearch *search) {
    const uint64_t bound =
        shadowspill_best_placed_bound(search->options->best_placed);
    return bound != 0U && search->simulation.makespan_ns >= bound;
}

/*
 * Measure whether this plan has a layout that fits.
 *
 * Every new plan that could still win is worth measuring. The gate is what
 * keeps this affordable: a plan no better than one already placed cannot
 * become the answer, so it is never measured. Skipping plans on any other
 * ground -- waiting for a local minimum, say -- can leave a candidate that
 * never placed anything at all, and a candidate with no placed plan has no
 * answer to give.
 *
 * A skipped measurement is not neutral: it also skips the capacity
 * refinement a failed placement would have triggered, so a skip edits the
 * candidate's whole descent. The default consults the shared record, which
 * makes a candidate's outcome depend on what other workers had placed by
 * then -- near-tied points settle on different plans run to run. That
 * variance is an accepted trade for the wall time the gate saves.
 * Deterministic mode declines the trade: its gate is candidate-local, so
 * every outcome is a pure function of its inputs and parallel evaluation
 * reproduces exactly, at the cost of measuring plans the shared bound
 * would have skipped.
 */
static StageOutcome search_place(CandidateSearch *search) {
    const int refused = search->options->deterministic
        ? candidate_refuses(search)
        : shared_refuses(search);
    if (refused ||
        shadowspill_candidate_fingerprint_equal(
            search->placed_identity, search->simulation_entry->identity
        )) {
        return STAGE_NEXT;
    }
    search->placed_identity = search->simulation_entry->identity;
    const uint64_t pool_bytes = search->problem->context.placement == NULL
        ? 0U
        : search->problem->context.placement->pool_capacity_bytes;
    ++search->diagnostic->placements_attempted;
    /*
     * The simulation cache keeps makespans, not timelines: its entries drop
     * the interval arrays, which point into the shared workspace. Placement is
     * derived from those intervals, so a cache hit has to be replayed before
     * it can be measured -- otherwise a plan that simply came from the cache
     * reads as a plan that cannot be placed.
     */
    if (search->simulation.task_intervals == NULL) {
        ShadowSpillStatus replay_status = SHADOWSPILL_STATUS_OK;
        ShadowSpillAdmissionReplayResult replay_result = {0};
        if (shadowspill_candidate_simulate_schedule(
                search->problem,
                &search->workspace->schedule.value,
                &search->workspace->simulation,
                &search->workspace->admission,
                &search->workspace->first_violation,
                &search->simulation,
                &replay_status,
                &replay_result
            ) != 0) {
            return search_done(search, -1);
        }
    }
    uint64_t required_bytes = 0U;
    const int placed = shadowspill_candidate_place_plan(
        search->problem, search->workspace, &search->simulation, &required_bytes
    );
    mark_search_step(
        search,
        placed == 0 ? SHADOWSPILL_STEP_MEASURED : 0U,
        placed == 0 ? required_bytes : 0U
    );
    if (placed != 0) {
        return STAGE_NEXT;
    }
    if (required_bytes <= pool_bytes) {
        return search_keep_placed(search);
    }
    return search_refine_capacity(search, required_bytes, pool_bytes);
}

/* Present the recorded shortfall as a simulation error, so the repair path
 * below reads a success that waited the same way it reads a failure. */
static void present_shortfall_as_error(CandidateSearch *search) {
    const ShadowSpillCapacityViolation *shortfall =
        &search->workspace->first_violation;
    search->simulation.status =
        shortfall->reason == SHADOWSPILL_CAPACITY_TASK_DEVICE
            ? (uint32_t)SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY
            : (uint32_t)SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY;
    search->simulation.error_task = shortfall->task;
    search->simulation.error_alias = shortfall->alias;
    search->simulation.error_device = shortfall->device;
    search->simulation.error_location = shortfall->location;
    search->simulation.error_time_ns = shortfall->time_ns;
    search->simulation.error_capacity_bytes = shortfall->capacity_bytes;
    search->simulation.error_used_bytes = shortfall->used_bytes;
    search->simulation.error_requested_bytes = shortfall->requested_bytes;
}

/* Decide whether the plan in hand is this candidate's answer. */
static StageOutcome search_settle(CandidateSearch *search) {
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    /* A plan that waits for memory is valid but unfinished: the wait is time
     * it pays, and the shortfall that caused it is what repair relieves.
     * Stopping here accepts that cost untouched. */
    const int stalling = search->simulation.capacity_violation_count > 0U;
    if (stalling &&
        shadowspill_candidate_may_repair_again(search->options, diagnostic)) {
        /* Continuing, so this plan has to be kept: repairing past a success
         * can make it worse before it makes it better. When placing, the
         * buffer already holds the best placed plan, which outranks a faster
         * plan that has no layout. */
        if (search->improves && !search->placing) {
            search->best_makespan_ns = search->simulation.makespan_ns;
            if (shadowspill_schedule_storage_copy(
                    &search->workspace->best, &search->workspace->schedule
                ) != 0) {
                return search_done(search, -1);
            }
            memcpy(
                search->best_digest,
                schedule_name(search),
                sizeof(search->best_digest)
            );
        }
        present_shortfall_as_error(search);
        return STAGE_NEXT;
    }
    /*
     * With a pool to place into, the candidate's answer is the best plan whose
     * layout fit -- not the fastest plan it simulated. A plan that cannot be
     * placed cannot run, so offering it as an answer only pushes the rejection
     * to a later layer that has to walk capacity down to escape it.
     */
    if (search->placing) {
        if (search->placed_makespan_ns == 0U) {
            diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_UNPLACEABLE;
            return search_done(search, 0);
        }
        diagnostic->capacity_violation_count =
            search->simulation.capacity_violation_count;
        return search_done(
            search, answer_with_kept(search, search->placed_makespan_ns)
        );
    }
    diagnostic->capacity_violation_count =
        search->simulation.capacity_violation_count;
    if (search->improves) {
        /* The live schedule is already the answer, so nothing needs moving. */
        set_answer(
            diagnostic,
            search->simulation.makespan_ns,
            schedule_name(search)
        );
        return search_done(search, 1);
    }
    /* An earlier repair reached a better plan; the caller reads the winner
     * from the live schedule, so put it back. */
    return search_done(search, answer_with_kept(search, search->best_makespan_ns));
}

/* The plan came up short. Move the fetch that caused it, or make room for
 * it and reduce again; if neither is possible the candidate is finished. */
static StageOutcome search_repair(CandidateSearch *search) {
    trace_repair(search);
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    if (shadowspill_candidate_may_repair_again(search->options, diagnostic)) {
        ShadowSpillFetchTriggerConstraint constraint = {0};
        const int delayed = shadowspill_delay_indexed_fetch(
            &search->facts,
            &search->simulation,
            &search->workspace->schedule,
            &constraint
        );
        if (delayed < 0) {
            return search_done(search, -1);
        }
        if (delayed > 0) {
            const int recorded =
                shadowspill_candidate_record_fetch_constraint(search->workspace, constraint);
            if (recorded < 0) {
                return search_done(search, -1);
            }
            if (recorded > 0) {
                diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_SIMULATION_INFEASIBLE;
                shadowspill_candidate_copy_simulation_error(diagnostic, &search->simulation);
                return search_done(search, answer_or_stop(search));
            }
            ++diagnostic->repairs.simulation_fetch_delay_attempts;
            return STAGE_REPEAT;
        }
        /* The same task, at the same moment, after the last ask was met:
         * the room did not appear where the simulator looks, so ask for more
         * this time. Anywhere else is a new failure and a plain ask. */
        if (search->simulation.error_task == search->last_error_task &&
            search->simulation.error_time_ns == search->last_error_time_ns) {
            ++search->failure_repeats;
        } else {
            search->failure_repeats = 0U;
        }
        search->last_error_task = search->simulation.error_task;
        search->last_error_time_ns = search->simulation.error_time_ns;
        uint64_t asked_beyond_shortfall = 0U;
        if (shadowspill_candidate_add_repair_pressure(
                search->problem,
                search->workspace,
                &search->simulation,
                search->failure_repeats,
                &asked_beyond_shortfall
            ) != 0) {
            ++diagnostic->repairs.simulation_pressure_boundary_attempts;
            if (asked_beyond_shortfall != 0U) {
                ++diagnostic->pressure_escalations;
            }
            int reduced = shadowspill_candidate_reduce_repaired_candidate(
                search->problem,
                search->workspace,
                &search->reduce_options,
                search->strategy,
                diagnostic
            );
            if (reduced == 0 && asked_beyond_shortfall != 0U) {
                /* No cut can meet the larger ask. Take the extra back and ask
                 * for the shortfall alone: the candidate is only ever worse
                 * off for having asked for more, never dead of it. */
                search->workspace->extra_pressure[
                    search->workspace->last_pressure_position
                ] -= asked_beyond_shortfall;
                search->failure_repeats = 0U;
                ++diagnostic->escalations_taken_back;
                reduced = shadowspill_candidate_reduce_repaired_candidate(
                    search->problem,
                    search->workspace,
                    &search->reduce_options,
                    search->strategy,
                    diagnostic
                );
            }
            if (reduced < 0) {
                return search_done(search, -1);
            }
            if (reduced > 0) {
                search->need_emit = 1;
                return STAGE_REPEAT;
            }
            if (search->best_makespan_ns != 0U) {
                return search_done(
                    search, answer_with_kept(search, search->best_makespan_ns)
                );
            }
            return search_done(search, answer_or_stop(search));
        }
    }
    /* A candidate that reached a plan keeps it. Falling through here means the
     * last repair found nothing further to try, which says the search stopped
     * improving -- not that the plan it already has stopped working. */
    if (search->best_makespan_ns != 0U) {
        return search_done(
            search, answer_with_kept(search, search->best_makespan_ns)
        );
    }
    diagnostic->status =
        !shadowspill_candidate_may_repair_again(search->options, diagnostic) &&
            shadowspill_candidate_simulation_failure_may_be_repairable(search->simulation_status)
        ? (uint32_t)SHADOWSPILL_PRESSUREFIT_CANDIDATE_REPAIR_EXHAUSTED
        : (uint32_t)SHADOWSPILL_PRESSUREFIT_CANDIDATE_SIMULATION_INFEASIBLE;
    shadowspill_candidate_copy_simulation_error(diagnostic, &search->simulation);
    return search_done(search, answer_or_stop(search));
}

int shadowspill_candidate_evaluate_candidate(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillPressureFitOptions *candidate_options,
    CandidateWorkspace *workspace,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    CandidateSearch search;
    search_begin(
        &search,
        problem,
        facts,
        candidate_options,
        workspace,
        strategy,
        rule,
        coalesced,
        diagnostic
    );

    while (1) {
        StageOutcome outcome = STAGE_NEXT;
        if (search.need_emit) {
            const Section emit = shadowspill_candidate_section_open(&workspace->sections.emit_ns);
            outcome = search_emit(&search);
            shadowspill_candidate_section_close(emit);
        }
        if (outcome == STAGE_NEXT) {
            const uint64_t admitted_ns = workspace->admission.time_ns;
            const Section simulate = shadowspill_candidate_section_open(&workspace->sections.simulate_ns);
            outcome = search_simulate(&search);
            shadowspill_candidate_section_close(simulate);
            /* Admission runs as part of simulating, so its time is nested
             * inside the section just closed rather than beside it. */
            workspace->sections.admit_ns +=
                workspace->admission.time_ns - admitted_ns;
        }
        if (outcome == STAGE_NEXT) {
            const Section repair = shadowspill_candidate_section_open(&workspace->sections.repair_ns);
            outcome = search_repair_admission(&search);
            shadowspill_candidate_section_close(repair);
        }
        /* A plan that simulated is a plan that could be the answer: name it,
         * measure whether it fits, and decide whether to keep looking. */
        if (outcome == STAGE_NEXT &&
            search.simulation_status == SHADOWSPILL_STATUS_OK) {
            const Section digest = shadowspill_candidate_section_open(&workspace->sections.digest_ns);
            outcome = search_digest(&search);
            shadowspill_candidate_section_close(digest);
            if (outcome == STAGE_NEXT) {
                const Section place = shadowspill_candidate_section_open(&workspace->sections.place_ns);
                outcome = search_place(&search);
                shadowspill_candidate_section_close(place);
            }
            if (outcome == STAGE_NEXT) {
                const Section settle = shadowspill_candidate_section_open(&workspace->sections.select_ns);
                outcome = search_settle(&search);
                shadowspill_candidate_section_close(settle);
            }
        }
        if (outcome == STAGE_NEXT) {
            const Section repair = shadowspill_candidate_section_open(&workspace->sections.repair_ns);
            outcome = search_repair(&search);
            shadowspill_candidate_section_close(repair);
        }
        if (outcome == STAGE_DONE) {
            return search.answer;
        }
    }
}

/* Hand the winner's schedule to the result, which owns it afterwards. */
int shadowspill_candidate_adopt_selected_schedule(
    ShadowSpillPressureFitResult *result,
    ShadowSpillScheduleStorage *selected
) {
    ShadowSpillIndexedSchedule *source = &selected->value;
    ShadowSpillIndexedSchedule *destination = &result->selected_schedule;
    *destination = *source;
    memset(source, 0, sizeof(*source));
    selected->action_capacity = 0U;
    selected->initial_capacity = 0U;
    selected->final_capacity = 0U;
    return 0;
}

