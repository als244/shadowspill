/* A layout that overran the pool, answered where the miss is.
 *
 * The extent of a layout is set by one lease, and that lease sits as high
 * as it does because of the leases live at the same time beneath it. A miss
 * of a few tens of MiB is therefore a few leases overlapping, and the local
 * answer is to move one of them: a fetch's destination lease begins at its
 * trigger task's end, so delaying the trigger one task shortens the lease
 * from the front. Two delays can free the extent-setting lease -- its own,
 * past the leases that end soon after it begins, or that of a fetch it
 * overlaps, past its end -- and the one that frees the most bytes is taken,
 * when that is at least the overrun. The candidate then emits, simulates
 * and measures again; when no delay can close the miss it gives capacity
 * back, which the reducer pays in cuts.
 */
#include "internal.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Diagnostic-only layout tracing, enabled by SHADOWSPILL_LAYOUT_TRACE and
 * never active in normal planning: the lease that set the extent and the
 * largest leases live at the same time, so a miss can be read lease by lease.
 */
static void trace_layout(
    const CandidateWorkspace *workspace, uint64_t shortfall_bytes, uint64_t top
) {
    static _Thread_local int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("SHADOWSPILL_LAYOUT_TRACE") != NULL;
    }
    if (!enabled) {
        return;
    }
    const PlacementWorkspace *place = &workspace->placement;
    const ShadowSpillLeaseLifetime *lifetimes = place->lifetimes;
    const ShadowSpillLeaseIdentity *identities = place->identities;
    const ShadowSpillLeaseLifetime *extent = &lifetimes[top];
    fprintf(
        stderr,
        "layout-trace extent=%llu shortfall=%llu top: lease=%llu bytes=%llu purpose=%u alias=%u task=%u action=%u live=[%llu,%llu)\n",
        (unsigned long long)place->extent_bytes,
        (unsigned long long)shortfall_bytes,
        (unsigned long long)top,
        (unsigned long long)extent->bytes,
        identities[top].purpose,
        identities[top].alias,
        identities[top].task,
        identities[top].action,
        (unsigned long long)extent->start_ns,
        (unsigned long long)extent->end_ns
    );
    for (uint64_t lease = 0U; lease < place->placed_count; ++lease) {
        if (lease == top || place->excluded[lease] != 0U ||
            !(lifetimes[lease].start_ns < extent->end_ns &&
              extent->start_ns < lifetimes[lease].end_ns) ||
            lifetimes[lease].bytes < (64ULL << 20)) {
            continue;
        }
        fprintf(
            stderr,
            "layout-trace   overlaps: lease=%llu bytes=%llu purpose=%u alias=%u task=%u action=%u live=[%llu,%llu) offset=%llu\n",
            (unsigned long long)lease,
            (unsigned long long)lifetimes[lease].bytes,
            identities[lease].purpose,
            identities[lease].alias,
            identities[lease].task,
            identities[lease].action,
            (unsigned long long)lifetimes[lease].start_ns,
            (unsigned long long)lifetimes[lease].end_ns,
            (unsigned long long)place->offsets[lease]
        );
    }
}

/* A fetch whose destination lease can begin one task later than it does. */
typedef struct {
    uint64_t lease;
    uint32_t alias;
    uint32_t trigger;
    uint32_t consumer;
    /* Where the lease would begin after the delay. */
    uint64_t start_ns;
} Delay;

static int overlapping(
    const ShadowSpillLeaseLifetime *left, const ShadowSpillLeaseLifetime *right
) {
    return left->start_ns < right->end_ns && right->start_ns < left->end_ns;
}

/* Whether lease `index` is a fetch destination that can fire a task later,
 * and where its lifetime would then begin. A fetch destination is dated from
 * its trigger task's end, so one task later is the next task's end. */
static int delayable(
    const ShadowSpillScheduleFacts *facts,
    const CandidateWorkspace *workspace,
    const uint64_t *task_end_ns,
    uint64_t index,
    Delay *delay
) {
    const ShadowSpillLeaseIdentity *identity = &workspace->placement.identities[index];
    const ShadowSpillIndexedSchedule *schedule = &workspace->schedule.value;
    if (identity->purpose != SHADOWSPILL_ADMISSION_PURPOSE_FETCH_DESTINATION ||
        identity->action == SHADOWSPILL_PLANNER_NO_INDEX ||
        identity->action >= schedule->action_count ||
        schedule->action_kinds[identity->action] != SHADOWSPILL_MEMORY_FETCH) {
        return 0;
    }
    const uint32_t alias = schedule->action_aliases[identity->action];
    const uint32_t trigger = schedule->action_trigger_tasks[identity->action];
    const uint32_t consumer =
        shadowspill_schedule_next_input_consumer(facts, alias, trigger);
    if (consumer == UINT32_MAX || trigger + 1U >= consumer ||
        trigger + 1U >= facts->task_count) {
        return 0;
    }
    *delay = (Delay){
        .lease = index,
        .alias = alias,
        .trigger = trigger,
        .consumer = consumer,
        .start_ns = task_end_ns[trigger + 1U],
    };
    return 1;
}

/* Diagnostic-only: every measurement, with the occupancy peak the simulation
 * saw, so the extent's fragmentation -- extent over peak -- can be read per
 * plan. Enabled by SHADOWSPILL_LAYOUT_TRACE. */
void shadowspill_candidate_trace_measurement(
    const CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *simulation,
    uint64_t required_bytes,
    uint64_t pool_bytes,
    uint32_t cuts
) {
    static _Thread_local int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("SHADOWSPILL_LAYOUT_TRACE") != NULL;
    }
    if (!enabled) {
        return;
    }
    const ShadowSpillDevicePeak *peak = simulation->device_peaks;
    /* SHADOWSPILL_LAYOUT_DUMP names a directory; every measurement writes its
     * leases there, one CSV per measurement, for packing studies offline. */
    static _Thread_local unsigned dumped = 0U;
    const char *dump = getenv("SHADOWSPILL_LAYOUT_DUMP");
    if (dump != NULL) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/measure_%lu_%u.csv", dump, (unsigned long)pthread_self(), dumped++);
        FILE *file = fopen(path, "w");
        if (file != NULL) {
            const PlacementWorkspace *place = &workspace->placement;
            fprintf(file, "# makespan_ns=%llu required=%llu pool=%llu extent=%llu peak_total=%llu peak_object=%llu peak_workspace=%llu fits=%d\n",
                (unsigned long long)simulation->makespan_ns, (unsigned long long)required_bytes, (unsigned long long)pool_bytes,
                (unsigned long long)place->extent_bytes, (unsigned long long)(peak == NULL ? 0U : peak[0].total_bytes),
                (unsigned long long)(peak == NULL ? 0U : peak[0].object_bytes), (unsigned long long)(peak == NULL ? 0U : peak[0].workspace_bytes),
                required_bytes <= pool_bytes);
            fprintf(file, "lease,bytes,alignment,start_ns,end_ns,purpose,alias,task,action,excluded,offset\n");
            for (uint64_t lease = 0U; lease < place->placed_count; ++lease) {
                fprintf(file, "%llu,%llu,%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu\n",
                    (unsigned long long)lease, (unsigned long long)place->lifetimes[lease].bytes,
                    (unsigned long long)place->lifetimes[lease].alignment,
                    (unsigned long long)place->lifetimes[lease].start_ns, (unsigned long long)place->lifetimes[lease].end_ns,
                    place->identities[lease].purpose, place->identities[lease].alias, place->identities[lease].task,
                    place->identities[lease].action, place->excluded[lease], (unsigned long long)place->offsets[lease]);
            }
            fclose(file);
        }
    }
    fprintf(
        stderr,
        "layout-measure cuts=%u makespan=%llu required=%llu pool=%llu extent=%llu peak_total=%llu peak_object=%llu peak_workspace=%llu fits=%d\n",
        cuts,
        (unsigned long long)simulation->makespan_ns,
        (unsigned long long)required_bytes,
        (unsigned long long)pool_bytes,
        (unsigned long long)workspace->placement.extent_bytes,
        (unsigned long long)(peak == NULL ? 0U : peak[0].total_bytes),
        (unsigned long long)(peak == NULL ? 0U : peak[0].object_bytes),
        (unsigned long long)(peak == NULL ? 0U : peak[0].workspace_bytes),
        required_bytes <= pool_bytes
    );
}

int shadowspill_candidate_move_for_layout(
    const ShadowSpillScheduleFacts *facts,
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *simulation,
    uint64_t shortfall_bytes
) {
    const PlacementWorkspace *place = &workspace->placement;
    const uint64_t top = place->extent_lease;
    if (top == SHADOWSPILL_ADMISSION_NO_LEASE || simulation->task_intervals == NULL ||
        facts->task_count == 0U) {
        return 0;
    }
    uint64_t *task_end_ns = calloc(facts->task_count, sizeof(*task_end_ns));
    if (task_end_ns == NULL) {
        return -1;
    }
    for (uint32_t item = 0U; item < simulation->task_interval_count; ++item) {
        const ShadowSpillTaskInterval *interval = &simulation->task_intervals[item];
        if (interval->task < facts->task_count) {
            task_end_ns[interval->task] = interval->end_ns;
        }
    }
    trace_layout(workspace, shortfall_bytes, top);
    const ShadowSpillLeaseLifetime *lifetimes = place->lifetimes;
    const ShadowSpillLeaseLifetime *extent = &lifetimes[top];
    Delay best = {0};
    uint64_t best_bytes = 0U;
    int found = 0;
    Delay delay;
    /* The extent-setting fetch itself, past every lease that ends by where
     * it would then begin. */
    if (delayable(facts, workspace, task_end_ns, top, &delay)) {
        uint64_t freed = 0U;
        for (uint64_t lease = 0U; lease < place->placed_count; ++lease) {
            if (lease == top || place->excluded[lease] != 0U ||
                !overlapping(&lifetimes[lease], extent)) {
                continue;
            }
            if (lifetimes[lease].end_ns <= delay.start_ns) {
                freed += lifetimes[lease].bytes;
            }
        }
        if (freed != 0U) {
            best = delay;
            best_bytes = freed;
            found = 1;
        }
    }
    /* A fetch the extent-setting lease overlaps, past that lease's end. The
     * largest wins, then the latest trigger, then the lowest lease. */
    for (uint64_t lease = 0U; lease < place->placed_count; ++lease) {
        if (lease == top || place->excluded[lease] != 0U ||
            !overlapping(&lifetimes[lease], extent)) {
            continue;
        }
        if (!delayable(facts, workspace, task_end_ns, lease, &delay) ||
            delay.start_ns < extent->end_ns) {
            continue;
        }
        const uint64_t freed = lifetimes[lease].bytes;
        if (!found || freed > best_bytes ||
            (freed == best_bytes &&
             (delay.trigger > best.trigger ||
              (delay.trigger == best.trigger && lease < best.lease)))) {
            best = delay;
            best_bytes = freed;
            found = 1;
        }
    }
    free(task_end_ns);
    /* A lease drops by at most the bytes it stops overlapping, so a delay
     * that frees less than the overrun cannot close the miss. */
    if (!found || best_bytes < shortfall_bytes) {
        return 0;
    }
    const ShadowSpillFetchTriggerConstraint constraint = {
        .alias = best.alias,
        .consumer_task = best.consumer,
        .minimum_trigger = best.trigger + 1U,
        .maximum_trigger = UINT32_MAX,
    };
    const int recorded =
        shadowspill_candidate_record_fetch_constraint(workspace, constraint);
    if (recorded < 0) {
        return -1;
    }
    return recorded == 0 ? 1 : 0;
}
