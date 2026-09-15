/* Where a candidate's time and work went, section by section. */
#include "internal.h"

Section shadowspill_candidate_section_open(uint64_t *sink) {
    return (Section){shadowspill_monotonic_ns(), sink};
}

void shadowspill_candidate_section_close(Section section) {
    *section.sink += shadowspill_monotonic_ns() - section.started;
}

/* What the named sections did not claim. Reported so the parts add up. */
void shadowspill_candidate_section_close_total(
    ShadowSpillPressureFitSectionTiming *timing,
    uint64_t started
) {
    timing->total_ns = shadowspill_monotonic_ns() - started;
    const uint64_t named = timing->prepare_ns + timing->setup_ns +
        timing->reduce_ns + timing->emit_ns + timing->simulate_ns +
        timing->repair_ns + timing->digest_ns + timing->place_ns +
        timing->select_ns + timing->teardown_ns;
    timing->residual_ns =
        timing->total_ns > named ? timing->total_ns - named : 0U;
}

/* Append one step of a candidate's descent, growing the record as needed. */
int shadowspill_candidate_record_step(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    ShadowSpillPressureFitReductionStep step
) {
    if (diagnostic->step_count == diagnostic->step_capacity) {
        uint32_t capacity = diagnostic->step_capacity == 0U
            ? 32U
            : diagnostic->step_capacity * 2U;
        if (capacity < diagnostic->step_capacity) {
            return -1;
        }
        ShadowSpillPressureFitReductionStep *grown = realloc(
            diagnostic->steps, (size_t)capacity * sizeof(*grown)
        );
        if (grown == NULL) {
            return -1;
        }
        diagnostic->steps = grown;
        diagnostic->step_capacity = capacity;
    }
    diagnostic->steps[diagnostic->step_count++] = step;
    return 0;
}

/* Move the cuts a reduction reported into the candidate's flat record. */
int shadowspill_candidate_drain_cuts(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    CandidateWorkspace *workspace
) {
    const uint64_t added = workspace->cut_scratch_count;
    workspace->cut_scratch_count = 0U;
    if (added == 0U) {
        return 0;
    }
    if (diagnostic->cut_count + added > diagnostic->cut_capacity) {
        uint32_t capacity = diagnostic->cut_capacity == 0U
            ? 256U
            : diagnostic->cut_capacity;
        while (diagnostic->cut_count + added > capacity) {
            const uint32_t grown = capacity * 2U;
            if (grown < capacity) {
                return -1;
            }
            capacity = grown;
        }
        uint32_t *aliases = realloc(
            diagnostic->cut_aliases, (size_t)capacity * sizeof(*aliases)
        );
        if (aliases == NULL) {
            return -1;
        }
        diagnostic->cut_aliases = aliases;
        diagnostic->cut_capacity = capacity;
    }
    memcpy(
        diagnostic->cut_aliases + diagnostic->cut_count,
        workspace->cut_scratch,
        (size_t)added * sizeof(*workspace->cut_scratch)
    );
    diagnostic->cut_count += (uint32_t)added;
    return 0;
}

/* Mark the last recorded step, for facts only known after it was taken. */
void shadowspill_candidate_mark_last_step(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    uint32_t flags,
    uint64_t required_bytes
) {
    if (diagnostic->step_count == 0U) {
        return;
    }
    ShadowSpillPressureFitReductionStep *step =
        &diagnostic->steps[diagnostic->step_count - 1U];
    step->flags |= flags;
    if (required_bytes != 0U) {
        step->required_bytes = required_bytes;
    }
}

uint64_t shadowspill_candidate_repair_total(
    const ShadowSpillPressureFitRepairDiagnostics *repairs
) {
    return repairs->admission_fetch_advance_attempts +
        repairs->admission_fetch_delay_attempts +
        repairs->admission_pressure_boundary_attempts +
        repairs->simulation_fetch_delay_attempts +
        repairs->simulation_pressure_boundary_attempts;
}

ShadowSpillPressureFitWorkDiagnostics shadowspill_candidate_workspace_work(
    const CandidateWorkspace *workspace
) {
    return (ShadowSpillPressureFitWorkDiagnostics){
        .schedule_emissions = workspace->schedule_emissions,
        .schedule_cache_hits = workspace->schedule_cache_hits,
        .simulation_calls = workspace->simulation_calls,
        .simulation_cache_hits = workspace->simulation_cache_hits,
        .admission_calls = workspace->admission.calls,
        .sections = workspace->sections,
    };
}

static ShadowSpillPressureFitSectionTiming section_delta(
    ShadowSpillPressureFitSectionTiming after,
    ShadowSpillPressureFitSectionTiming before
) {
    return (ShadowSpillPressureFitSectionTiming){
        .prepare_ns = after.prepare_ns - before.prepare_ns,
        .setup_ns = after.setup_ns - before.setup_ns,
        .reduce_ns = after.reduce_ns - before.reduce_ns,
        .emit_ns = after.emit_ns - before.emit_ns,
        .simulate_ns = after.simulate_ns - before.simulate_ns,
        .repair_ns = after.repair_ns - before.repair_ns,
        .digest_ns = after.digest_ns - before.digest_ns,
        .place_ns = after.place_ns - before.place_ns,
        .select_ns = after.select_ns - before.select_ns,
        .teardown_ns = after.teardown_ns - before.teardown_ns,
        .admit_ns = after.admit_ns - before.admit_ns,
    };
}

ShadowSpillPressureFitWorkDiagnostics shadowspill_candidate_work_delta(
    ShadowSpillPressureFitWorkDiagnostics after,
    ShadowSpillPressureFitWorkDiagnostics before
) {
    return (ShadowSpillPressureFitWorkDiagnostics){
        .schedule_emissions =
            after.schedule_emissions - before.schedule_emissions,
        .schedule_cache_hits =
            after.schedule_cache_hits - before.schedule_cache_hits,
        .simulation_calls = after.simulation_calls - before.simulation_calls,
        .simulation_cache_hits =
            after.simulation_cache_hits - before.simulation_cache_hits,
        .admission_calls = after.admission_calls - before.admission_calls,
        .sections = section_delta(after.sections, before.sections),
    };
}

ShadowSpillPressureFitWorkDiagnostics shadowspill_candidate_add_work(
    ShadowSpillPressureFitWorkDiagnostics total,
    ShadowSpillPressureFitWorkDiagnostics part
) {
    total.schedule_emissions += part.schedule_emissions;
    total.schedule_cache_hits += part.schedule_cache_hits;
    total.simulation_calls += part.simulation_calls;
    total.simulation_cache_hits += part.simulation_cache_hits;
    total.admission_calls += part.admission_calls;
    total.sections.prepare_ns += part.sections.prepare_ns;
    total.sections.setup_ns += part.sections.setup_ns;
    total.sections.reduce_ns += part.sections.reduce_ns;
    total.sections.emit_ns += part.sections.emit_ns;
    total.sections.simulate_ns += part.sections.simulate_ns;
    total.sections.repair_ns += part.sections.repair_ns;
    total.sections.digest_ns += part.sections.digest_ns;
    total.sections.place_ns += part.sections.place_ns;
    total.sections.select_ns += part.sections.select_ns;
    total.sections.teardown_ns += part.sections.teardown_ns;
    total.sections.admit_ns += part.sections.admit_ns;
    /* The total and the residual add like every other span, which is what
     * keeps total == named + residual true of the sum. */
    total.sections.total_ns += part.sections.total_ns;
    total.sections.residual_ns += part.sections.residual_ns;
    return total;
}

void shadowspill_candidate_add_repairs(
    ShadowSpillPressureFitRepairDiagnostics *destination,
    const ShadowSpillPressureFitRepairDiagnostics *source
) {
    destination->admission_fetch_advance_attempts +=
        source->admission_fetch_advance_attempts;
    destination->admission_fetch_delay_attempts +=
        source->admission_fetch_delay_attempts;
    destination->admission_pressure_boundary_attempts +=
        source->admission_pressure_boundary_attempts;
    destination->simulation_fetch_delay_attempts +=
        source->simulation_fetch_delay_attempts;
    destination->simulation_pressure_boundary_attempts +=
        source->simulation_pressure_boundary_attempts;
}

/* FNV-1a over `data`, eight bytes per step and a byte-wise tail. Two
 * multipliers give the two independent halves of a fingerprint. */
