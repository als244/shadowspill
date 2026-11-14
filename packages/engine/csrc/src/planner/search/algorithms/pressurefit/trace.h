#ifndef SHADOWSPILL_PRESSUREFIT_TRACE_H
#define SHADOWSPILL_PRESSUREFIT_TRACE_H

#include "residency_internal.h"
#include <stdio.h>

#define SHADOWSPILL_TRACE_CUT_WORDS 96U

/* Private, bounded writer buffers. A file belongs to one candidate thread. */
typedef struct ShadowSpillPressureFitTrace {
    FILE *file;
    int failed;
    uint64_t *capacity;
    uint64_t *extra;
    uint64_t *pressure;
    uint64_t reduction_count;
    uint64_t current_reduction;
    uint64_t current_step;
    uint64_t kept_step;
    uint64_t answer_step;
    uint64_t cut_count;
    uint64_t cut_size;
    /* Why the next reduction starts: kind, status, task, alias, location,
     * requested, used, capacity, shortfall. No device identity is recorded. */
    uint64_t reason[9];
    uint32_t buffered_cuts;
    uint32_t cut_words;
    uint32_t buffered_ops;
    uint64_t cuts[128][SHADOWSPILL_TRACE_CUT_WORDS];
    uint64_t ops[512][4];
} ShadowSpillPressureFitTrace;

ShadowSpillPressureFitTrace *shadowspill_candidate_trace_open(
    const char *directory, uint32_t problem_index, uint8_t strategy,
    uint8_t rule,
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillResidency *seed
);
int shadowspill_candidate_trace_close(
    ShadowSpillPressureFitTrace *trace,
    const ShadowSpillPressureFitCandidateDiagnostic *diagnostic
);
ShadowSpillStatus shadowspill_candidate_trace_reduce(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result,
    ShadowSpillPressureFitResidencyWorkspace *workspace
);
int shadowspill_candidate_trace_cut(
    ShadowSpillPressureFitTrace *trace,
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint32_t device, uint32_t boundary, uint64_t used, uint32_t chosen,
    int minimize_transfer
);
void shadowspill_candidate_trace_pressure_add(
    ShadowSpillPressureFitTrace *trace,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint32_t device, uint32_t first, uint32_t last, uint64_t delta
);
void shadowspill_candidate_trace_step(
    ShadowSpillPressureFitTrace *trace, uint64_t step
);
#endif
