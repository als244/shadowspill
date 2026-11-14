/* Versioned little-endian array blocks; no struct padding reaches disk. */
#include "internal.h"
#include "../trace.h"
#include "../residency/internal.h"

static void words(ShadowSpillPressureFitTrace *trace, const uint64_t *values, uint64_t count) {
    unsigned char bytes[8192];
    while (count != 0U && !trace->failed) {
        const uint64_t take = count < 1024U ? count : 1024U;
        for (uint64_t i = 0U; i < take; ++i)
        for (uint32_t byte = 0U; byte < 8U; ++byte) {
            bytes[8U * i + byte] = (unsigned char)(values[i] >> (8U * byte));
        }
        if (fwrite(bytes, 8U, (size_t)take, trace->file) != take) {
            trace->failed = 1;
        }
        values += take;
        count -= take;
    }
}

static void block(ShadowSpillPressureFitTrace *trace, uint32_t kind, uint32_t width, uint64_t count) {
    const uint64_t header[] = {kind | ((uint64_t)width << 32U), count};
    words(trace, header, 2U);
}

static void record(ShadowSpillPressureFitTrace *trace, uint32_t kind, uint32_t width, const uint64_t *row) {
    block(trace, kind, width, 1U);
    words(trace, row, width);
}

static void flush_cuts(ShadowSpillPressureFitTrace *trace) {
    if (trace->buffered_cuts != 0U) {
        block(trace, 6U, trace->cut_words, trace->buffered_cuts);
        for (uint32_t row = 0U; row < trace->buffered_cuts; ++row) {
            words(trace, trace->cuts[row], trace->cut_words);
        }
        trace->buffered_cuts = 0U;
    }
}

static void flush_ops(ShadowSpillPressureFitTrace *trace) {
    if (trace->buffered_ops != 0U) {
        block(trace, 7U, 4U, trace->buffered_ops);
        words(trace, &trace->ops[0][0], (uint64_t)trace->buffered_ops * 4U);
        trace->buffered_ops = 0U;
    }
}

ShadowSpillPressureFitTrace *shadowspill_candidate_trace_open(
    const char *directory, uint32_t problem_index, uint8_t strategy,
    uint8_t rule,
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillResidency *seed
) {
    if (problem->device_count != 1U) { return NULL; }
    const size_t length = strlen(directory);
    if (length > SIZE_MAX - 80U) {
        return NULL;
    }
    char *path = malloc(length + 80U);
    ShadowSpillPressureFitTrace *trace = calloc(1U, sizeof(*trace));
    if (path == NULL || trace == NULL) {
        free(path);
        free(trace);
        return NULL;
    }
    (void)snprintf(path, length + 80U, "%s/p%u_s%u_r%u.pftrace",
        directory, problem_index, strategy, rule);
    trace->file = fopen(path, "wbx");
    free(path);
    const uint64_t cells = (uint64_t)problem->device_count * problem->boundary_count;
    trace->capacity = calloc((size_t)(cells != 0U ? cells : 1U), sizeof(uint64_t));
    trace->extra = calloc((size_t)(cells != 0U ? cells : 1U), sizeof(uint64_t));
    trace->pressure = calloc((size_t)(cells != 0U ? cells : 1U), sizeof(uint64_t));
    if (trace->file == NULL || trace->capacity == NULL || trace->extra == NULL || trace->pressure == NULL) {
        if (trace->file != NULL) { (void)fclose(trace->file); }
        free(trace->capacity);
        free(trace->extra);
        free(trace->pressure);
        free(trace);
        return NULL;
    }
    trace->current_step = trace->kept_step = trace->answer_step = UINT64_MAX;
    trace->reason[2] = trace->reason[3] = UINT64_MAX;
    trace->failed = fwrite("PFTRACE\0", 1U, 8U, trace->file) != 8U;
    trace->cut_words = 96U;
    const uint64_t header[] = {4U, problem_index, strategy, rule, 0U,
        problem->alias_count, problem->boundary_count, 2U};
    record(trace, 0U, 8U, header);
    block(trace, 1U, 1U, problem->alias_count);
    uint64_t spans = 0U;
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        words(trace, &problem->alias_size_bytes[alias], 1U);
        spans += seed->aliases[alias].count;
    }
    block(trace, 2U, 3U, spans);
    for (uint32_t alias = 0U; alias < seed->alias_count; ++alias)
    for (uint32_t span = 0U; span < seed->aliases[alias].count; ++span) {
        const uint64_t row[] = {alias, seed->aliases[alias].bounds[2U * span],
            seed->aliases[alias].bounds[2U * span + 1U]};
        words(trace, row, 3U);
    }
    return trace;
}

/* FNV-1a over little-endian uint64 words, including each alias's span count. */
static uint64_t hash_word(uint64_t hash, uint64_t value) {
    for (uint32_t byte = 0U; byte < 8U; ++byte) {
        hash = (hash ^ (unsigned char)(value >> (8U * byte))) * UINT64_C(1099511628211);
    }
    return hash;
}

ShadowSpillStatus shadowspill_candidate_trace_reduce(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    ShadowSpillPressureFitTrace *trace = result->trace;
    if (shadowspill_residency_prepare_seed_geometry(problem, options, workspace) != 0 ||
        shadowspill_residency_prepare_base_pressure(problem, options, workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const uint64_t cells = (uint64_t)problem->device_count * problem->boundary_count;
    if (trace->reduction_count == 0U) {
        block(trace, 3U, 2U, cells);
        const uint64_t *base = workspace->base_pressure[options->fetch_headroom != 0U];
        for (uint64_t position = 0U; position < cells; ++position) {
            trace->capacity[position] = problem->boundary_capacity_bytes[position];
            const uint64_t row[] = {trace->capacity[position], base[position]};
            words(trace, row, 2U);
        }
    }
    trace->current_reduction = trace->reduction_count++;
    trace->cut_count = 0U;
    record(trace, 4U, 1U, &trace->current_reduction);
    record(trace, 12U, 9U, trace->reason);
    uint64_t changes = 0U;
    for (uint64_t p = 0U; p < cells; ++p) {
        changes += trace->capacity[p] != problem->boundary_capacity_bytes[p] ||
            trace->extra[p] != options->extra_pressure_bytes[p];
    }
    block(trace, 5U, 3U, changes);
    for (uint64_t p = 0U; p < cells; ++p) {
        trace->pressure[p] = workspace->base_pressure[options->fetch_headroom != 0U][p] +
            options->extra_pressure_bytes[p];
        if (trace->capacity[p] != problem->boundary_capacity_bytes[p] ||
            trace->extra[p] != options->extra_pressure_bytes[p]) {
            trace->capacity[p] = problem->boundary_capacity_bytes[p];
            trace->extra[p] = options->extra_pressure_bytes[p];
            const uint64_t row[] = {p, trace->capacity[p], trace->extra[p]};
            words(trace, row, 3U);
        }
    }
    const ShadowSpillStatus status = shadowspill_pressurefit_reduce_residency_reusing(
        problem, options, result, workspace);
    flush_cuts(trace);
    flush_ops(trace);
    uint64_t residency_hash = UINT64_C(14695981039346656037);
    for (uint32_t alias = 0U; alias < result->residency->alias_count; ++alias) {
        const ShadowSpillSpanList *spans = &result->residency->aliases[alias];
        residency_hash = hash_word(residency_hash, spans->count);
        for (uint32_t i = 0U; i < spans->count * 2U; ++i) {
            residency_hash = hash_word(residency_hash, spans->bounds[i]);
        }
    }
    uint64_t pressure_hash = UINT64_C(14695981039346656037);
    for (uint32_t device = 0U; device < problem->device_count; ++device)
    for (uint32_t boundary = 0U; boundary < problem->boundary_count; ++boundary) {
        pressure_hash = hash_word(pressure_hash,
            shadowspill_residency_tree_pressure_at(workspace, device, boundary) +
            options->extra_pressure_bytes[(uint64_t)device * problem->boundary_count + boundary]);
    }
    const uint64_t end[] = {trace->current_reduction, status,
        (uint64_t)(int64_t)result->error_boundary, result->required_bytes,
        result->capacity_bytes, residency_hash, pressure_hash};
    record(trace, 8U, 7U, end);
    return trace->failed ? SHADOWSPILL_STATUS_INTERNAL_FAILURE : status;
}

/* Single-device excess order: excess descending, then boundary ascending. */
static int boundary_before(const uint64_t *left, const uint64_t *right) {
    if (left[3] != right[3]) { return left[3] > right[3]; }
    return left[0] < right[0];
}

static void boundaries(ShadowSpillPressureFitTrace *trace,
    const ShadowSpillPressureFitResidencyProblem *problem, uint64_t *row) {
    uint64_t count = 0U;
    uint64_t best[3][4] = {{0}};
    for (uint32_t boundary = 0U; boundary < problem->boundary_count; ++boundary) {
        const uint64_t p = boundary;
        if (trace->pressure[p] <= trace->capacity[p]) { continue; }
        const uint64_t item[] = {boundary, trace->pressure[p], trace->capacity[p],
            trace->pressure[p] - trace->capacity[p]};
        for (uint32_t rank = 0U; rank < 3U; ++rank) {
            if (rank >= count || boundary_before(item, best[rank])) {
                for (uint32_t move = 2U; move > rank; --move) {
                    memcpy(best[move], best[move - 1U], sizeof(best[move]));
                }
                memcpy(best[rank], item, sizeof(best[rank]));
                if (count < 3U) { ++count; }
                break;
            }
        }
    }
    row[51] = count;
    memcpy(row + 52U, best, sizeof(best));
    if (count == 0U || best[0][0] != row[0] || best[0][1] != row[1]) {
        trace->failed = 1;
    }
}

/* Rank all active alternatives with the selector's exact boundary-dependent key. */
int shadowspill_candidate_trace_cut(
    ShadowSpillPressureFitTrace *trace,
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint32_t device, uint32_t boundary, uint64_t used, uint32_t chosen,
    int minimize_transfer
) {
    const CutIndex *index = &workspace->cut_index;
    const uint32_t *refs = index->refs[minimize_transfer != 0];
    uint32_t best[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    CutScore scores[4] = {{0}};
    for (uint32_t node = index->leaf_base + boundary; node >= 1U; node >>= 1U) {
        const uint64_t slot = (uint64_t)device * index->node_count + node;
        for (uint64_t ref = index->node_offsets[slot] + workspace->node_cursors[slot];
             ref < index->node_offsets[slot + 1U]; ++ref) {
            const uint32_t id = refs[ref];
            if (workspace->cut_active[id] == 0U) { continue; }
            const CutScore score = shadowspill_residency_score_at(
                problem, &index->cuts[id], boundary, minimize_transfer);
            for (uint32_t rank = 0U; rank < 4U; ++rank) {
                if (best[rank] == id) { break; }
                const int compared = best[rank] == UINT32_MAX ? -1 :
                    shadowspill_residency_compare_score(
                        &score, &scores[rank], minimize_transfer);
                if (compared < 0 || (compared == 0 && id < best[rank])) {
                    for (uint32_t move = 3U; move > rank; --move) {
                        best[move] = best[move - 1U];
                        scores[move] = scores[move - 1U];
                    }
                    best[rank] = id;
                    scores[rank] = score;
                    break;
                }
            }
        }
    }
    if (best[0] != chosen) { trace->failed = 1; return -1; }
    uint64_t *row = trace->cuts[trace->buffered_cuts++];
    memset(row, 0, SHADOWSPILL_TRACE_CUT_WORDS * sizeof(*row));
    row[0] = boundary; row[1] = used;
    boundaries(trace, problem, row);
    for (uint32_t rank = 0U; rank < 4U && best[rank] != UINT32_MAX; ++rank) {
        const IndexedCut *cut = &index->cuts[best[rank]];
        uint64_t *choice = row + 3U + 12U * rank;
        choice[0] = best[rank]; choice[1] = cut->cut.alias;
        choice[2] = (uint64_t)(int64_t)cut->cut.start;
        choice[3] = (uint64_t)(int64_t)cut->cut.end;
        choice[4] = scores[rank].exposed_ns;
        /* The v3 record reserves words 6, 7 and 9; only active facts are written. */
        choice[5] = (uint64_t)scores[rank].values[0];
        choice[8] = (uint64_t)scores[rank].values[1];
        choice[10] = (uint64_t)scores[rank].values[2];
        choice[11] = (uint64_t)scores[rank].values[3];
        if (trace->cut_words == 96U) {
            const uint64_t begin = boundary == 0U ? 0U : problem->task_ideal_end_ns[boundary - 1U];
            const uint64_t end = boundary + 1U < problem->boundary_count
                ? problem->task_ideal_end_ns[boundary] : begin;
            const CutTiming *timing = &cut->timing;
            const uint64_t components[] = {
                timing->prior_use_end_ns, timing->next_use_begin_ns,
                timing->evict_ns, timing->fetch_ns,
                begin > timing->prior_use_end_ns ? begin - timing->prior_use_end_ns : 0U,
                timing->next_use_begin_ns > end ? timing->next_use_begin_ns - end : 0U,
                scores[rank].evict_exposed_ns, scores[rank].fetch_exposed_ns,
            };
            memcpy(row + 64U + 8U * rank, components, sizeof(components));
        }
        row[2] = rank;
    }
    trace->cut_size = problem->alias_size_bytes[index->cuts[chosen].cut.alias];
    ++trace->cut_count;
    if (trace->buffered_cuts == 128U) { flush_cuts(trace); }
    return trace->failed ? -1 : 0;
}

void shadowspill_candidate_trace_pressure_add(
    ShadowSpillPressureFitTrace *trace,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint32_t device, uint32_t first, uint32_t last, uint64_t delta
) {
    shadowspill_residency_pressure_add(workspace, device, first, last, delta);
    for (uint32_t boundary = first; boundary <= last; ++boundary) {
        trace->pressure[(uint64_t)device * workspace->boundary_count + boundary] += delta;
    }
    uint64_t *row = trace->ops[trace->buffered_ops++];
    row[0] = trace->cut_count - 1U; row[1] = first; row[2] = last;
    row[3] = delta == trace->cut_size ? 1U : UINT64_MAX;
    if (trace->buffered_ops == 512U) { flush_ops(trace); }
}

void shadowspill_candidate_trace_step(ShadowSpillPressureFitTrace *trace, uint64_t step) {
    trace->current_step = step;
    const uint64_t row[] = {step, trace->current_reduction};
    record(trace, 9U, 2U, row);
}

int shadowspill_candidate_trace_close(
    ShadowSpillPressureFitTrace *trace,
    const ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    if (trace == NULL) { return 0; }
    block(trace, 10U, 7U, diagnostic->step_count);
    for (uint32_t i = 0U; i < diagnostic->step_count; ++i) {
        const ShadowSpillPressureFitReductionStep *step = &diagnostic->steps[i];
        const uint64_t row[] = {step->makespan_ns, step->capacity_bytes,
            step->required_bytes, step->flags, step->repairs,
            step->simulation_status, step->capacity_violations};
        words(trace, row, 7U);
    }
    const uint64_t end[] = {diagnostic->status, trace->answer_step, diagnostic->makespan_ns};
    record(trace, 11U, 3U, end);
    const int close_failed = fclose(trace->file) != 0;
    const int failed = trace->failed || close_failed;
    free(trace->capacity); free(trace->extra); free(trace->pressure); free(trace);
    return failed ? -1 : 0;
}
