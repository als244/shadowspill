#include "internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct ResidencyCut {
    uint32_t alias;
    int32_t start;
    int32_t end;
} ResidencyCut;

typedef struct CutScore {
    uint64_t exposed_ns;
    int64_t values[7];
} CutScore;

static uint64_t cell(uint32_t alias, uint32_t boundary_count, uint32_t index) {
    return (uint64_t)alias * boundary_count + index;
}

static int find_span(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    uint32_t index,
    uint32_t *start,
    uint32_t *end
) {
    uint64_t position = cell(alias, boundary_count, index);
    if (resident[position] == 0U) {
        return 0;
    }
    uint32_t left = index;
    while (left > 0U) {
        uint64_t previous = cell(alias, boundary_count, left - 1U);
        if (resident[previous] == 0U || breaks[previous] != 0U) {
            break;
        }
        --left;
    }
    uint32_t right = index;
    while (right + 1U < boundary_count) {
        uint64_t current = cell(alias, boundary_count, right);
        uint64_t next = cell(alias, boundary_count, right + 1U);
        if (resident[next] == 0U || breaks[current] != 0U) {
            break;
        }
        ++right;
    }
    *start = left;
    *end = right;
    return 1;
}

static void alias_contribution(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint8_t *contribution
) {
    uint32_t count = problem->boundary_count;
    memset(contribution, 0, count);
    uint32_t index = 0U;
    while (index < count) {
        uint32_t start = 0U;
        uint32_t end = 0U;
        if (!find_span(
                resident,
                breaks,
                alias,
                count,
                index,
                &start,
                &end
            )) {
            ++index;
            continue;
        }
        uint32_t charged_start = start;
        if (options->prefetch_headroom != 0U && start > 0U &&
            problem->productions[cell(alias, count, start)] == 0U) {
            --charged_start;
        }
        int32_t charged_end = (int32_t)end;
        int32_t end_boundary = (int32_t)end - 1;
        if (end_boundary >= 0 && problem->final_location[alias] != 0) {
            int future_access = 0;
            for (uint32_t anchor = start; anchor <= end; ++anchor) {
                uint32_t task =
                    problem->latest_access_task[cell(alias, count, anchor)];
                if (task != UINT32_MAX && (int32_t)task > end_boundary) {
                    future_access = 1;
                    break;
                }
            }
            if (!future_access) {
                --charged_end;
            }
        }
        if (charged_end >= (int32_t)charged_start) {
            for (uint32_t boundary = charged_start;
                 boundary <= (uint32_t)charged_end;
                 ++boundary) {
                contribution[boundary] = 1U;
            }
        }
        index = end + 1U;
    }
    for (uint32_t boundary = 0U; boundary < count; ++boundary) {
        if (problem->output_reservations[cell(alias, count, boundary)] != 0U) {
            contribution[boundary] = 1U;
        }
    }
}

static int compare_score(
    const CutScore *left,
    const CutScore *right,
    int minimize_transfer
) {
    if (!minimize_transfer && left->exposed_ns != right->exposed_ns) {
        return left->exposed_ns < right->exposed_ns ? -1 : 1;
    }
    for (uint32_t index = 0U; index < 7U; ++index) {
        if (left->values[index] == right->values[index]) {
            continue;
        }
        return left->values[index] < right->values[index] ? -1 : 1;
    }
    return 0;
}

static CutScore score_cut(
    const ShadowSpillResidencyProblem *problem,
    const ResidencyCut *cut,
    int minimize_transfer
) {
    uint32_t alias = cut->alias;
    int32_t departure = cut->start - 1;
    int32_t entry = cut->end + 1;
    int writeback = problem->alias_retain_spill_copy[alias] == 0U;
    if (!writeback && departure >= -1) {
        uint32_t index = (uint32_t)(departure + 1);
        writeback = problem->write_prefix[cell(
            alias,
            problem->boundary_count,
            index
        )] != 0U;
    }
    uint64_t h2d_ns = problem->h2d_runtime_ns[alias];
    uint64_t d2h_ns = writeback ? problem->d2h_runtime_ns[alias] : 0U;
    uint64_t departure_ns =
        departure >= 0 ? problem->task_ideal_end_ns[departure] : 0U;
    int32_t last_task = (int32_t)problem->boundary_count - 2;
    int32_t deadline_task = entry + 1;
    if (deadline_task > last_task) {
        deadline_task = last_task;
    }
    uint64_t deadline_ns = deadline_task > 0
        ? problem->task_ideal_end_ns[deadline_task - 1]
        : 0U;
    uint64_t finish_ns = departure_ns + d2h_ns + h2d_ns;
    uint64_t exposed_ns = finish_ns > deadline_ns ? finish_ns - deadline_ns : 0U;
    int64_t length = cut->end >= cut->start
        ? (int64_t)cut->end - cut->start + 1
        : 0;
    CutScore score = {
        .exposed_ns = minimize_transfer ? 0U : exposed_ns,
        .values = {
            writeback,
            cut->start <= -1 ? -1 : 0,
            -(int64_t)problem->first_input_task[alias],
            -(int64_t)problem->alias_size_bytes[alias],
            -length,
            (int64_t)alias,
            (int64_t)cut->start,
        },
    };
    return score;
}

static uint32_t collect_cuts(
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    const uint32_t *first_required,
    const int32_t *gap_start,
    const int32_t *gap_end,
    uint32_t device,
    int32_t boundary,
    ResidencyCut *cuts
) {
    uint32_t boundary_index = (uint32_t)(boundary + 1);
    uint32_t count = 0U;
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        if (problem->alias_device[alias] != device) {
            continue;
        }
        uint64_t position = cell(
            alias,
            problem->boundary_count,
            boundary_index
        );
        if (resident[position] == 0U) {
            continue;
        }
        if (resident[cell(alias, problem->boundary_count, 0U)] != 0U &&
            problem->initial_location[alias] == 1 &&
            problem->anchors[cell(alias, problem->boundary_count, 0U)] == 0U) {
            if (first_required[alias] != UINT32_MAX &&
                boundary_index < first_required[alias]) {
                cuts[count++] = (ResidencyCut){
                    .alias = alias,
                    .start = -1,
                    .end = (int32_t)first_required[alias] - 2,
                };
                continue;
            }
        }

        int32_t start = boundary;
        int32_t end = boundary;
        if (problem->anchors[position] == 0U) {
            start = gap_start[position];
            end = gap_end[position];
            if (start == INT32_MIN) {
                continue;
            }
        } else {
            uint32_t latest = problem->latest_access_task[position];
            int connected_after = boundary_index + 1U < problem->boundary_count &&
                resident[cell(
                    alias,
                    problem->boundary_count,
                    boundary_index + 1U
                )] != 0U && breaks[position] == 0U;
            int can_split = connected_after &&
                (latest == UINT32_MAX || (int32_t)latest <= boundary);
            if (!can_split) {
                continue;
            }
            start = boundary + 1;
            end = boundary;
        }
        if (start <= -1) {
            continue;
        }
        cuts[count++] = (ResidencyCut){
            .alias = alias,
            .start = start,
            .end = end,
        };
    }
    return count;
}

static void build_cut_geometry(
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t *first_required,
    int32_t *gap_start,
    int32_t *gap_end
) {
    uint32_t count = problem->boundary_count;
    uint64_t cells = (uint64_t)problem->alias_count * count;
    for (uint64_t position = 0U; position < cells; ++position) {
        gap_start[position] = INT32_MIN;
        gap_end[position] = INT32_MIN;
    }
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        first_required[alias] = UINT32_MAX;
        for (uint32_t index = 1U; index < count; ++index) {
            if (problem->anchors[cell(alias, count, index)] != 0U) {
                first_required[alias] = index;
                break;
            }
        }

        uint32_t index = 0U;
        while (index < count) {
            uint32_t span_start = 0U;
            uint32_t span_end = 0U;
            if (!find_span(
                    resident,
                    breaks,
                    alias,
                    count,
                    index,
                    &span_start,
                    &span_end
                )) {
                ++index;
                continue;
            }
            uint32_t cursor = span_start;
            while (cursor <= span_end) {
                if (problem->anchors[cell(alias, count, cursor)] != 0U) {
                    ++cursor;
                    continue;
                }
                uint32_t run_start = cursor;
                while (cursor < span_end &&
                       problem->anchors[cell(alias, count, cursor + 1U)] == 0U) {
                    ++cursor;
                }
                uint32_t run_end = cursor;
                for (uint32_t value = run_start; value <= run_end; ++value) {
                    uint64_t position = cell(alias, count, value);
                    gap_start[position] = (int32_t)run_start - 1;
                    gap_end[position] = (int32_t)run_end - 1;
                }
                ++cursor;
            }
            index = span_end + 1U;
        }
    }
}

static void apply_cut(
    const ShadowSpillResidencyProblem *problem,
    uint8_t *resident,
    uint8_t *breaks,
    const ResidencyCut *cut
) {
    if (cut->start > cut->end) {
        uint32_t after = (uint32_t)(cut->end + 1);
        breaks[cell(cut->alias, problem->boundary_count, after)] = 1U;
        return;
    }
    for (int32_t boundary = cut->start; boundary <= cut->end; ++boundary) {
        uint32_t index = (uint32_t)(boundary + 1);
        resident[cell(cut->alias, problem->boundary_count, index)] = 0U;
    }
}

static int valid_problem(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const ShadowSpillResidencyResult *result
) {
    if (problem == NULL || options == NULL || result == NULL ||
        problem->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        problem->alias_count == 0U || problem->boundary_count == 0U ||
        problem->device_count == 0U) {
        return 0;
    }
    uint64_t cells = (uint64_t)problem->alias_count * problem->boundary_count;
    return problem->alias_size_bytes != NULL && problem->alias_device != NULL &&
        problem->alias_retain_spill_copy != NULL && problem->initial_location != NULL &&
        problem->final_location != NULL && problem->anchors != NULL &&
        problem->productions != NULL && problem->latest_access_task != NULL &&
        problem->output_reservations != NULL && problem->write_prefix != NULL &&
        problem->first_input_task != NULL && problem->h2d_runtime_ns != NULL &&
        problem->d2h_runtime_ns != NULL && problem->task_ideal_end_ns != NULL &&
        problem->device_capacity_bytes != NULL &&
        problem->device_priority != NULL && options->seed_resident != NULL &&
        options->seed_breaks != NULL && options->extra_pressure_bytes != NULL &&
        result->resident != NULL && result->breaks != NULL &&
        result->resident_capacity >= cells && result->break_capacity >= cells;
}

ShadowSpillPlannerStatus shadowspill_reduce_residency(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result
) {
    if (!valid_problem(problem, options, result)) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    uint8_t *resident_output = result->resident;
    uint64_t resident_capacity = result->resident_capacity;
    uint8_t *break_output = result->breaks;
    uint64_t break_capacity = result->break_capacity;
    memset(result, 0, sizeof(*result));
    result->resident = resident_output;
    result->resident_capacity = resident_capacity;
    result->breaks = break_output;
    result->break_capacity = break_capacity;
    result->error_device = UINT32_MAX;
    result->error_boundary = INT32_MIN;

    uint64_t cells = (uint64_t)problem->alias_count * problem->boundary_count;
    memcpy(result->resident, options->seed_resident, cells);
    memcpy(result->breaks, options->seed_breaks, cells);

    uint64_t pressure_cells =
        (uint64_t)problem->device_count * problem->boundary_count;
    uint64_t *pressure = calloc(pressure_cells, sizeof(*pressure));
    uint8_t *before = calloc(problem->boundary_count, sizeof(*before));
    uint8_t *after = calloc(problem->boundary_count, sizeof(*after));
    ResidencyCut *cuts = calloc(problem->alias_count, sizeof(*cuts));
    uint32_t *first_required = calloc(
        problem->alias_count,
        sizeof(*first_required)
    );
    int32_t *gap_start = calloc(cells, sizeof(*gap_start));
    int32_t *gap_end = calloc(cells, sizeof(*gap_end));
    if (pressure == NULL || before == NULL || after == NULL || cuts == NULL ||
        first_required == NULL || gap_start == NULL || gap_end == NULL) {
        free(pressure);
        free(before);
        free(after);
        free(cuts);
        free(first_required);
        free(gap_start);
        free(gap_end);
        result->status = SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }

    build_cut_geometry(
        problem,
        result->resident,
        result->breaks,
        first_required,
        gap_start,
        gap_end
    );

    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        alias_contribution(
            problem,
            options,
            result->resident,
            result->breaks,
            alias,
            before
        );
        uint32_t device = problem->alias_device[alias];
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            if (before[boundary] != 0U) {
                pressure[(uint64_t)device * problem->boundary_count + boundary] +=
                    problem->alias_size_bytes[alias];
            }
        }
    }

    while (1) {
        uint32_t selected_device = UINT32_MAX;
        uint32_t selected_boundary = UINT32_MAX;
        uint64_t selected_excess = 0U;
        uint64_t selected_used = 0U;
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            for (uint32_t device = 0U; device < problem->device_count; ++device) {
                uint64_t position =
                    (uint64_t)device * problem->boundary_count + boundary;
                uint64_t used = pressure[position] +
                    options->extra_pressure_bytes[position];
                uint64_t capacity = problem->device_capacity_bytes[device];
                if (used <= capacity) {
                    continue;
                }
                uint64_t excess = used - capacity;
                int better = selected_device == UINT32_MAX ||
                    excess > selected_excess ||
                    (excess == selected_excess && boundary < selected_boundary) ||
                    (excess == selected_excess && boundary == selected_boundary &&
                     problem->device_priority[device] <
                         problem->device_priority[selected_device]);
                if (better) {
                    selected_device = device;
                    selected_boundary = boundary;
                    selected_excess = excess;
                    selected_used = used;
                }
            }
        }
        if (selected_device == UINT32_MAX) {
            free(pressure);
            free(before);
            free(after);
            free(cuts);
            free(first_required);
            free(gap_start);
            free(gap_end);
            result->status = SHADOWSPILL_PLANNER_OK;
            return SHADOWSPILL_PLANNER_OK;
        }

        int32_t boundary_value = (int32_t)selected_boundary - 1;
        uint32_t cut_count = collect_cuts(
            problem,
            result->resident,
            result->breaks,
            first_required,
            gap_start,
            gap_end,
            selected_device,
            boundary_value,
            cuts
        );
        if (cut_count == 0U) {
            result->status = SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE;
            result->error_device = selected_device;
            result->error_boundary = boundary_value;
            result->required_bytes = selected_used;
            result->capacity_bytes =
                problem->device_capacity_bytes[selected_device];
            free(pressure);
            free(before);
            free(after);
            free(cuts);
            free(first_required);
            free(gap_start);
            free(gap_end);
            return SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE;
        }

        uint32_t chosen = 0U;
        CutScore chosen_score = score_cut(
            problem,
            &cuts[0],
            options->minimize_transfer != 0U
        );
        for (uint32_t index = 1U; index < cut_count; ++index) {
            CutScore candidate = score_cut(
                problem,
                &cuts[index],
                options->minimize_transfer != 0U
            );
            if (compare_score(
                    &candidate,
                    &chosen_score,
                    options->minimize_transfer != 0U
                ) < 0) {
                chosen = index;
                chosen_score = candidate;
            }
        }

        uint32_t alias = cuts[chosen].alias;
        alias_contribution(
            problem,
            options,
            result->resident,
            result->breaks,
            alias,
            before
        );
        apply_cut(problem, result->resident, result->breaks, &cuts[chosen]);
        alias_contribution(
            problem,
            options,
            result->resident,
            result->breaks,
            alias,
            after
        );
        uint32_t device = problem->alias_device[alias];
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            int64_t delta = (int64_t)after[boundary] - before[boundary];
            if (delta < 0) {
                pressure[(uint64_t)device * problem->boundary_count + boundary] -=
                    problem->alias_size_bytes[alias];
            } else if (delta > 0) {
                pressure[(uint64_t)device * problem->boundary_count + boundary] +=
                    problem->alias_size_bytes[alias];
            }
        }
    }
}
