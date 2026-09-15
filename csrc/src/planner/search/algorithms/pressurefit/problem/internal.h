#ifndef SHADOWSPILL_PROBLEM_INTERNAL_H
#define SHADOWSPILL_PROBLEM_INTERNAL_H

/*
 * One PressureFit problem: everything the search reads, derived once.
 *
 * A problem is prepared before any candidate is placed, and nothing in it
 * changes while the search runs. Buffers holds the arrays it owns; facts
 * derives what the program says about its tasks and aliases; floor proves
 * the residency the plan cannot go below; placement chooses what starts
 * resident; and search is the entry the caller reaches all of it through.
 */

#include "../../../../../common/platform.h"
#include "../../../../internal.h"
#include "../candidates_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct PreparedProblem {
    ShadowSpillPressureFitResidencyProblem residency;
    ShadowSpillPressureFitProblem problem;

    int8_t *initial_location;
    int8_t *final_location;
    uint8_t *anchors;
    uint8_t *productions;
    uint32_t *latest_access_task;
    ShadowSpillPressureFitResidencySparseLists sparse;
    uint8_t *output_reservations;
    uint8_t *write_prefix;
    uint32_t *first_input_task;
    uint64_t *fetch_runtime_ns;
    uint64_t *evict_runtime_ns;
    uint64_t *task_ideal_end_ns;
    uint64_t *device_capacity_bytes;
    uint64_t *boundary_capacity_bytes;
    uint8_t *seed_resident;
    uint8_t *seed_breaks;

    uint32_t *first_access_task;
    uint8_t *produced;
    uint8_t *seen_input;
    uint8_t *charged_anchors;
    uint64_t *required_bytes;
    /* Which aliases the reducer may cut; for the ones it may not, the
       trigger of their one fetch, the resident slice reserved for them
       (bytes per device), and what they add up to. The eligibility and the
       slice move to the result once evaluation is done. */
    uint8_t *evict_eligible;
    uint32_t *fixed_fetch_trigger;
    uint64_t *resident_slice_bytes;
    uint32_t evict_ineligible_aliases;
    uint64_t evict_ineligible_bytes;

    uint8_t failure_kind;
    uint32_t error_device;
    uint32_t error_alias;
    int32_t error_boundary;
    uint64_t failure_required_bytes;
    uint64_t failure_capacity_bytes;
} PreparedProblem;

int shadowspill_problem_allocate_prepared_buffers(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

void shadowspill_problem_prepared_problem_destroy(PreparedProblem *prepared);

int shadowspill_problem_add_u64(uint64_t left, uint64_t right, uint64_t *result);

int shadowspill_problem_compare_u32(uint32_t left, uint32_t right);

int shadowspill_problem_compare_u64(uint64_t left, uint64_t right);

int shadowspill_problem_program_problem_valid(
    const ShadowSpillIndexedProblem *problem,
    const ShadowSpillPressureFitOptions *options
);

int shadowspill_problem_bind_residency(
    uint32_t alias_count,
    uint32_t value_count,
    const uint32_t *aliases,
    const uint8_t *locations,
    int8_t *destination
);

ShadowSpillStatus shadowspill_problem_build_sparse_lists(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

ShadowSpillStatus shadowspill_problem_derive_task_facts(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

ShadowSpillStatus shadowspill_problem_finalize_boundary_capacities(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

ShadowSpillStatus shadowspill_problem_finalize_alias_facts(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

ShadowSpillStatus shadowspill_problem_validate_required_floor(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

ShadowSpillStatus shadowspill_problem_reserve_resident_slice(
    const ShadowSpillIndexedProblem *source,
    const ShadowSpillPressureFitOptions *options,
    PreparedProblem *prepared
);

void shadowspill_problem_build_anchor_seed(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

ShadowSpillStatus shadowspill_problem_greedily_place_initial_aliases(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
);

#endif  /* SHADOWSPILL_PROBLEM_INTERNAL_H */
