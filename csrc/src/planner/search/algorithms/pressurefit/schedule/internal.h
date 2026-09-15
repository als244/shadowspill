#ifndef SHADOWSPILL_SCHEDULE_INTERNAL_H
#define SHADOWSPILL_SCHEDULE_INTERNAL_H

/*
 * Where a plan's transfers go: the pressure a span is under, the trigger each
 * reload is given, and the actions those choices emit.
 *
 * The order is the order the emitter runs in. Facts index the program's
 * writes once; pressure prices what a span holds; triggers choose when each
 * reload fires; the clamp moves triggers that do not fit; fetches move one
 * action within an existing schedule; and emit turns all of it into the
 * schedule the simulator is handed.
 */

#include "../../../../internal.h"
#include "../candidates_internal.h"
#include "../residency_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct Span {
    uint32_t start;
    uint32_t end;
} Span;

typedef struct Reload {
    uint32_t alias;
    uint32_t earliest_trigger;
    uint32_t latest_trigger;
    uint32_t entry_boundary;
    uint32_t ordinal;
    uint32_t trigger;
} Reload;

typedef struct ReloadRank {
    uint32_t index;
    uint32_t entry_boundary;
    uint64_t size_bytes;
    uint32_t alias;
} ReloadRank;

typedef struct Action {
    uint32_t trigger;
    uint32_t alias;
    uint8_t kind;
} Action;

uint64_t shadowspill_schedule_cell(uint32_t alias, uint32_t count, uint32_t index);

int shadowspill_schedule_checked_cells(uint32_t left, uint32_t right, size_t *result);

int shadowspill_schedule_reserve_actions(
    ShadowSpillScheduleStorage *storage,
    uint32_t capacity
);

void shadowspill_schedule_storage_clear(ShadowSpillScheduleStorage *storage);

uint32_t shadowspill_schedule_collect_spans(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    Span *spans
);

int shadowspill_schedule_build_pressure(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    int fetch_headroom,
    uint64_t *pressure
);

uint32_t shadowspill_schedule_event_min_task(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
);

uint32_t shadowspill_schedule_event_max_task(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
);

int shadowspill_schedule_has_write_since(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    int32_t refreshed_at,
    int32_t through
);

int shadowspill_schedule_reload_rank_compare(const void *left_value, const void *right_value);

void shadowspill_schedule_clear_active_reload(
    uint64_t *active,
    uint32_t word_count,
    uint32_t rank,
    uint32_t start,
    uint32_t end
);

uint32_t shadowspill_schedule_first_active_reload(
    const uint64_t *active,
    uint32_t word_count,
    uint32_t boundary,
    const ReloadRank *ranked
);

void shadowspill_schedule_choose_latest_safe_triggers(
    const ShadowSpillScheduleFacts *facts,
    Reload *reloads,
    uint32_t reload_count
);

void shadowspill_schedule_choose_packed_triggers(
    const ShadowSpillScheduleFacts *facts,
    Reload *reloads,
    uint32_t reload_count
);

int shadowspill_schedule_clamp_triggers_to_fit(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    Reload *reloads,
    uint32_t reload_count,
    int fetch_headroom
);

int shadowspill_schedule_action_compare(const void *left_value, const void *right_value);

int shadowspill_schedule_copy_actions(
    const ShadowSpillScheduleFacts *facts,
    const Action *actions,
    uint32_t action_count,
    int coalesced,
    ShadowSpillScheduleStorage *storage
);

#endif  /* SHADOWSPILL_SCHEDULE_INTERNAL_H */
