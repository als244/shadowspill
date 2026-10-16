/* Task clocks, not boundary-number arithmetic, define transfer exposure. */
#include "planner/search/algorithms/pressurefit/residency/internal.h"

#include <stdio.h>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "score canary failed at line %d: %s\n", __LINE__, #condition); \
        return EXIT_FAILURE; \
    } \
} while (0)

int main(void) {
    const uint64_t ends[] = {30U, 80U, 130U, 250U, 300U};
    const uint64_t size[] = {200U}, fetch[] = {200U}, evict[] = {90U};
    uint8_t retain[] = {0U};
    const uint32_t offsets[] = {0U, 2U}, empty[] = {0U, 0U};
    uint32_t positions[] = {1U, 4U}, tasks[] = {1U, 4U};
    ShadowSpillPressureFitResidencyProblem problem = {
        .alias_count = 1U, .boundary_count = 6U, .device_count = 1U,
        .task_ideal_end_ns = ends, .alias_size_bytes = size,
        .fetch_runtime_ns = fetch, .evict_runtime_ns = evict,
        .alias_retain_spill_copy = retain,
        .anchor_offsets = offsets, .anchor_positions = positions,
        .anchor_tasks = tasks, .anchor_earliest = tasks,
        .write_offsets = empty,
    };
    IndexedCut cut = {.cut = {.alias = 0U, .start = 1, .end = 2}};
    cut.score = shadowspill_residency_score_cut(&problem, &cut.cut);
    cut.timing = shadowspill_residency_cut_timing(&problem, &cut.cut);
    REQUIRE(cut.timing.prior_use_end_ns == 80U);  /* input task 1 ends at 80, not 30 */
    REQUIRE(cut.timing.next_use_begin_ns == 250U);
    CutScore score = shadowspill_residency_score_at(&problem, &cut, 2U, 0);
    REQUIRE(score.evict_exposed_ns == 90U);
    REQUIRE(score.fetch_exposed_ns == 80U);      /* excludes task 2's 50 ns */
    REQUIRE(score.exposed_ns == 170U);
    REQUIRE(score.slack_high == -1 && score.slack_low == UINT64_MAX - 169U);
    score = shadowspill_residency_score_at(&problem, &cut, 3U, 0);
    REQUIRE(score.evict_exposed_ns == 40U && score.fetch_exposed_ns == 200U);
    REQUIRE(score.exposed_ns == 240U);          /* same cut, different pressure task */

    /* A long eviction window must not cancel an exposed fetch. */
    cut.timing.evict_ns = 1U;
    score = shadowspill_residency_score_at(&problem, &cut, 3U, 0);
    REQUIRE(score.evict_exposed_ns == 0U && score.exposed_ns == 200U);
    cut.timing.fetch_ns = 10U;
    score = shadowspill_residency_score_at(&problem, &cut, 2U, 0);
    REQUIRE(score.fetch_exposed_ns == 0U);

    /* Production is anchored after task 1; its completion is still 80. */
    positions[0] = 2U;
    cut.cut.start = 2;
    cut.timing = shadowspill_residency_cut_timing(&problem, &cut.cut);
    REQUIRE(cut.timing.prior_use_end_ns == 80U);

    retain[0] = 1U; /* clean release has no outgoing transfer */
    cut.timing = shadowspill_residency_cut_timing(&problem, &cut.cut);
    score = shadowspill_residency_score_at(&problem, &cut, 3U, 0);
    REQUIRE(cut.timing.evict_ns == 0U && score.evict_exposed_ns == 0U);

    retain[0] = 0U;
    cut.cut.start = -1; /* already in spill: no eviction even without retained copies */
    cut.timing = shadowspill_residency_cut_timing(&problem, &cut.cut);
    cut.score = shadowspill_residency_score_cut(&problem, &cut.cut);
    REQUIRE(cut.timing.prior_use_end_ns == 0U && cut.timing.evict_ns == 0U);
    REQUIRE(cut.score.values[0] == 0);

    /* Final residency has a deadline at timeline end, not final-task start. */
    positions[1] = 5U;
    tasks[1] = UINT32_MAX;
    cut.timing = shadowspill_residency_cut_timing(&problem, &cut.cut);
    REQUIRE(cut.timing.next_use_begin_ns == 300U);
    score = shadowspill_residency_score_at(&problem, &cut, 5U, 0);
    REQUIRE(score.fetch_exposed_ns == 200U);
    score = shadowspill_residency_score_at(&problem, &cut, 5U, 1);
    REQUIRE(score.exposed_ns == 0U); /* transfer-only ranking ignores E */

    /* Saturation preserves a total order even at the uint64 limit. */
    cut.timing.evict_ns = UINT64_MAX;
    cut.timing.fetch_ns = UINT64_MAX;
    score = shadowspill_residency_score_at(&problem, &cut, 3U, 0);
    REQUIRE(score.exposed_ns == UINT64_MAX);

    /* Signed slack stays exact beyond int64/uint64; no positive-part clamp. */
    cut.timing.prior_use_end_ns = 0U;
    cut.timing.next_use_begin_ns = UINT64_MAX;
    cut.timing.evict_ns = 0U;
    cut.timing.fetch_ns = 0U;
    score = shadowspill_residency_score_at(&problem, &cut, 3U, 0);
    REQUIRE(score.slack_high == 0 && score.slack_low == UINT64_MAX - 120U);
    cut.timing.next_use_begin_ns = 0U;
    cut.timing.evict_ns = UINT64_MAX;
    cut.timing.fetch_ns = UINT64_MAX;
    score = shadowspill_residency_score_at(&problem, &cut, 3U, 0);
    REQUIRE(score.slack_high == -2 && score.slack_low == 132U);

    /* Writeback precedes slack; exposure precedes both. */
    CutScore clean = {.values = {0, -200, 0, 0}, .slack_low = 5U};
    CutScore dirty = {.values = {1, -400, 1, 0}, .slack_low = 50U};
    REQUIRE(shadowspill_residency_compare_score(&clean, &dirty, 0) < 0);
    clean.exposed_ns = 1U;
    REQUIRE(shadowspill_residency_compare_score(&clean, &dirty, 0) > 0);
    REQUIRE(shadowspill_residency_compare_score(&clean, &dirty, 1) < 0);
    clean.exposed_ns = 0U;
    dirty.values[0] = 0;
    REQUIRE(shadowspill_residency_compare_score(&clean, &dirty, 0) > 0);
    dirty.slack_high = -1;
    dirty.slack_low = UINT64_MAX;
    REQUIRE(shadowspill_residency_compare_score(&clean, &dirty, 0) < 0);
    /* More slack wins before size; larger size wins only when slack ties. */
    CutScore smaller = {.values = {0, -200, 0, 0}, .slack_low = 50U};
    CutScore larger = {.values = {0, -400, 1, 0}, .slack_low = 5U};
    REQUIRE(shadowspill_residency_compare_score(&smaller, &larger, 0) < 0);
    larger.slack_low = smaller.slack_low;
    REQUIRE(shadowspill_residency_compare_score(&smaller, &larger, 0) > 0);
    larger.values[1] = smaller.values[1];
    REQUIRE(shadowspill_residency_compare_score(&smaller, &larger, 0) < 0);
    return EXIT_SUCCESS;
}
