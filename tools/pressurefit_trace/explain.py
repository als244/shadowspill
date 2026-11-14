"""Explain the recorded comparator, without reimplementing cut selection."""

from __future__ import annotations


def decision(data, reduction, index):
    row = data.trace.reductions[reduction].cuts[index]
    choices = row[3:51].reshape(4, 12)[: int(row[2]) + 1]
    comparisons = data.trace.comparisons(reduction, index)
    cuts = []
    for rank, choice in enumerate(choices):
        signed = [int(x) for x in choice[5:].view("<i8")]
        cuts.append(
            {
                "rank": rank,
                "cut_id": int(choice[0]),
                "alias_index": int(choice[1]),
                "semantic_name": data.aliases[int(choice[1])]["label"],
                "start": int(choice[2:3].view("<i8")[0]),
                "end": int(choice[3:4].view("<i8")[0]),
                "exposed_ns": int(choice[4]),
                "writeback": signed[0],
                "size_bytes": -signed[3],
                "rank_alias": signed[5],
                "rank_start": signed[6],
                "first_difference": comparisons[rank - 1] if rank else None,
            }
        )
        cuts[-1]["score_components"] = dict(
            zip(
                (
                    "prior_use_end_ns",
                    "next_use_begin_ns",
                    "evict_ns",
                    "fetch_ns",
                    "evict_slack_ns",
                    "fetch_slack_ns",
                    "evict_exposed_ns",
                    "fetch_exposed_ns",
                ),
                (int(word) for word in row[64 + rank * 8 : 72 + rank * 8]),
                strict=True,
            )
        )
        components = cuts[-1]["score_components"]
        cuts[-1]["slack_ns"] = (
            components["evict_slack_ns"]
            - components["evict_ns"]
            + components["fetch_slack_ns"]
            - components["fetch_ns"]
        )
    return {
        "reduction": reduction,
        "cut": index + 1,
        "choices": cuts,
        "uses_exposed_score": int(data.trace.header[2]) not in (1, 3),
        "cut_score": data.trace.cut_score,
        "cut_ranking": data.trace.cut_ranking,
        "ranking_criteria": list(data.trace.ranking_criteria),
        "boundaries": [
            {
                "position": int(p),
                "used_bytes": int(u),
                "capacity_bytes": int(c),
                "excess_bytes": int(e),
            }
            for p, u, c, e in row[52:64].reshape(3, 4)[: int(row[51])]
        ],
    }
