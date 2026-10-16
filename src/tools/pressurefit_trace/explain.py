"""Explain the recorded comparator, without reimplementing cut selection."""

from __future__ import annotations

import textwrap

from .data import GIB, byte_label


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


def draw_explanation(fig, data, reduction, index, *, pressure_budget=None):
    """A readable pause over the focus panel; pressure remains visible below."""
    value = decision(data, reduction, index)
    ax = fig.add_axes((0.028, 0.275, 0.755, 0.365), zorder=30, facecolor="#142235")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30455f")
    ax.text(
        0.02,
        0.955,
        "1  Choose the most over-budget boundary",
        fontsize=12,
        weight="bold",
        va="top",
    )
    ax.text(
        0.025,
        0.817,
        r"$p^* = \operatorname{argmax}_{p:\ U_p>C_p}\ (U_p-C_p)$",
        fontsize=13,
    )
    b = value["boundaries"][0]
    pressure_offset = (
        0 if pressure_budget is None else pressure_budget - b["capacity_bytes"]
    )
    ax.text(
        0.43,
        0.831,
        f"b{b['position']}: {(b['used_bytes'] + pressure_offset) / GIB:.3f} - "
        f"{(b['capacity_bytes'] + pressure_offset) / GIB:.3f}"
        f" = {b['excess_bytes'] / GIB:.3f} GiB excess",
        fontsize=11,
        color="#ff9aa6",
    )
    ax.text(
        0.43,
        0.774,
        "Ties go to the earlier boundary. Top three are listed at right.",
        fontsize=9,
        color="#a6b9d1",
    )
    ax.text(
        0.02,
        0.693,
        "2  Rank legal cuts relieving that boundary, in priority order",
        fontsize=12,
        weight="bold",
    )
    symbols = {
        "exposed_ns": "E",
        "writeback": "w",
        "negative_size_bytes": "-S",
        "alias": "a",
        "start": "s",
        "cut_id": "id",
        "negative_slack_ns": r"-\sigma",
    }
    key = ", ".join(symbols[name] for name in value["ranking_criteria"])
    legend = (
        "w: writeback   S: bytes\na / s / id: stable tie-breaks\n"
        + r"$\sigma=S_e-wT_e+S_f-T_f$"
        + "  (maximize)"
    )
    ax.text(
        0.025,
        0.604,
        r"$K(c)=(" + key + r")$" + "   smallest lexicographic key wins",
        fontsize=12,
    )
    ax.text(
        0.025,
        0.524,
        r"$E=\max(0,\ wT_e-S_e)+\max(0,\ T_f-S_f)$",
        fontsize=11,
    )
    ax.text(
        0.025,
        0.480,
        r"$S_e=\max(0,t_{task,start}-t_{prior,end}),\quad "
        r"S_f=\max(0,t_{next,start}-t_{task,end})$",
        fontsize=9.5,
        color="#a6b9d1",
    )
    ax.text(
        0.555,
        0.531,
        legend,
        fontsize=8.5,
        color="#a6b9d1",
        linespacing=1.4,
    )
    headers = ["Object / cut", "E (ns) ↓", "w ↓", "Slack (ms) ↑", "S ↑", "a / s / id ↓"]
    xs = [0.025, 0.395, 0.50, 0.61, 0.71, 0.83]
    field_column = {
        "exposed_ns": 1,
        "writeback": 2,
        "negative_slack_ns": 3,
        "negative_size_bytes": 4,
        "alias": 5,
        "start": 5,
        "cut_id": 5,
    }
    for x, heading in zip(xs, headers, strict=True):
        ax.text(x, 0.422, heading, fontsize=9, color="#a6b9d1", weight="bold")
    for i, cut in enumerate(value["choices"]):
        y = 0.345 - i * 0.076
        if i == 0:
            ax.axhspan(y - 0.016, y + 0.050, color="#64dec1", alpha=0.10)
        cells = [
            textwrap.shorten(cut["semantic_name"], width=41, placeholder="…"),
            str(cut["exposed_ns"]) if value["uses_exposed_score"] else "unused",
            str(cut["writeback"]),
            f"{cut['slack_ns'] / 1e6:,.3f}",
            byte_label(cut["size_bytes"]),
            f"{cut['rank_alias']} / {cut['rank_start']} / {cut['cut_id']}",
        ]
        decisive = field_column[cut["first_difference"]["criterion"]] if i else None
        for col, x in enumerate(xs):
            ax.text(
                x,
                y,
                cells[col],
                fontsize=8.5 if col == 5 else 9,
                color="#ffb97b"
                if col == decisive
                else "#64dec1"
                if i == 0
                else "#e9f0fa",
                weight="bold" if col == decisive or i == 0 else "normal",
            )
    ax.text(
        0.025,
        0.026,
        "Winner in green. Orange marks each alternative's first losing criterion. "
        "Later fields matter only on a tie.",
        fontsize=8.5,
        color="#a6b9d1",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax
