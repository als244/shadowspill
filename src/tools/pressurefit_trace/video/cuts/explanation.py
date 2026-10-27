"""Draw a review card for the recorded cut comparator."""

import textwrap

from ...data import GIB, byte_label
from ...explain import decision


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
