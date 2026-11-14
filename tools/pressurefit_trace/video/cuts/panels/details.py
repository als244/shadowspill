"""Draw the details panel without changing replay state."""

from __future__ import annotations

import textwrap

from ....data import GIB, ROLE_COLORS, byte_label, short_task
from ...style import GREEN, MUTED, WHITE


def explain_comparison(comparison):
    field = comparison["criterion"]
    a, b = comparison["chosen_value"], comparison["alternative_value"]
    if field == "exposed_ns":
        return f"Less exposed transfer time\n{a:,} vs {b:,} ns"
    if field == "writeback":
        return "No writeback needed\nAlternative requires writeback"
    if field == "negative_slack_ns":
        return f"More signed transfer slack\n{-a / 1e6:,.3f} vs {-b / 1e6:,.3f} ms"
    if field == "negative_size_bytes":
        return f"Larger object\n{byte_label(-a)} vs {byte_label(-b)}"
    return f"Tie-break: {field.replace('_', ' ')}\n{a} precedes {b}"


def draw_details(self, view, rid, count, event, pressure, capacity):
    d = self.data
    row = view.row
    selected = view.selected
    position = view.position
    self.style(self.side, ticks=False)
    self.side.set_xlim(0, 1)
    self.side.set_ylim(0, 1)

    def side(y, content, **kw):
        self.side.text(0.06, y, content, va="top", **kw)

    side(0.975, "DECISION DETAIL", fontsize=10.5, weight="bold")
    if row is not None:
        side(
            0.935,
            f"Cut {event + 1:,} (Boundary {position})",
            fontsize=13,
            color=GREEN,
        )
        name = d.aliases[selected]["label"]
        name_lines = textwrap.wrap(name, 28, max_lines=3, placeholder="…")
        side(
            0.900,
            "\n".join(name_lines),
            fontsize=10.5,
            linespacing=1.2,
            color=ROLE_COLORS.get(d.aliases[selected]["role"], MUTED),
        )
        size_y = 0.894 - 0.028 * len(name_lines)
        side(
            size_y,
            byte_label(int(d.trace.alias_sizes[selected]))
            + f"  ·  {d.aliases[selected]['role']}",
            fontsize=10,
            color=MUTED,
        )
        side(
            size_y - 0.033,
            "Evict modified data" if int(row[8]) else "Release · spill copy available",
            fontsize=10,
            color=MUTED,
        )
        side(
            size_y - 0.066,
            f"Exposed transfer score  {int(row[7]) / 1e6:.3f} ms",
            fontsize=10,
        )
        self.side.plot([0.06, 0.94], [0.714, 0.714], color=MUTED, alpha=0.25, lw=0.7)
        comparisons = d.trace.comparisons(rid, event)
        side(
            0.695,
            "WHY IT BEATS RUNNER-UP",
            fontsize=9.7,
            weight="bold",
            color=MUTED,
        )
        side(
            0.662,
            explain_comparison(comparisons[0]) if comparisons else "Only eligible cut",
            fontsize=10.5,
            linespacing=1.2,
        )
        side(
            0.591,
            f"TOP BOUNDARIES BEFORE CUT {event + 1:,}",
            fontsize=9.4,
            weight="bold",
            color=MUTED,
        )
        for i, (pos, _used, _cap, over) in enumerate(
            row[52:64].reshape(3, 4)[: int(row[51])]
        ):
            pos = int(pos)
            task = short_task(d.tasks[pos]) if pos < len(d.tasks) else "end"
            side(
                0.558 - i * 0.058,
                f"{i + 1}.  b{pos}  ·  +{int(over) / GIB:.2f} GiB\n     before {task}",
                fontsize=10.5,
                linespacing=1.2,
                color=WHITE if i == 0 else MUTED,
            )
        boundary_offset = self.composition.flat_budget - int(capacity[int(row[0])])
        used = (int(row[1]) + boundary_offset) / GIB
        self.side.plot([0.06, 0.94], [0.388, 0.388], color=MUTED, alpha=0.25, lw=0.7)
        side(
            0.373,
            f"Used at b{int(row[0])}",
            fontsize=10.5,
            color=MUTED,
        )
        if count > event:
            after = (int(pressure[int(row[0])]) + boundary_offset) / GIB
            side(
                0.341,
                f"{used:.3f} → {after:.3f} GiB",
                fontsize=13,
                weight="bold",
                color=GREEN,
            )
        else:
            side(
                0.341,
                f"{used:.3f} GiB",
                fontsize=13,
                weight="bold",
                color=WHITE,
            )
    else:
        side(0.90, "No cut selected", fontsize=11)
    current_task = d.tasks[position] if position < len(d.tasks) else None
    current_memory = current_task["required_memory"] if current_task else None
    self.side.plot([0.06, 0.94], [0.305, 0.305], color=MUTED, alpha=0.25, lw=0.7)
    side(
        0.289,
        f"CURRENT TASK · b{position}",
        fontsize=9.7,
        weight="bold",
        color=MUTED,
    )
    side(
        0.257,
        short_task(current_task) if current_task else "End of program",
        fontsize=11,
    )
    if current_memory is not None:
        for y, key, label in (
            (0.213, "inputs", "Inputs"),
            (0.181, "outputs", "Outputs"),
            (0.149, "mutated", "Mutated"),
        ):
            entry = current_memory[key]
            side(y, f"{label} ({entry['object_count']:,})", fontsize=10.5)
            self.side.text(
                0.94,
                y,
                byte_label(entry["bytes"]),
                ha="right",
                va="top",
                fontsize=10.5,
            )
        side(0.117, "Workspace", fontsize=10.5)
        self.side.text(
            0.94,
            0.117,
            byte_label(current_memory["workspace_bytes"]),
            ha="right",
            va="top",
            fontsize=10.5,
        )
        entry = current_memory["total"]
        side(0.073, f"Total ({entry['object_count']:,})", fontsize=11, weight="bold")
        self.side.text(
            0.94,
            0.073,
            byte_label(entry["bytes"]),
            ha="right",
            va="top",
            fontsize=11,
            weight="bold",
        )
    return current_task, current_memory
