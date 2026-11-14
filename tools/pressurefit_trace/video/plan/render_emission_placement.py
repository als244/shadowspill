"""Release/evict/fetch explanations and physical placement construction."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

from ...data import GIB, byte_label, short_task
from ..style import BG, GREEN, MUTED, RED, WHITE
from .render_common import ROLE_COLOR, PlanCanvas, boxes, truncate


class EmissionPlacementRenderer(PlanCanvas):
    """Compose residency/action scenes and the address-versus-time layout."""

    def draw_scene(self, state):
        if state["kind"].startswith("placement"):
            self.placement(state)
        elif state["kind"].startswith("emission"):
            self.emission(state)
        else:
            raise ValueError("expected an emission or placement scene")

    def emission(self, state):
        self.minimap(state)
        self.task_strip(state["window_ns"])
        lo, hi = state["window_ns"]
        ax = self.axis(
            (0.07, 0.35, 0.63, 0.33),
            xlim=(lo / 1e9, hi / 1e9),
            ylim=(-0.5, 3.6),
            ticks=False,
        )
        ax.set_xticks(np.linspace(lo / 1e9, hi / 1e9, 5))
        ax.set_xlabel("Simulated time (s)", fontsize=11)
        side = self.card()
        if "action" not in state:
            aliases = sorted(
                (
                    a
                    for a in self.data.facts.alias_size
                    if self.data.facts.alias_size[a] >= 2**20 and self.data.uses[a]
                ),
                key=lambda a: (self.data.uses[a][0], a),
            )
            offsets, total = {}, 0
            for alias in aliases:
                offsets[alias] = total
                total += self.data.facts.alias_size[alias]
            rectangles, colors = [], []
            for p in self.data.leases:
                alias = p["alias_group_id"]
                if alias in offsets:
                    rectangles.append(
                        (
                            p["predicted_start_ns"] / 1e9,
                            offsets[alias] / max(1, total),
                            (p["predicted_end_ns"] - p["predicted_start_ns"]) / 1e9,
                            self.data.facts.alias_size[alias] / max(1, total),
                        )
                    )
                    colors.append(ROLE_COLOR.get(p["role"], MUTED))
            boxes(ax, rectangles, colors)
            ax.set_ylim(1, 0)
            ax.set_ylabel("Object residency", fontsize=11)
            counts = self.data.summary()["actions"]
            y = self.card_text(side, 0.95, "Residency → actions", size=18)
            for kind in ("release", "evict", "fetch"):
                y = self.card_text(
                    side, y - 0.025, f"{counts.get(kind, 0):,} {kind} actions", size=15
                )
            self.card_text(
                side,
                y - 0.05,
                "A departure changes residency. A fetch reserves "
                "a destination before the next use.",
                size=13,
            )
        else:
            a = self.data.action_detail(state["action"])
            alias = a["alias_group_id"]
            color = ROLE_COLOR.get(a["role"], MUTED)
            ax.set_yticks(
                [2.5, 1.4, 0.3], ["Execution\nallocation", "Required\nuse", "Transfers"]
            )
            for p in self.data.leases:
                if (
                    p["alias_group_id"] == alias
                    and p["predicted_end_ns"] >= lo
                    and p["predicted_start_ns"] <= hi
                ):
                    ax.add_patch(
                        Rectangle(
                            (p["predicted_start_ns"] / 1e9, 2.2),
                            (p["predicted_end_ns"] - p["predicted_start_ns"]) / 1e9,
                            0.6,
                            facecolor=color,
                            alpha=0.8,
                        )
                    )
            for i in self.data.uses[alias]:
                t = self.data.tasks[i]
                if t["end_ns"] < lo or t["start_ns"] > hi:
                    continue
                ax.add_patch(
                    Rectangle(
                        (t["start_ns"] / 1e9, 1.1),
                        (t["end_ns"] - t["start_ns"]) / 1e9,
                        0.6,
                        facecolor=color,
                        hatch="////",
                        edgecolor=WHITE,
                        linewidth=0.5,
                    )
                )
            for other in self.data.alias_actions[alias]:
                x = other["trigger_ns"] / 1e9
                if lo <= other["trigger_ns"] <= hi:
                    ax.plot(
                        x,
                        2.95,
                        marker={
                            "fetch": "v",
                            "release": "^",
                            "evict": "^",
                            "write_back": "^",
                        }[other["kind"]],
                        color=GREEN if other["kind"] == "release" else WHITE,
                        markersize=8,
                    )
                    ax.text(
                        x, 3.17, other["kind"], ha="center", va="bottom", fontsize=10
                    )
                t = other["transfer"]
                if t is not None and t["end_ns"] >= lo and other["trigger_ns"] <= hi:
                    ax.plot(
                        [x, t["start_ns"] / 1e9],
                        [0.3, 0.3],
                        color=color,
                        linestyle="--",
                        linewidth=2,
                    )
                    ax.add_patch(
                        Rectangle(
                            (t["start_ns"] / 1e9, 0.05),
                            (t["end_ns"] - t["start_ns"]) / 1e9,
                            0.5,
                            facecolor=color,
                            edgecolor=WHITE,
                            linewidth=0.4,
                        )
                    )
            ax.axvline(a["trigger_ns"] / 1e9, color=GREEN, linewidth=1, linestyle=":")
            ax.axvline(state["cursor_ns"] / 1e9, color=WHITE, linewidth=1.2)
            self.action_card(side, a)
        self.pressure(state)

    def action_card(self, side, a):
        color = ROLE_COLOR.get(a["role"], MUTED)
        y = self.card_text(
            side,
            0.95,
            f"{a['kind'].replace('_', ' ').title()} · Boundary {a['boundary']}",
            size=17,
        )
        y = self.card_text(side, y - 0.01, a["label"], size=15, color=color, wrap=30)
        size_label = byte_label(a["bytes"])
        if a["transfer"] is not None and a["transfer_bytes"] != a["bytes"]:
            size_label = (
                f"GPU {size_label} · CPU transfer {byte_label(a['transfer_bytes'])}"
            )
        y = self.card_text(side, y, size_label, size=13, color=MUTED)
        if a["kind"] == "release":
            reason = (
                "Current spill copy retained."
                if a["next_task"] is not None
                else "No later task needs this value."
            )
        elif a["kind"] == "evict":
            reason = (
                "Write back this value; keep the source allocated "
                "until the transfer completes."
            )
        else:
            reason = (
                "Reserve destination → wait for the lane → fetch before the consumer."
            )
        y = self.card_text(side, y - 0.025, reason, size=13)
        for name, index in (("Prior", a["prior_task"]), ("Next", a["next_task"])):
            if index is not None:
                t = self.data.tasks[index]
                y = self.card_text(
                    side,
                    y - 0.015,
                    f"{name}: {short_task(t)}\n"
                    f"Boundary {index if name == 'Next' else index + 1}",
                    size=12,
                )
        if a["legal_fetch_boundaries"] is not None:
            first, last = a["legal_fetch_boundaries"]
            y = self.card_text(
                side,
                y - 0.025,
                f"Legal fetch window: B{first} to B{last}",
                size=13,
                color=GREEN,
            )
            self.card_text(
                side,
                y,
                f"Final trigger: B{a['boundary']}\n{self.data.policy.split('/')[-1]}",
                size=12,
            )

    def placement_rectangles(
        self, ax, window, placed, *, selected=None, labels=True, alpha=1
    ):
        r = self.replay
        lo, hi = window
        mask = (r.ranks < placed) & (r.ends > lo) & (r.starts < hi)
        indices = np.flatnonzero(mask)
        rectangles = [
            (
                r.starts[i] / 1e9,
                r.offsets[i] / GIB,
                (r.ends[i] - r.starts[i]) / 1e9,
                r.sizes[i] / GIB,
            )
            for i in indices
        ]
        boxes(
            ax,
            rectangles,
            [self._lease_colors[i] for i in indices],
            alpha=alpha,
            edgecolor=BG if labels else "none",
            linewidth=0.25 if labels else 0,
        )
        if labels:
            height = ax.get_ylim()[1] - ax.get_ylim()[0]
            for i in indices:
                p = self.data.leases[i]
                visible = (min(r.ends[i], hi) - max(r.starts[i], lo)) / max(1, hi - lo)
                if visible > 0.13 and r.sizes[i] / GIB / height > 0.065:
                    x = (max(r.starts[i], lo) + min(r.ends[i], hi)) / 2e9
                    y = (r.offsets[i] + r.sizes[i] / 2) / GIB
                    label = truncate(p["label"], int(visible * 65))
                    ax.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=BG,
                        clip_on=True,
                    )
                    if placed <= 20:
                        ax.text(
                            max(r.starts[i], lo) / 1e9 + (hi - lo) / 1e9 * 0.008,
                            (r.offsets[i] + r.sizes[i] * 0.77) / GIB,
                            str(r.ranks[i] + 1),
                            ha="center",
                            va="center",
                            fontsize=9,
                            color=BG,
                            bbox={
                                "boxstyle": "circle,pad=.2",
                                "facecolor": WHITE,
                                "edgecolor": "none",
                            },
                            clip_on=True,
                        )
        if selected is not None:
            i = selected
            ax.add_patch(
                Rectangle(
                    (r.starts[i] / 1e9, r.offsets[i] / GIB),
                    (r.ends[i] - r.starts[i]) / 1e9,
                    r.sizes[i] / GIB,
                    facecolor="none",
                    edgecolor=WHITE,
                    linewidth=2,
                    zorder=6,
                )
            )

    def placement(self, state):
        self.minimap(state, placement=True)
        self.task_strip(state["window_ns"])
        lo, hi = state["window_ns"]
        index = state.get("lease")
        placed = state.get("placed", 0)
        ids = self.replay.order[:placed]
        span = int((self.replay.offsets[ids] + self.replay.sizes[ids]).max(initial=0))
        extent = (
            max(
                span,
                self.data.leases[index]["offset"] + self.data.leases[index]["bytes"],
            )
            if index is not None
            else span
        )
        zoomed = index is not None and extent < self.data.pool_bytes * 0.7
        ax = self.axis(
            (0.07, 0.345, 0.63, 0.34),
            xlim=(lo / 1e9, hi / 1e9),
            ylim=(
                0,
                max(1, extent) / GIB * 1.25
                if zoomed
                else self.data.pool_bytes / GIB * 1.08,
            ),
            ylabel="Pool address (GiB)",
        )
        if state["kind"] in ("placement-build", "placement-final"):
            ax.imshow(
                self.layout_raster.image(state.get("placed", 0)),
                extent=(0, self.data.end_ns / 1e9, 0, self.data.pool_bytes / GIB),
                origin="upper",
                aspect="auto",
                interpolation="antialiased",
            )
        else:
            self.placement_rectangles(
                ax, [lo, hi], state.get("placed", 0), selected=index
            )
        ax.axhline(
            self.data.pool_bytes / GIB, color=RED, linestyle=(0, (4, 4)), linewidth=1
        )
        pool_label = ax.text(
            0.995,
            0.975,
            f"Pool {byte_label(self.data.pool_bytes)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=RED,
        )
        pool_label.set_visible(not zoomed)
        if placed:
            ax.axhline(span / GIB, color=WHITE, linewidth=0.7, linestyle=":")
        if state["kind"] == "placement-sweep":
            cursor = state["cursor_ns"]
            ax.axvline(cursor / 1e9, color=WHITE, linewidth=1.3)
        side = self.card()
        if index is not None:
            if index not in self.detail_cache:
                self.detail_cache[index] = self.replay.detail(index)
            detail = self.detail_cache[index]
            p = self.data.leases[index]
            y = self.card_text(
                side, 0.96, f"Placement {detail['placement_number']:,}", size=18
            )
            y = self.card_text(
                side,
                y - 0.01,
                p["label"],
                size=15,
                color=ROLE_COLOR.get(p["role"], MUTED),
                wrap=30,
            )
            y = self.card_text(side, y, byte_label(p["bytes"]), size=13, color=MUTED)
            y = self.card_text(
                side,
                y - 0.025,
                "Largest size first\nThen most lifetime boundaries\n"
                "Then start and input order",
                size=12,
            )
            y = self.card_text(
                side,
                y - 0.01,
                f"{detail['lifetime_boundary_span']:,} lifetime boundaries\n"
                f"Alignment {byte_label(p['alignment'])}",
                size=12,
            )
            y = self.card_text(
                side,
                y - 0.02,
                f"Lowest fitting address\n{byte_label(p['offset'])}",
                size=15,
                color=GREEN,
            )
            if detail["attempts"]:
                count = len(detail["attempts"])
                y = self.card_text(
                    side,
                    y - 0.015,
                    f"{count} lower range{'s' if count != 1 else ''} blocked.",
                    size=12,
                )
            if detail["reused_predecessor_indices"]:
                self.card_text(
                    side,
                    y - 0.015,
                    "Reuse: earlier allocations at this address have ended.",
                    size=12,
                )
            # Project the overlapping lifetimes onto address space. The new
            # allocation must fit a gap in this occupied-address union.
            ax.set_xticklabels([])
            self.text(
                0.07,
                0.323,
                "Address ranges occupied by overlapping lifetimes",
                size=11,
                color=MUTED,
            )
            address = self.axis(
                (0.07, 0.282, 0.63, 0.027),
                xlim=(0, max(extent, 1) / GIB * 1.12),
                ylim=(0, 1),
            )
            address.set_yticks([])
            boxes(
                address,
                [
                    (a / GIB, 0.1, (b - a) / GIB, 0.8)
                    for a, b in detail["occupied_union"]
                ],
                MUTED,
            )
            address.add_patch(
                Rectangle(
                    (p["offset"] / GIB, 0.1),
                    p["bytes"] / GIB,
                    0.8,
                    facecolor=ROLE_COLOR.get(p["role"], MUTED),
                    edgecolor=WHITE,
                    linewidth=1.2,
                    hatch="///" if not state.get("reveal") else None,
                )
            )
            address.text(
                0.99,
                1.2,
                "Address (GiB)",
                transform=address.transAxes,
                ha="right",
                fontsize=10,
                color=MUTED,
            )
            if not state.get("reveal"):
                # Ghost is a placement preview, never a change to saved offsets.
                ax.add_patch(
                    Rectangle(
                        (p["predicted_start_ns"] / 1e9, p["offset"] / GIB),
                        (p["predicted_end_ns"] - p["predicted_start_ns"]) / 1e9,
                        p["bytes"] / GIB,
                        facecolor=ROLE_COLOR.get(p["role"], MUTED),
                        alpha=0.22,
                        edgecolor=WHITE,
                        hatch="///",
                        linewidth=1.1,
                        zorder=5,
                    )
                )
        else:
            y = self.card_text(
                side,
                0.95,
                f"{placed:,} / {len(self.data.leases):,}\nallocations placed",
                size=20,
            )
            y = self.card_text(
                side, y - 0.03, f"Layout span\n{byte_label(span)}", size=16
            )
            y = self.card_text(
                side,
                y - 0.03,
                f"Peak live allocations\n{byte_label(self.data.execution.peak)}",
                size=15,
                color=GREEN,
            )
            y = self.card_text(
                side,
                y - 0.03,
                "Width = allocation lifetime\nHeight = allocated bytes\n"
                "Position = fixed address",
                size=13,
            )
            if state["kind"] == "placement-intro":
                self.card_text(
                    side,
                    y - 0.02,
                    r"$\min\ o\quad\mathrm{subject\ to\ no\ overlap}$",
                    size=15,
                )
        self.pressure(
            state,
            rect=(0.07, 0.11, 0.63, 0.12)
            if index is not None
            else (0.07, 0.105, 0.63, 0.17),
        )
