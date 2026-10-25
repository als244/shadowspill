"""Run write for pool-occupancy diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..intervals import GIB
from ..page import describe_plan, write_pages
from ..plan import all_save, attribute, unconstrained
from ..stored import (
    StoredPlan,
    _budget_label,
    _request,
    _shape,
    _stored_selection,
    _tokens_per_step,
    program_path_for,
    stored_plans,
)
from .index import _write_plan_index, _write_run_indexes
from .records import _Entry, _share
from .traces import _traced_steps


def _floor_line(
    selection: Mapping[str, Any],
    model: str,
    record: StoredPlan,
    sequence_length: int | None,
    sequences_per_step: int | None,
) -> str:
    """The line under a plan's own floor page: the plan's alternatives with
    nothing to plan around, so the capacities are left off."""

    return describe_plan(
        {"selections": selection.get("selections", ())},
        model=model,
        geometry=record.geometry,
        sequence_length=sequence_length,
        sequences_per_step=sequences_per_step,
        unconstrained=True,
    )


def recompute_share_of(
    program: Mapping[str, Any], selections: Iterable[Mapping[str, str]]
) -> str:
    """The share of the program's flexible groups ``selections`` recomputes,
    by the planner's own rule, as the store names a kept resolution by it;
    empty when the program is not one the planner's IR reads."""

    from shadowspill.ir import ShadowSpillProgram, TaskAlternativeChoice
    from shadowspill.planner.search.toolkit.resolution import recompute_share

    try:
        parsed = ShadowSpillProgram.from_dict(program)
        choices = tuple(
            TaskAlternativeChoice.from_value(item, "selections") for item in selections
        )
    except Exception:
        return ""
    return str(recompute_share(parsed, choices))


def _share_folder(share: str) -> str:
    """``recompute_<share>``, the store's own naming of a kept resolution."""

    numerator, _, denominator = share.partition("/")
    return f"recompute_{numerator}" + (f"of{denominator}" if denominator else "")


def write_run_timelines(
    run_root: Path,
    out: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Pages for every plan a run made, budget by budget.

    ``timelines/<budget>/<geometry>/recompute_<share>/`` holds the plan the
    search chose for that geometry at that budget, named by the share of
    the flexible groups it recomputes: ``simulated.html``,
    ``unconstrained.html``, its own floor, and ``traced.html`` on the
    device's clock when this budget ran with this geometry, found by the
    makespan the traced step records; that traced page is copied to
    ``timelines/<budget>/traced.html``, since one geometry runs per budget.
    When the search kept its resolutions, the others sit beside the choice
    under their own shares, each with ``simulated.html`` and its own
    ``unconstrained.html``. ``timelines/all_save/<geometry>/all_save.html``
    is each geometry at its cheapest alternatives, which no budget bounds.
    Each page carries the summary, the pools and the lanes on one zoom.

    The root ``index.html`` is the table of contents for everything, and
    every level below has its own: each budget, each geometry within it,
    each plan's folder, and ``all_save/``. ``summary.csv`` at the root holds
    one row per page, the summary the page's cards show with what
    identifies the plan (`SUMMARY_COLUMNS`). ``progress`` hears one line at
    the start and one per budget finished, since a tour's pages take
    minutes to write. Returns the root index.
    """

    out = out or (run_root / "timelines")
    plans = stored_plans(run_root)
    if progress is not None:
        kept_count = sum(1 for record in plans if record.resolution)
        progress(
            f"timelines: writing pages for {len(plans) - kept_count} plans"
            + (f" and {kept_count} kept resolutions" if kept_count else "")
            + f" under {out}"
        )
    tokens = _tokens_per_step(run_root)
    request = _request(run_root)
    model = str(request.get("model") or "")
    sequence_length, sequences_per_step = _shape(request)
    traced_for, missing = _traced_steps(run_root, plans)
    entries: list[_Entry] = []
    taken: set[Path] = set()
    programs: dict[str, tuple[str, Mapping[str, Any]]] = {}
    kept_by_key: dict[str, list[StoredPlan]] = defaultdict(list)
    for record in plans:
        if record.resolution:
            kept_by_key[record.key].append(record)
    ordered = sorted(
        (record for record in plans if not record.resolution),
        key=lambda item: (item.budget_bytes or 0, item.geometry, item.makespan_ns),
    )

    def naming(
        record: StoredPlan, budget: str, kept: StoredPlan | None = None
    ) -> dict[str, Any]:
        return dict(
            model=model,
            geometry=record.geometry,
            budget=budget,
            sequence_length=sequence_length,
            sequences_per_step=sequences_per_step,
            resolution="" if kept is None else kept.resolution,
            selected=False if kept is None else kept.selected,
        )

    finished: str | None = None
    for record in ordered:
        selection = _stored_selection(record.path)
        program = json.loads(program_path_for(record.path, selection).read_text())
        geometry = record.geometry or f"plan_{record.path.parent.name[:8]}"
        budget = (
            _budget_label(record.budget_bytes)
            if record.budget_bytes
            else f"{record.pool_capacity_bytes / GIB:.2f}gib_slab"
        )
        if progress is not None and finished not in (None, budget):
            progress(f"timelines: {finished} written")
        finished = budget
        directory = out / budget / geometry
        if directory in taken:
            directory = out / budget / f"{geometry}_{record.path.parent.name[:8]}"
        taken.add(directory)
        programs.setdefault(record.program_digest, (geometry, program))
        kept_records = sorted(
            kept_by_key.get(record.key, ()), key=lambda item: _share(item.resolution)
        )
        # the choice is one of the kept resolutions when there are any, and
        # its folder is theirs; otherwise the planner's rule names its share
        chosen_record = next((kept for kept in kept_records if kept.selected), None)
        if chosen_record is not None:
            chosen_share, chosen_folder = (
                chosen_record.resolution,
                chosen_record.path.parent.name,
            )
        else:
            chosen_share = recompute_share_of(program, selection.get("selections", ()))
            chosen_folder = _share_folder(chosen_share) if chosen_share else "chosen"
        others = [kept for kept in kept_records if not kept.selected]
        spill_capacity = (selection.get("simulation") or {}).get("spill_capacity_bytes")
        spill_bytes = None if spill_capacity is None else int(spill_capacity)
        views = [attribute(selection, program)]
        diagnostics = traced_for.get(record.path)
        if diagnostics is not None:
            views.append(attribute(selection, program, diagnostics=diagnostics))
        views.append(unconstrained(program, selection.get("selections", ())))
        pages = write_pages(
            views,
            directory / chosen_folder,
            title=f"{geometry} at {budget}, recompute {chosen_share or '?'}",
            tokens_per_step=tokens,
            plan=describe_plan(
                selection,
                **{
                    **naming(record, budget),
                    "resolution": chosen_share,
                    "selected": True,
                },
            ),
            plan_by_view={
                "unconstrained": _floor_line(
                    selection, model, record, sequence_length, sequences_per_step
                )
            },
            related=[
                (
                    f"resolution {kept.resolution}",
                    f"../{kept.path.parent.name}/index.html",
                )
                for kept in others
            ]
            + [
                (
                    f"{geometry} at its cheapest, all save",
                    f"../../../all_save/{geometry}/index.html",
                ),
                (f"all of {budget}", "../../index.html"),
                ("everything", "../../../index.html"),
            ],
        )
        if diagnostics is not None:
            # the budget's traced step, a copy at the budget's own level: the
            # page is self-contained, so the bytes are the same anywhere
            traced_page = next(page for page in pages if page.name == "traced.html")
            (out / budget / "traced.html").write_bytes(traced_page.read_bytes())
        entries.append(
            _Entry(
                "chosen",
                geometry,
                budget,
                chosen_share,
                True,
                directory / chosen_folder,
                views,
                record.pool_capacity_bytes,
                spill_bytes,
            )
        )
        for kept in others:
            folder = directory / kept.path.parent.name
            kept_selection = _stored_selection(kept.path)
            kept_views = [
                attribute(kept_selection, program),
                unconstrained(program, kept_selection.get("selections", ())),
            ]
            write_pages(
                kept_views,
                folder,
                title=f"{geometry} at {budget}, recompute {kept.resolution}",
                tokens_per_step=tokens,
                plan=describe_plan(kept_selection, **naming(record, budget, kept)),
                # the resolution's own floor: nothing to plan around, the
                # alternatives fixed as this resolution fixes them
                plan_by_view={
                    "unconstrained": describe_plan(
                        {}, unconstrained=True, **naming(record, "", kept)
                    )
                },
                related=[
                    (
                        f"the search's choice, recompute {chosen_share or '?'}",
                        f"../{chosen_folder}/index.html",
                    ),
                    (f"all of {budget}", "../../index.html"),
                    ("everything", "../../../index.html"),
                ],
            )
            entries.append(
                _Entry(
                    "resolution",
                    geometry,
                    budget,
                    kept.resolution,
                    kept.selected,
                    folder,
                    kept_views,
                    kept.pool_capacity_bytes,
                    spill_bytes,
                )
            )
        _write_plan_index(out, directory, entries)
    if progress is not None and finished is not None:
        progress(f"timelines: {finished} written")

    if progress is not None:
        progress("timelines: the geometries' all-save floors")
    for geometry, program in programs.values():
        result = all_save(program)
        directory = out / "all_save" / geometry
        write_pages(
            [result],
            directory,
            title=f"{geometry} all save",
            tokens_per_step=tokens,
            plan=describe_plan(
                {},
                model=model,
                geometry=geometry,
                sequence_length=sequence_length,
                sequences_per_step=sequences_per_step,
                unconstrained=True,
            ),
            related=[
                ("every geometry's floor", "../index.html"),
                ("everything", "../../index.html"),
            ],
        )
        entries.append(_Entry("all_save", geometry, "", "", False, directory, [result]))

    _write_run_indexes(out, run_root, entries, missing, tokens)
    return out / "index.html"
