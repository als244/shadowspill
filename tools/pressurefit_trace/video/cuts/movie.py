"""Encode an animated sequence of exact replay states."""

import json
from dataclasses import asdict

import matplotlib.pyplot as plt

from ..encoding import video_stream
from .canvas import Renderer
from .pacing import explanation_events
from .playback import Playback, Timing
from .recording import Recording


def render_movie(
    data,
    output,
    *,
    title=None,
    reductions,
    width=1920,
    fps=30,
    context=3,
    focus_rows=7,
    opening_cuts=None,
    slow_seconds=8,
    accelerated_seconds=45,
    pacing="overview",
    explain_cuts=None,
    detailed=False,
    explain_seconds=15,
    detail_at=None,
    step_seconds=1,
    bridge_seconds=24,
    bridge_hold_seconds=0.6,
    accelerated_hold_seconds=0.1,
    seed_seconds=None,
    final_seconds=None,
    cut_zoom=True,
    zoom_seconds=1.2,
    zoom_hold_seconds=3,
    quality="high",
    pressure_scale="adaptive",
    pressure_view="total",
    focus_mode="cut",
):
    if opening_cuts is None:
        opening_cuts = 6 if detailed else 0
    if explain_cuts is None:
        explain_cuts = 3 if detailed else 0
    if seed_seconds is None:
        seed_seconds = 5 if detailed else 0
    if final_seconds is None:
        final_seconds = 4 if detailed else 0
    details = (
        sorted(set(cut - 1 for cut in detail_at))
        if detail_at is not None
        else explanation_events(
            data, reductions[0], explain_cuts, minimum_gap=opening_cuts
        )
    )
    if any(
        i < 0 or i >= len(data.trace.reductions[reductions[0]].cuts) for i in details
    ):
        raise ValueError(
            "--detail-at cut numbers must belong to the first rendered reduction"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(
        data,
        title=title,
        width=width,
        context=context,
        focus_rows=focus_rows,
        pressure_scale=pressure_scale,
        pressure_view=pressure_view,
        focus_mode=focus_mode,
    )
    height = int(width * 9 / 16)
    timing = Timing(
        opening_cuts=opening_cuts,
        slow_seconds=slow_seconds,
        explain_seconds=explain_seconds,
        step_seconds=step_seconds,
        bridge_seconds=bridge_seconds,
        bridge_hold_seconds=bridge_hold_seconds,
        accelerated_seconds=accelerated_seconds,
        accelerated_hold_seconds=accelerated_hold_seconds,
        seed_seconds=seed_seconds,
        final_seconds=final_seconds,
        cut_zoom=cut_zoom,
        zoom_seconds=zoom_seconds,
        zoom_hold_seconds=zoom_hold_seconds,
    )
    try:
        with video_stream(
            output, width=width, height=height, fps=fps, quality=quality
        ) as stream:
            recording = Recording(renderer, stream, fps)
            for chapter, rid in enumerate(reductions):
                playback = Playback(recording, rid, pacing=pacing, output=output)
                playback.chapter(
                    timing,
                    details=details if chapter == 0 else [],
                    opening=chapter == 0,
                )
    finally:
        plt.close(renderer.fig)
    report = {
        "title": title,
        "width": width,
        "height": height,
        "quality": quality,
        "axis": data.axis,
        "source": str(data.directory),
        "candidate_index": data.candidate_index,
        "candidate_file": data.candidate["file"],
        "collection_context": data.collection_context,
        "recompute_share": data.candidate.get("recompute_share"),
        "resolution_recompute_fraction": data.candidate.get(
            "resolution_recompute_fraction"
        ),
        "recompute_tasks": [
            {
                key: task[key]
                for key in (
                    "task_id",
                    "profiled_duration_ns",
                    "save_backward_task_id",
                    "save_backward_duration_ns",
                    "recompute_extra_ns",
                    "recompute_prefix_ns",
                )
            }
            for task in data.tasks
            if "recompute_prefix_ns" in task
        ],
        "cut_score": data.trace.cut_score,
        "cut_ranking": data.trace.cut_ranking,
        "ranking_criteria": list(data.trace.ranking_criteria),
        "answer_reduction": data.trace.answer_reduction,
        "rendered_reductions": list(reductions),
        "fps": fps,
        "frames": recording.frame_count,
        "seconds": recording.frame_count / fps,
        "pacing": pacing,
        "detailed_cuts": [i + 1 for i in details],
        "timing": asdict(timing),
        "pressure_scale": pressure_scale,
        "pressure_view": pressure_view,
        "focus_mode": focus_mode,
        "pressure_attributed": renderer.composition.attributed,
        "row_filter": data.row_filter_summary(),
        "shots": recording.shots,
    }
    output.with_suffix(".storyboard.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
