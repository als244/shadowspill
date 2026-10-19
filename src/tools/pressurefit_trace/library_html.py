"""Small offline HTML views for the saved-video catalog."""

# HTML template lines stay intact so their markup can be reviewed directly.
# ruff: noqa: E501

from __future__ import annotations

import html
from collections import Counter


def escape(value):
    return html.escape(str(value), quote=True)


def shell(title, content, root, directory, *, subtitle=""):
    from .library import href

    home = href(root / "index.html", directory)
    css = href(root / "_catalog/style.css", directory)
    js = href(root / "_catalog/index.js", directory)
    guide = root / "Data and rerendering.md"
    guide_link = (
        f'<a href="{href(guide, directory)}">Rerendering</a>' if guide.exists() else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><link rel="stylesheet" href="{css}"></head>
<body><header class="masthead"><a href="{home}">ShadowSpill <span>Video library</span></a>
<nav>{guide_link}<a href="{href(root / "README.md", directory)}">README</a>
<a href="{href(root / "results.csv", directory)}">CSV</a>
<a href="{href(root / "results.json", directory)}">JSON</a></nav></header>
<main><div class="eyebrow">ALGORITHM STUDIES · SAVED SIMULATIONS</div>
<h1>{escape(title)}</h1><p class="subtitle">{escape(subtitle)}</p>{content}</main>
<footer>Saved plans, raw traces, and original movie files stay linked to each configuration.
<br>Simulated and measured throughput are labeled separately; video playback follows teaching pace.</footer>
<script src="{js}"></script></body></html>"""


def video_links(row, root, directory):
    from .library import relative_url

    links = []
    for v in row["videos"]:
        label = {
            "cuts": "Cuts",
            "cuts-task": "Cuts (tasks)",
            "emission-placement": "Emission + placement",
            "execution": "Execution",
        }.get(v.get("kind"), v["label"])
        if v.get("url"):
            url = relative_url(v["url"], root, directory)
            links.append(f'<a class="video-link" href="{url}">{escape(label)} ↗</a>')
        else:
            links.append(
                f'<span class="pending">{escape(label)} · {escape(v["status"])}</span>'
            )
    return "".join(links)


def table_row(row, root, directory):
    from .library import number, relative_url

    url = relative_url(row["page"], root, directory)
    title = row["model"]
    rate = number(row.get("tokens_per_second"))
    step = number(row.get("step_seconds"), 3)
    search = " ".join(
        str(row.get(k, ""))
        for k in (
            "model",
            "budget_gib",
            "transfer",
            "geometry",
            "ordering",
            "resolution",
            "candidate_id",
            "edition",
            "category",
            "status",
        )
    )
    videos = video_links(row, root, directory)
    status = ""
    if row["status"] == "infeasible":
        status = '<span class="badge infeasible">Infeasible</span>'
        videos = f'<a href="{url}">View failure reasons →</a>'
    edition = row.get("edition") or row["category"]
    return f"""<tr data-model="{escape(row["model"])}" data-budget="{row["budget_gib"]:g}"
data-category="{escape(row["category"])}" data-status="{escape(row["status"])}"
data-search="{escape(search.casefold())}" data-rate="{row.get("tokens_per_second") or -1}">
<td><a class="row-title" href="{url}">{escape(title)}</a>
<div class="muted">{row["budget_gib"]:g} GiB · {escape(row["transfer"])}</div>
<div class="edition">{escape(edition)} {status}</div></td>
<td><b>{escape(row.get("geometry", "—"))}</b> seqs/mb
<div>{escape(row.get("ordering", "—"))}</div></td>
<td>{escape(row.get("resolution", "—"))}</td>
<td class="numeric">{step}<small>s / step</small></td>
<td class="numeric throughput">{rate}<small>tokens / s</small></td>
<td class="numeric throughput">{number(row.get("measured_tokens_per_second"))}<small>tokens / s</small></td>
<td class="video-links">{videos}<a class="details" href="{url}">Details &amp; source data →</a></td></tr>"""


def index_html(document, root, directory):
    from .library import href, number, relative_url

    rows = document["entries"]
    cards = []
    featured = [r for r in rows if r.get("featured")]
    for r in featured:
        url = relative_url(r["page"], root, directory)
        cards.append(f"""<a class="feature" href="{url}">
<span class="eyebrow">{escape(r["model"])} · WINNER</span>
<h2>{r["budget_gib"]:g} GiB</h2><div>{escape(r["transfer"])}</div>
<div class="big-number">{number(r.get("tokens_per_second"))}<span>simulated tokens/s</span></div>
<p>{number(r.get("step_seconds"), 3)} s / step · {escape(r.get("geometry"))} seqs/mb<br>
{escape(r.get("ordering"))} · recompute {escape(r.get("resolution"))}</p>
<strong>{r["ready_videos"]} / {r["expected_videos"]} videos ready <span>Explore →</span></strong></a>""")
    counts = document["counts"]
    matrix = Counter(
        v["status"] for r in rows if r["category"] == "Matrix" for v in r["videos"]
    )
    ready = sum(r["ready_videos"] for r in rows)
    infeasible = sum(r["status"] == "infeasible" for r in rows)
    videos_total = sum(len(r["videos"]) for r in rows)
    status = f"{ready} / {videos_total} videos ready · {len(rows)} catalog entries"
    if infeasible:
        status += f" · {infeasible} infeasible configurations"
    matrix_total = sum(matrix.values())
    progress = ""
    if matrix_total:
        progress = f"""<div class="progress-panel"><div><strong>Expanded matrix</strong>
<span>{matrix.get("complete", 0)} / {matrix_total} videos ready ·
{matrix.get("running", 0)} rendering · {matrix.get("queued", 0)} queued ·
{matrix.get("failed", 0)} failed</span></div>
<progress value="{matrix.get("complete", 0)}" max="{matrix_total}"></progress></div>"""
    options = lambda values: "".join(  # noqa: E731
        f'<option value="{escape(v)}">{escape(v)}</option>' for v in values
    )
    models = options(sorted({r["model"] for r in rows}))
    budgets = options(f"{v:g}" for v in sorted({r["budget_gib"] for r in rows}))
    notes = "".join(f"<li>{escape(n)}</li>" for n in document["notes"])
    directories = sorted({r["directory"].split("/")[0] for r in rows})
    folders = "".join(
        f'<a class="pill" href="{href(root / d / "index.html", directory)}">{escape(d)} →</a>'
        for d in directories
    )
    body = f"""<div class="statusline"><span>{status}</span>
<span>Updated {escape(document["updated_at"])} <button id="refresh">Refresh</button></span></div>
<div class="features">{"".join(cards)}</div>{progress}
<div class="folder-links">{folders}</div>
<section class="catalog"><div class="section-title"><h2>Configurations &amp; results</h2>
<a href="{href(root / "results.csv", directory)}">Download results CSV ↗</a></div>
<p class="muted">Throughput = tokens per step ÷ step time. Measured results appear when training steps were recorded.
Open a configuration for timing evidence, policy, video lengths, and saved inputs.</p>
<div class="filters"><label class="search">Search<input id="search" type="search" placeholder="Model, ordering, policy…"></label>
<label>Model<select id="model"><option value="">All models</option>{models}</select></label>
<label>Budget<select id="budget"><option value="">All GiB</option>{budgets}</select></label>
<label>Show<select id="scope"><option value="active">Current &amp; matrix</option>
<option value="Winner">Winner videos</option><option value="Matrix">Full matrix</option>
<option value="earlier">Archive &amp; earlier cuts</option><option value="all">Everything</option></select></label>
<label>Sort<select id="sort"><option value="library">Library order</option>
<option value="fastest">Throughput: highest first</option><option value="slowest">Throughput: lowest first</option></select></label></div>
<div id="match-count" class="muted"></div>
<div class="table-wrap"><table id="configurations"><thead><tr><th>Model &amp; configuration</th>
<th>Microbatch / order</th><th>Recompute</th><th class="numeric">Sim. step</th>
<th class="numeric">Sim. throughput</th><th class="numeric">Measured throughput</th><th>Videos &amp; details</th></tr></thead>
<tbody>{"".join(table_row(r, root, directory) for r in rows)}</tbody></table></div>
<p id="empty" hidden>No matching configurations. Try a different filter.</p></section>
<section class="notes"><h2>Reading these results</h2><ul>{notes}</ul>
<p>Current videos ready: {counts.get("complete", 0)}. “Ready” means the published MP4 exists;
per-video records identify decode validation. Visual review is separate from decode validation.</p>
<p>Data links open preserved local run directories. Keep those directories when moving the library,
or copy with symlinks dereferenced. This catalog uses no external services.</p></section>"""
    return shell(
        document["title"],
        body,
        root,
        directory,
        subtitle="Explore the algorithm, compare configurations, and return to the original data.",
    )


def detail_html(row, document, root, directory):
    from .library import duration, href, metric_rows, number, relative_url

    link = lambda url: relative_url(url, root, directory)  # noqa: E731
    ready = [v for v in row["videos"] if v.get("url")]
    buttons = []
    downloads = []
    for v in row["videos"]:
        length = duration(v.get("seconds"))
        resolution = f"{v.get('width', '—')} x {v.get('height', '—')}"
        if v.get("url"):
            buttons.append(
                f'<button class="choose-video" data-src="{link(v["url"])}" data-label="{escape(v["label"])}">'
                f"{escape(v['label'])}<small>{length} · {resolution}</small></button>"
            )
            downloads.append(f'<a href="{link(v["url"])}">{escape(v["label"])} ↗</a>')
        else:
            buttons.append(
                f"<button disabled>{escape(v['label'])}<small>{escape(v['status'])}</small></button>"
            )
    player = ""
    if ready:
        first = ready[0]
        player = f'<video id="player" controls preload="none" src="{link(first["url"])}"></video><p id="now-playing">{escape(first["label"])}</p>'
    elif row["status"] == "ready":
        player = '<div class="waiting">This plan is saved. Its videos will appear here as rendering finishes.</div>'
    facts = "".join(
        f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>"
        for k, v in metric_rows(row)
    )
    sources = "".join(
        f'<a class="source-link" href="{link(s["url"])}">{escape(s["label"])} ↗</a>'
        for s in row["sources"]
    )
    related = "".join(
        f'<a class="source-link" href="{link(s["url"])}">{escape(s["label"])} →</a>'
        for s in row.get("related", [])
    )
    if related:
        related = "<h3>Also in this folder</h3>" + related
    note = f'<p class="notice">{escape(row["note"])}</p>' if row.get("note") else ""
    failures = ""
    if row["status"] == "infeasible":
        failures = (
            '<section class="notes"><h2>Why this configuration is infeasible</h2><ul>'
        )
        failures += "".join(
            f"<li><b>{escape(f['candidate_id'])}</b><br>{escape(f.get('detail', f.get('kind')))}</li>"
            for f in row.get("failures", [])
        )
        failures += "</ul></section>"
    bandwidths = row.get("transfer_bandwidths") or {}
    measured = ""
    if measurement := row.get("measurement"):
        error = row.get("simulation_error")
        error_label = "—" if error is None else f"{error:+.2%}"
        measured = f"""<div class="metrics"><div><span>Measured throughput</span><strong>{number(row.get("measured_tokens_per_second"))}<small>tokens / s</small></strong></div>
<div><span>Measured step time</span><strong>{number(row.get("measured_step_seconds"), 3)}<small>seconds</small></strong></div>
<div><span>Simulation error</span><strong>{error_label}<small>simulated / measured - 1</small></strong></div>
<div><span>Useful training throughput</span><strong>{number(row.get("useful_tflops_per_gpu"), 1)}<small>TFLOP/s/GPU</small></strong></div></div>
<p class="muted">{escape(measurement["aggregation"])}</p>"""
    bandwidth = ""
    if bandwidths:
        bandwidth = '<h3>Transfer calibration / assumption</h3><table class="facts">'
        for direction in ("fetch", "evict"):
            for mode in ("solo", "concurrent"):
                rate = bandwidths.get(f"{direction}_{mode}_bytes_per_second")
                if rate is not None:
                    bandwidth += f"<tr><th>{direction.capitalize()} · {mode}</th><td>{rate / 1e9:g} GB/s</td></tr>"
            latency = bandwidths.get(f"{direction}_latency_ns")
            if latency is not None:
                bandwidth += f"<tr><th>{direction.capitalize()} latency</th><td>{latency / 1e3:g} µs</td></tr>"
        bandwidth += "</table>"
    body = f"""<div class="breadcrumbs"><a href="{href(root / "index.html", directory)}">All configurations</a>
<span> / {escape(row["category"])}</span><a href="README.md">Configuration README ↗</a></div>{note}
<div class="metrics"><div><span>Simulated throughput</span><strong>{number(row.get("tokens_per_second"))}<small>tokens / s</small></strong></div>
<div><span>Simulated step time</span><strong>{number(row.get("step_seconds"), 3)}<small>seconds</small></strong></div>
<div><span>Microbatch / ordering</span><strong>{escape(row.get("geometry", "—"))}<small>seqs · {escape(row.get("ordering", "—"))}</small></strong></div>
<div><span>Recompute resolution</span><strong>{escape(row.get("resolution", "—"))}<small>{escape(row.get("candidate_id") or "No feasible plan")}</small></strong></div></div>{measured}
<section class="screen"><div class="video-choices">{"".join(buttons)}</div>{player}
<div class="downloads">Open original MP4: {"".join(downloads)}</div></section>{failures}
<div class="detail-columns"><section><h2>Configuration &amp; results</h2><table class="facts">{facts}</table>
<p class="muted">Task waits sum start - ready over tasks; this excludes final transfer drain.
Peak execution memory includes task workspace. Memory and lane statistics come from the saved simulation.</p>{bandwidth}</section>
<section><h2>Source data &amp; reproduction</h2><p>These links open the exact saved inputs for this entry.</p>
<div class="source-links">{sources}</div>{related}<p class="muted">Source folders are symlinks to the canonical local run.
Preserve their targets if moving this library.</p><a href="result.json">Machine-readable configuration ↗</a>
<h3>Interpretation</h3><p>Throughput = {number(row.get("tokens_per_step"))} tokens ÷ the corresponding simulated or measured step time.
The movie's duration reflects teaching pace.</p><p>{escape(row["timing_basis"])}.</p>
<p>Updated {escape(document["updated_at"])}.</p></section></div>"""
    title = f"{row['model']} · {row['budget_gib']:g} GiB"
    subtitle = f"{row['transfer']} · {row.get('edition') or row['category']}"
    return shell(title, body, root, directory, subtitle=subtitle)
