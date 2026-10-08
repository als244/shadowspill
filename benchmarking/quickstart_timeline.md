# Quickstart plots for slides

Two exporters read saved quickstart results without running training:

- [Timeline exporter](#timeline-exporter): memory, compute, Fetch/Evict lanes and
  summary statistics, as in the simulated/measured timeline slides.
- [Tradeoff exporter](#tradeoff-exporter): full-range and detailed throughput,
  unconstrained/recomputation references, and winning-plan time shares, as in
  the memory-budget comparison slide.

Both export PNG/SVG/PDF plus provenance metadata, and optionally a PowerPoint
slide with editable titles and captions. Run them from the repository root.

## Timeline exporter

`python -m benchmarking.quickstart_timeline` reads an existing quickstart report and creates the
same aligned memory / GPU compute / host-to-device / device-to-host plot used
in the presentation. It does not run training or change quickstart plotting
code. **Fetch** means host → GPU and **Evict** means GPU → host. Empty transfer
lanes are omitted automatically.

## Setup

Use an environment with `matplotlib`, `numpy`, and (for PowerPoint)
`python-pptx`. Run from the ShadowSpill repository root. For a separate plotting
environment: `python -m pip install matplotlib numpy python-pptx`.

## Select a quickstart result

Pass the directory for **one sequence length and step size**. The script reads
`timelines/summary.csv`, chooses the fastest simulated plan at the requested
budget, and loads that plan's simulated or traced HTML. The summary's
`selected=True` is per geometry, so the script also compares geometries.

```bash
python -m benchmarking.quickstart_timeline /path/to/report/seq1024/seqsperstep64 \
  --budget-gib 8 --view traced --outdir figures/measured-8gib --pptx \
  --title 'The same plan runs within 8 GiB.' \
  --subtitle 'Llama 8B · 64K tokens per step · eight 8K microbatches'

python -m benchmarking.quickstart_timeline /path/to/report/seq1024/seqsperstep64 \
  --budget-gib 8 --view simulated --outdir figures/simulated-8gib --pptx
```

`--geometry 8x8_1x8rp` and `--resolution 3/4` select a particular candidate.
`--list` shows all available choices. An ambiguous report root or a missing
traced timeline produces an explicit error; the script does not silently
switch to a different plan.

An exact timeline HTML (or its embedded JSON saved as a file) also works:

```bash
python -m benchmarking.quickstart_timeline /path/to/timelines/8gib/8x8_1x8rp/recompute_3of4/traced.html \
  --outdir figures/my-plan --pptx
```

The execution budget is inferred from the HTML's plan description when
possible. `--budget-gib` can set it explicitly. `--time-max 18.5` makes paired
plots use an identical time axis; it must include every event. `--help` lists
all options, including title, subtitle, takeaway caption, DPI and overview
resolution.

## CLI options

| Option | Meaning |
| --- | --- |
| `SOURCE` | One report directory, its `timelines/` directory, or an exact HTML/JSON timeline |
| `--view traced\|simulated\|all-save\|unconstrained` | Default: `traced` for directories; the file's own view for files |
| `--budget-gib N` | Budget to select; also labels the memory-budget line |
| `--geometry NAME`, `--resolution FRACTION` | Optional exact candidate filters |
| `--list` | List a directory's available timelines and exit |
| `--outdir PATH` | Output directory; default `timeline-slide`; rerunning replaces its generated files |
| `--time-max SECONDS` | Align multiple plots to a common full-step time range |
| `--transfers auto\|show\|hide` | Default `auto`; refuses to hide transfers that actually exist |
| `--columns N`, `--dpi N` | Overview bins (1,800) and PNG resolution (300) |
| `--pptx` | Also create an importable PowerPoint slide |
| `--title TEXT`, `--subtitle TEXT` | Editable slide labels |
| `--summary`, `--no-summary` | HTML summary statistics under simulated/traced slides; enabled by default |
| `--caption TEXT` | Editable takeaway when summary cards are disabled or inapplicable |
| `--persistent-gradients-gib N` | Explicit unconstrained illustration: replace the gradient series with a constant bank |

## Outputs

| File | Purpose |
| --- | --- |
| `timeline.png` | 300 DPI, slide-ready image; insert without stretching |
| `timeline.svg` | Vector figure with text retained |
| `timeline.pdf` | Standalone vector figure |
| `metadata.json` | Source path and SHA-256, timings, peaks, event counts, display choices and caveats |
| `timeline-slide.pptx` | With `--pptx`: one slide with editable titles, subtitle, object legend and summary statistics (or takeaway); plot is an image |

The image is 12 × 3.65 inches. The PowerPoint uses a 16:9 slide. Import its slide
into Google Slides, or insert the PNG into an existing slide. This script does
not require Drive credentials or automatically overwrite an online deck.

Simulated/traced slides show all twelve summary statistics from the HTML:
step time, tokens/s, peak execution and spill memory, idle and recompute
percentages, fetch/evict lane utilization, fetched/evicted GiB, and transfer
rates. Rate cells show **result / assumed** in GB/s. These are read directly
from the selected page, including when measured and simulated statistics differ.

## Interpretation

- Memory is the report's tensor leases plus task workspace envelopes, not
  every physical CUDA allocation. Exact event-accounted peaks may differ
  slightly from the rounded, sampled stack.
- Workspace and control temporaries are included in **Activations** to simplify
  the presentation legend. No bytes are removed.
- The all-save view is an unconstrained, profiled-cost baseline, not a measured
  resident execution. Its gradient lifetimes are those of the source program.
- The `--persistent-gradients-gib` option replaces the gradient series with a
  constant bank for an explicitly adjusted illustration. It changes the memory
  stack, not task timings; it is disallowed for simulated/traced plans.
- Recompute segments use the source report's profile-cost split inside
  backward tasks. They are not separately measured kernel boundaries.
- Transfer colors identify the tensor class. Overview bins use the dominant
  duration per category; very short/overlapping transfers are condensed as in
  the HTML overview. No total-memory outline or smoothing is applied.
- Only gaps **between** compute tasks receive idle shading. Opening restores
  and terminal writeback are visible in transfer lanes without shading the
  surrounding initial/final GPU gaps.
- A single traced step and a report's multi-step median are different
  measurements; their timings and throughputs need not be identical.

## Tradeoff exporter

`python -m benchmarking.quickstart_tradeoff` creates the comparison slide from
`run_budgets.csv` and `points.csv`. It accepts a single quickstart run root,
its `figures/` directory, or its `figures/raw_data/` directory. The same Python
dependencies listed above apply; a GPU and Google credentials are unnecessary.

```bash
python -m benchmarking.quickstart_tradeoff \
  /path/to/report/seq1024/seqsperstep64 \
  --outdir figures/memory-tradeoff --pptx \
  --title 'Memory budget versus training throughput.' \
  --subtitle 'Llama 8B · 64K tokens per step · RTX 5090'
```

The exporter automatically reads the throughput unit and step size from
`throughput.json`. Archived text reports with `*_tokens_per_second` columns
also work; their step size is inferred and checked across budgets. Measurement
counts come from `steps.csv`, when present. Neither tokens per step nor the
number of iterations is hardcoded.

It finds the fastest all-save timeline under the run's `timelines/all_save/`
directory and uses its compute floor and peak memory for the unconstrained
reference. To choose that reference explicitly, pass `--unconstrained` an
all-save/unconstrained HTML or JSON file. If no such timeline exists, the plot
omits the unconstrained reference; it does not substitute an unrelated value.

### Tradeoff CLI options

| Option | Meaning |
| --- | --- |
| `SOURCE` | A quickstart run root, `figures/`, or `figures/raw_data/` containing `run_budgets.csv` and `points.csv` |
| `--outdir PATH` | Generated files; default `tradeoff-slide`; rerunning replaces generated files there |
| `--budget-gib 8,16,29` | Optional subset of measured budgets; permits rounded labels such as 29 for 29.0137 GiB |
| `--unconstrained PATH` | Explicit all-save/unconstrained timeline HTML or JSON instead of automatic discovery |
| `--unconstrained-peak-gib N` | Explicit override of the reference's memory label for an adjusted illustration; original peak and override are recorded in metadata |
| `--full-ymax N` | Upper throughput limit of the zero-based plot; auto extends slightly above all series/references |
| `--zoom-ylim MIN MAX` | Limits of the detailed throughput plot; auto fits measured, simulated and chosen-recomputation series |
| `--share-axis auto\|full` | Auto breaks a large unused percentage range when useful; full shows the complete percentage scale |
| `--unit-label TEXT` | Display label such as tokens, examples or updates; default comes from report metadata/columns |
| `--dpi N` | PNG resolution; default 300 |
| `--pptx` | Also write an importable one-slide PowerPoint |
| `--title TEXT` | Editable slide title; default “Memory budget versus training throughput.” |
| `--subtitle TEXT` | Editable subtitle; default gives step size and measured budget count |
| `--caption TEXT` | Editable takeaway; default compares the smallest selected budget with the best measured throughput |

Axis limits that would clip a curve or reference produce an error. Budget
filters retain the original winning plans. Inconsistent winner timings,
throughput units, or measured medians also produce an error rather than a
misleading plot.

### Example matching the current comparison slide

For the Llama report used in the presentation:

```bash
python -m benchmarking.quickstart_tradeoff /path/to/report/seq1024/seqsperstep64 \
  --outdir figures/slide12 --pptx \
  --full-ymax 4800 --zoom-ylim 3450 4100 \
  --unconstrained-peak-gib 296.3293 \
  --title 'Tight budgets preserve most of the throughput.' \
  --subtitle 'Llama 8B · 64K tokens per step · RTX 5090 · 160 GiB host spill pool'
```

The explicit 296.3293 GiB override matches the presentation's persistent FP32
gradient illustration. Omit it for an unmodified quickstart report's peak.

### Tradeoff outputs and interpretation

| File | Purpose |
| --- | --- |
| `tradeoff.png`, `tradeoff.svg`, `tradeoff.pdf` | Slide-ready image and vector figure, 12 × 4 inches |
| `tradeoff-slide.pptx` | With `--pptx`: 16:9 slide, editable title/subtitle/takeaway, aspect-preserving plot image |
| `winners.csv` | Selected budgets, measured/simulated rates, chosen compute ceilings, time shares and geometry identifiers |
| `metadata.json` | Source paths/SHA-256, units, axis ranges, optional unconstrained reference, all plotted values and definitions |

- The left column shows the same throughput series twice: zero-based scale
  above, detailed scale below, separated by a dark dotted divider. Grid lines
  use a thicker, contrasting solid stroke.
- **Measured** is the recorded per-budget median, verified against `steps.csv`
  when available. **Simulated** is the winning plan's predicted throughput.
- **Chosen recompute (no stalls)** is units per step divided by effective
  compute plus recomputation time for that budget's winning plan. It retains
  the chosen geometry and all task costs, including optimizer work, while
  excluding idle and final-writeback waiting. It is a profiled compute-only
  ceiling and can vary by budget.
- **Unconstrained (peak GiB)** is the all-save/unconstrained profile-cost floor,
  not a measured fully resident run. Its source must have the same step size.
- The right plot uses **simulated** effective-compute, recompute and idle
  shares. Idle includes final writeback; the three shares sum to 100%.
- All four throughput and six share endpoint labels use the smallest and
  largest selected budgets. The divider is not a data/reference line.
- The script exports files locally. Import the PPTX slide into Google Slides,
  or insert the PNG in an existing slide; it does not overwrite an online deck.
