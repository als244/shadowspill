"""Summarize longer runs and classify recorded Python garbage collections."""
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for file in sorted((ROOT/"evidence/llama-1b-timing").glob("*/result.json")):
    data=json.loads(file.read_text()); steps=data['steps']
    events=json.loads(file.with_name('gc-events.json').read_text())
    starts={};intervals=[]
    for e in events:
        if e['phase']=='start':starts[e['generation']]=e['time']
        elif e['generation'] in starts:
            a=starts.pop(e['generation']); intervals.append(dict(start=a,end=e['time'],generation=e['generation'],milliseconds=(e['time']-a)*1e3,collected=e['collected']))
    t=[s['seconds'] for s in steps]
    first=steps[0]['started_at']; last=steps[-1]['started_at']+steps[-1]['seconds']
    warmup=float(re.search(r'WARMUP \d+ iterations ([0-9.]+) seconds',file.with_name('console.log').read_text()).group(1))
    measured=[e for e in intervals if e['start']<last and e['end']>first]
    warmed=[e for e in intervals if e['start']<first and e['end']>first-warmup]
    row=dict(case=file.parent.name,steps=len(t),median_ms=statistics.median(t)*1000,mean_ms=statistics.mean(t)*1000,
             min_ms=min(t)*1000,max_ms=max(t)*1000,p95_ms=sorted(t)[int(.95*(len(t)-1))]*1000,
             aggregate_tokens_per_second=data['config']['tokens']*len(t)/sum(t),
             max_measured_gc_ms=max([e['milliseconds'] for e in measured],default=0),
             max_warmup_gc_ms=max([e['milliseconds'] for e in warmed],default=0),
             measured_generation2=sum(e['generation']==2 for e in measured),
             peak_host_rss_gib=data['peak_host_rss_execution_bytes']/2**30)
    file.with_name('timing-analysis.json').write_text(json.dumps(dict(summary=row,collections=intervals),indent=2)+'\n')
    rows.append(row)
(ROOT/'timing-followup.json').write_text(json.dumps(rows,indent=2)+'\n')
text=['# Longer 1.18B Llama timing checks','',
      'Same BF16 base, rank-32 FP32 factors, FP32 gradients/moments and 2048-token batches as the scale comparison. At least 10 exact-step warmups and 2 seconds at LR=0, then 40 measured updates. GC stays enabled; a callback records its timestamps. No artificial GC disabling or cache release is used.','',
      '| Case | Median ms | Mean ms | p95 ms | Range ms | Aggregate tok/s | Max GC during steps ms | Max warmup GC ms | Host RSS GiB |',
      '|---|---:|---:|---:|---|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['case']} | {r['median_ms']:.2f} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} | {r['min_ms']:.2f}–{r['max_ms']:.2f} | {r['aggregate_tokens_per_second']:,.0f} | {r['max_measured_gc_ms']:.3f} | {r['max_warmup_gc_ms']:.2f} | {r['peak_host_rss_gib']:.2f} |")
text += ['', 'The earlier 10-step runs remain in `evidence/llama-1b`; these follow-ups do not overwrite them. A long generation-2 collection during warmup, followed by stable measured steps, supports compilation-related garbage collection as the likely source of the earlier isolated stalls. The original runs did not record GC timestamps, so that attribution is an inference.', '',
         '`lora` freezes the output head. `lora_head` also trains its low-rank factors; it still freezes the large original head matrix. Each case retains complete graphpairs and optimizer-allocation audits.']
(ROOT/'TIMING.md').write_text('\n'.join(text)+'\n')
print(json.dumps(rows,indent=2))
