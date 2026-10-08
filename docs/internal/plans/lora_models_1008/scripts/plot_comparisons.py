"""One-off figures for the measured full-model comparison; no product plot edits."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/full-model-steady-benchmark"
families = ["llama3", "qwen35", "olmoe", "qwen3moe", "qwen35moe"]
labels = ["Llama 3", "Qwen 3.5\ndense", "OLMoE", "Qwen 3\nMoE", "Qwen 3.5\nMoE"]
results = {(f,m,v):json.loads((BASE/f"mlops-{f}-{m}-{v}/result.json").read_text())
           for f in families for m in ("full","lora") for v in ("save","recompute")}
fig, axes = plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
x=np.arange(len(families)); width=.34
for mode, offset, color, label in [("full",-.17,"#5875a4","Full training"),("lora",.17,"#d99a36","LoRA rank 32")]:
    values=[results[f,mode,"save"] for f in families]
    for ax, y in zip(axes,[[1000*r['median_step_seconds'] for r in values],
                          [r['peak_host_rss_execution_bytes']/2**30 for r in values],
                          [r['pools_after_execution']['spill']['peak_allocated_bytes']/2**30 for r in values]]):
        bars=ax.bar(x+offset,y,width,color=color,label=label)
        ax.bar_label(bars,fmt="%.1f",fontsize=8,padding=2)
for ax,title,ylabel in zip(axes,["Measured step time","Peak process host memory","Peak spill allocation"],["Milliseconds (lower is better)","GiB (includes reserved pinned pool)","GiB actually allocated"]):
    ax.set(title=title,ylabel=ylabel,xticks=x,xticklabels=labels)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0,ax.get_ylim()[1]*1.12)
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle("Whole-model BF16 training vs LoRA · save mode · RTX 5090",fontsize=14)
fig.supxlabel("4 layers · width 768 · 2048 tokens/step · frozen base/head for LoRA · FP32 factors, gradients and AdamW moments",fontsize=9)
for suffix in ("png","pdf"):
    fig.savefig(ROOT/f"memory_throughput.{suffix}",dpi=180)
plt.close(fig)
print(ROOT/"memory_throughput.png")
