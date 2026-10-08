"""Compare a GEMM epilogue for the dense LoRA residual; no production patch."""
import json
from pathlib import Path
import statistics
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parents[1]/"evidence/dense-epilogue.json"

def separate(x,w,a,b):
    return F.linear(x,w) + F.linear(F.linear(x,a.to(x.dtype)),b.to(x.dtype))

def combined(x,w,a,b):
    base=F.linear(x,w)
    inner=F.linear(x,a.to(x.dtype))
    return torch.addmm(base,inner,b.to(x.dtype).T)

def measure(fn, values, seed):
    call=torch.compile(fn,fullgraph=True)
    x,w,a,b=values
    for _ in range(5):
        y=call(*values)
        torch.autograd.grad(y,(x,a,b),seed)
    times=[]
    for _ in range(40):
        start=torch.cuda.Event(enable_timing=True);mid=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();y=call(*values);mid.record()
        grads=torch.autograd.grad(y,(x,a,b),seed);end.record();end.synchronize()
        times.append((start.elapsed_time(mid),mid.elapsed_time(end)))
    return y.detach(),tuple(g.detach() for g in grads),[statistics.median(t[i] for t in times) for i in (0,1)]

rows=[]
for d,h in [(768,2304),(1536,4096)]:
    torch.manual_seed(124)
    x=(torch.randn(2048,d,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_()
    w=torch.randn(h,d,device="cuda",dtype=torch.bfloat16)*.02
    a=(torch.randn(32,d,device="cuda",dtype=torch.float32)*.02).requires_grad_()
    b=(torch.randn(h,32,device="cuda",dtype=torch.float32)*.02).requires_grad_()
    seed=torch.randn(2048,h,device="cuda",dtype=torch.bfloat16)*.1
    old,go,to=measure(separate,(x,w,a,b),seed)
    new,gn,tn=measure(combined,(x,w,a,b),seed)
    errors=[float((left.float()-right.float()).norm()/right.float().norm().clamp_min(1e-20)) for left,right in zip((new,*gn),(old,*go))]
    assert max(errors)<.02,errors
    rows.append(dict(tokens=2048,width=d,output_width=h,separate_ms=to,combined_ms=tn,relative_l2_errors=errors))
    OUT.write_text(json.dumps(rows,indent=2)+"\n")
    print(rows[-1],flush=True)
