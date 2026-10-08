"""Small before/after numerical snapshot for the workload organization refactor."""
import sys
from dataclasses import replace
from pathlib import Path
import torch
from workloads.mlops import OLMoE, OLMoEConfig, Qwen35MoE, Qwen35MoEConfig
try:
    from workloads.mlops import Qwen3MoE, Qwen3MoEConfig
except ImportError:
    from workloads.mlops import Qwen30B as Qwen3MoE, Qwen30BConfig as Qwen3MoEConfig

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
models = [(OLMoE, OLMoEConfig(2, 32, 4, 2, 8, 4, 2, 16, 97, max_seq_len=16))]
for cls, c in [(Qwen3MoE, Qwen3MoEConfig()), (Qwen35MoE, Qwen35MoEConfig())]:
    models.append((cls, replace(c, n_layers=4, d_model=32, n_heads=4, n_kv_heads=2,
        head_dim=8, n_experts=4, top_k=2, d_ff_expert=16,
        d_ff_shared=16 if c.d_ff_shared else 0, vocab_size=97, max_seq_len=16,
        lin_k_heads=2, lin_v_heads=4, lin_k_head_dim=4, lin_v_head_dim=4)))
observed = {}
for index, (cls, cfg) in enumerate(models):
    torch.manual_seed(917)
    model = cls(cfg)
    tokens = torch.randint(cfg.vocab_size, (1, 7))
    output = model(tokens, (3, 4))
    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    observed[index] = dict(state=model.state_dict(), output=output.detach(),
        gradients={name: p.grad for name, p in model.named_parameters()})
path = ROOT / 'evidence/local-model-before.pt'
if sys.argv[1] == 'before':
    torch.save(observed, path)
    print('Saved small local numerical baseline', path, flush=True)
else:
    expected = torch.load(path, weights_only=True)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    print('PASS: all 3 local architectures have identical initialization, logits and parameter gradients', flush=True)
