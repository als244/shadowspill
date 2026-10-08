"""Probe generic gradient-demand handling without modifying MLOps."""
from pathlib import Path
import types
import sys
import pytest
import torch
from torch.nn import functional as F

source = Path(__file__).with_name('head_candidate.py').read_text()
source = source.split('\nIMPLEMENTATION = register_implementation(')[0]
source = source.replace('mlops::head_loss_builtin_chunked_fwd', 'lora_design_probe::head_loss_chunked_fwd')
probe = types.ModuleType('mlops.providers.builtin.lora_design_probe')
probe.__package__ = 'mlops.providers.builtin'
sys.modules[probe.__name__] = probe
exec(compile(source, 'head_candidate.py', 'exec'), probe.__dict__)


@pytest.mark.parametrize('needs', [(True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize('compiled', [False, True])
def test_loss_and_needed_gradients(needs, compiled):
    torch.manual_seed(32)
    hidden = torch.randn(7, 5, requires_grad=needs[0])
    weight = torch.randn(11, 5, requires_grad=needs[1])
    targets = torch.tensor([1, 3, -100, 4, 9, 0, 2])
    h_ref = hidden.detach().clone().requires_grad_(needs[0])
    w_ref = weight.detach().clone().requires_grad_(needs[1])
    call = lambda h,w: probe.apply(h,w,targets,chunk_size=3,reduction='sum')
    if compiled:
        call = torch.compile(call, backend='aot_eager', fullgraph=True)
    loss = call(hidden, weight)
    expected = F.cross_entropy(h_ref @ w_ref.T, targets, reduction='sum')
    torch.testing.assert_close(loss, expected)
    if any(needs):
        (1.7*loss).backward()
        (1.7*expected).backward()
        for actual, ref, required in zip((hidden,weight), (h_ref,w_ref),needs):
            if required:
                torch.testing.assert_close(actual.grad, ref.grad, atol=3e-6,rtol=3e-6)
            else:
                assert actual.grad is None
    _, seed_h, seed_w = probe._forward_op(hidden,weight,targets,3,1,None,*needs)
    assert (seed_h is not None) is needs[0]
    assert (seed_w is not None) is needs[1]
