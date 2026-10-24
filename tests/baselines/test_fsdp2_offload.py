"""The harness honors torchtune's stream and tensor-lifetime assumptions."""

import subprocess
import sys
import textwrap

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream ordering")
def test_offloader_integration_protects_gathered_storage_before_fsdp_frees_it():
    # Isolate the CUDA allocator from tests which install a ShadowSpill pool.
    script = textwrap.dedent(
        """
        import torch
        from torch import nn
        from baselines.engines.fsdp2.offload import blocks_context

        torch.set_num_threads(2)
        count = 1024 * 1024
        host = torch.empty(count, dtype=torch.float32, pin_memory=True)
        del host

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.full((count,), 3.0, device='cuda'))

            def forward(self, value):
                # An autograd save of a weight view calls this same pack hook.
                self.saved = offloader.pack_hook(self.weight.view(-1))
                return value

        block = Block()
        offloader = blocks_context([block])
        # A rejected call must not enter the saved-tensor context.
        with torch.cuda.stream(torch.cuda.Stream()):
            try:
                block(None)
            except RuntimeError as error:
                assert 'default compute stream' in str(error)
            else:
                raise AssertionError('accepted an unsupported compute stream')
        assert offloader.is_first_forward_call

        replacements = []
        def release(module, inputs, output):
            # FSDP registers its post-forward release after our integration.
            module.weight.untyped_storage().resize_(0)
            replacements.append(torch.full((count,), 7.0, device='cuda'))

        block.register_forward_hook(release)
        with torch.cuda.stream(offloader.s1):
            torch.cuda._sleep(100_000_000)
        block(None)
        torch.cuda.synchronize()
        actual, offloaded = offloader.tracker[block.saved]
        assert offloaded and torch.all(actual == 3), actual[:8]
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
