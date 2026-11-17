"""Full-node qualification uses the public setup and complete global workloads."""

import subprocess
import sys

import pytest

from qualification.public_training import settings
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client.protocol.setup import TrainingSetup


@pytest.mark.parametrize("parallelism", ["dp", "ep"])
def test_qwen_public_campaign_preserves_readme_batch_and_build_choices(parallelism):
    setup = resolve_training(
        load_registry(["shadowspill", "shadowspill-workloads"]),
        TrainingSetup(**settings(parallelism=parallelism)),
    )
    assert setup.context.parallelism.world_size == 8
    assert setup.config.batches[0].limits["token_capacity"] == 2097152
    assert len(setup.config.preparation.cases) == 6
    inventory = capture_inventory(prepare_layouts(setup))
    complete = set()
    for candidates in inventory.families.values():
        for candidate in candidates:
            total = 0
            for use in candidate.workers.values():
                template = inventory.templates[use.main.template]
                size = template.model_build.config.get("expert_options")
                if parallelism == "ep":
                    assert size["config"]["ep_size"] == 8
                    assert size["config"]["tokens_per_rank"] in (8192, 16384, 32768)
                else:
                    assert not template.model_build.resources
                (capacity,) = template.binding.inputs.layout.buffer.fields[
                    "inputs", "tokens"
                ]
                total += capacity * use.main.count
            assert total <= 2097152
            if total == 2097152:
                complete.add(capacity)
    assert complete == {8192, 16384, 32768}


def test_four_campaign_settings_remain_framework_free_and_glm_uses_real_base():
    script = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from qualification.public_training import settings
for kind in ('qwen', 'glm'):
    for mode in ('dp', 'ep'):
        requested = settings(kind, mode, '/server/models/GLM-5.3-Flash')
        assert requested['parallelism']['ranks_per_replica'] == 8
        if kind == 'glm':
            assert requested['model'].initialize.kind == 'pretrained'
            assert requested['model'].lora.rank == 32
            assert requested['batch'].limits['max_sequence_length'] == 32768
            assert requested['batch'].limits['token_capacity'] == 8 * 32768
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
