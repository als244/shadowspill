"""What every harness trains: a checkpoint loads into the catalog model the
objective wraps, under the model's own parameter names, which are the
names the family's reader maps."""

from baselines.common.geometry import Geometry
from baselines.common.models import ModelRequest
from baselines.common.training import TrainingModel


class Recording:
    """A reader that records what it was asked to load into."""

    def load_into(self, module, **options):
        self.module, self.options = module, options
        return {"written": [name for name, _ in module.named_parameters()]}


def test_a_checkpoint_loads_into_the_wrapped_model():
    model = TrainingModel(
        ModelRequest("llama3-tiny", "mlops"), Geometry(64, 1024, 128), dtype="bfloat16"
    )
    checkpoint = Recording()
    report = model.load_checkpoint(checkpoint, copy=print)
    assert checkpoint.module is model.model and checkpoint.options == {"copy": print}
    assert "embed.weight" in report["written"]
    assert not any(name.startswith("model.") for name in report["written"])
