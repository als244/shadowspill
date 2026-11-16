"""Qwen model selection and presets; shared text settings own preparation."""

from dataclasses import dataclass
from math import isfinite

from examples.training.text.settings import TextSettings
from shadowspill_client import LoRA, Model, Objective, Weights
from shadowspill_client.parallelism import data_parallel, expert_parallel


def objective(weight):
    return Objective.sum(
        {
            "tokens": Objective("shadowspill/cross_entropy@1"),
            "balance": Objective("shadowspill/moe_balance@1"),
        },
        weights={"balance": weight},
    )


def parallelism(ranks, kind="dp"):
    fields = (
        ("inputs", "tokens"),
        ("loss_inputs", "target_tokens"),
        ("loss_inputs", "loss_mask"),
        ("loss_inputs", "weights"),
        ("loss_inputs", "router_mask"),
    )
    if kind == "dp":
        return data_parallel(ranks, fields=fields)
    if kind == "ep":
        return expert_parallel(
            ranks,
            fields=fields,
            experts=(
                "blocks.*.moe.experts.gate_up_weight",
                "blocks.*.moe.experts.down_weight",
            ),
        )
    raise ValueError("parallelism must be dp or ep")


@dataclass(frozen=True)
class Settings(TextSettings):
    data: str
    checkpoint: str | None
    lora_rank: int | None = 32
    lora_alpha: float = 32
    compute_dtype: str = "bfloat16"
    balance_weight: float | None = None
    family: str = "qwen35moe"
    preset: str | None = None
    parallelism: str = "dp"

    def __post_init__(self):
        if self.balance_weight is not None and (
            type(self.balance_weight) not in (int, float)
            or not isfinite(self.balance_weight)
            or self.balance_weight < 0
        ):
            raise ValueError("balance_weight must be finite and nonnegative")
        if self.compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("compute_dtype must be float32 or bfloat16")
        if self.family not in ("qwen3moe", "qwen35moe"):
            raise ValueError("unsupported Qwen model family")
        if self.lora_rank is not None and (
            type(self.lora_rank) is not int or self.lora_rank < 1
        ):
            raise ValueError("lora_rank must be positive or None for full training")
        if self.parallelism not in ("dp", "ep"):
            raise ValueError("parallelism must be dp or ep")
        if self.parallelism == "ep" and (
            self.ranks < 2
            or self.lora_rank is not None
            or self.compute_dtype != "bfloat16"
            or any(self.chunk_sizes)
            or any(self.moe_chunk_sizes)
        ):
            raise ValueError("EP requires unchunked BF16 full training on 2+ ranks")
        if self.checkpoint is not None and (
            not self.checkpoint or not isinstance(self.checkpoint, str)
        ):
            raise ValueError("checkpoint must be an absolute server path or Hub name")
        if self.preset is not None and (
            not self.preset or not isinstance(self.preset, str)
        ):
            raise ValueError("preset must name an installed model configuration")
        if (self.checkpoint is None) == (self.preset is None):
            raise ValueError("supply exactly one checkpoint or random model preset")
        value = self.lora_alpha
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            raise ValueError("lora_alpha must be finite and positive")
        super().__post_init__()
        self.request()

    @property
    def default_microbatch_sizes(self):
        return (1, 2, 4, 8, 16, 32) if self.family == "qwen3moe" else (1, 2, 4, 8)

    @property
    def application(self):
        family = "qwen3" if self.family == "qwen3moe" else "qwen35"
        method = "full" if self.lora_rank is None else "lora"
        return f"shadowspill.example.{family}_{method}@1"

    def request(self):
        model_options = {
            "config": {
                "architecture": {"max_seq_len": self.sequence_length},
                **({"lora_seed": self.seed} if self.lora_rank is not None else {}),
                **({"preset": self.preset} if self.preset is not None else {}),
                **({"implementation": "mlops_ep"} if self.parallelism == "ep" else {}),
            },
            "precision": {
                "compute_dtype": self.compute_dtype,
                "grad_dtype": "float32",
                **(
                    {"lora_dtype": self.compute_dtype}
                    if self.lora_rank is not None
                    else {}
                ),
            },
            "lora": LoRA(self.lora_rank, alpha=self.lora_alpha)
            if self.lora_rank is not None
            else None,
        }
        model = (
            Model.from_pretrained(self.checkpoint, **model_options)
            if self.checkpoint is not None
            else Model(
                f"shadowspill/{self.family}@1",
                initialize=Weights.random(seed=self.seed),
                **model_options,
            )
        )
        return super().request(
            model=model,
            objective=objective(
                self.balance_weight if self.balance_weight is not None else 0.001
            ),
            parallelism=parallelism(self.ranks, self.parallelism),
        )


def qwen3_settings(**options):
    """Existing eight-worker full-training defaults, expressed as one global batch."""
    return Settings(
        **{
            "family": "qwen3moe",
            "checkpoint": None,
            "preset": None if options.get("checkpoint") is not None else "30b-a3b",
            "lora_rank": None,
            "ranks": 8,
            "sequence_length": 4096,
            "sequences_per_step": 512,
            "learning_rate": 3e-4,
            "execution_gib": 77,
            "spill_gib": 96,
            "external_headroom_mib": 768,
            "eval_every": 0,
            "checkpoint_every": 5,
            **options,
        }
    )
