"""Option records and permitted values for automodel."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Option:
    """One of AutoModel's options, as a case sets it: its name here, the
    recipe section that holds it (``distributed``; ``distributed.moe``;
    ``distributed.pipeline``; ``model``; ``model.backend``; ``loss_fn``;
    ``fp8``; ``compile``; ``step_scheduler``; ``peft``; ``optimizer``; or
    ``engine``, read by the engine to build a section), the key there when
    it differs from the name, the type of its value, what it does, its
    allowed values, and how a string is read (``names``, a comma-separated
    list of the allowed values; ``dtype``, a dtype's name)."""

    name: str
    section: str
    kind: type
    help: str
    choices: tuple = ()
    key: str | None = None
    form: str = "value"

    @property
    def field(self) -> str:
        return self.key or self.name


STRATEGIES = ("fsdp2", "megatron_fsdp", "ddp")
CHECKPOINTING = ("none", "full", "selective")
SCOPES = ("all", "language", "vision", "audio", "multimodal")
DTYPES = ("bfloat16", "float16", "float32")
LOSSES = ("chunked", "masked", "fused_linear", "te_parallel")
OPTIMIZERS = ("adamw", "adam", "fused_adam", "flash_adamw")
ATTENTION = (
    "sdpa",
    "eager",
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "te",
)


def _option(name, section, kind, help, choices=(), key=None, form="value"):
    return Option(name, section, kind, help, tuple(choices), key, form)
