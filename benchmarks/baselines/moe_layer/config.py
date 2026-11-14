"""Pure configuration and useful-work accounting for the layer benchmark."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Shape:
    width: int = 2048
    hidden: int = 768
    experts: int = 128
    top_k: int = 8
    lora_rank: int = 32
    lora_alpha: float = 32.0

    def __post_init__(self):
        if min(asdict(self).values()) < 1 or self.top_k > self.experts:
            raise ValueError("positive dimensions and top_k <= experts required")

    def flops(self, tokens, training):
        """Count useful linear algebra (FMA=2), excluding padding/recomputation.

        Full training: each linear has forward, input and weight gradients.
        LoRA: frozen base/router have forward and input gradients; each adapter
        has all three. Activation functions, top-k and communication are not
        added to the numerator, but their execution is inside the timed region.
        """
        if tokens < 1 or training not in {"full", "lora"}:
            raise ValueError("positive tokens and full/lora training required")
        projections = 3 * tokens * self.top_k * self.width * self.hidden
        router = tokens * self.width * self.experts
        adapters = (
            tokens * self.top_k * self.lora_rank * (2 * self.width + 3 * self.hidden)
            if training == "lora"
            else 0
        )
        return {
            "forward": 2 * (projections + router + adapters),
            "backward": (4 if training == "full" else 2) * (projections + router)
            + 4 * adapters,
            "total": (6 if training == "full" else 4) * (projections + router)
            + 6 * adapters,
        }


def summarize(samples_by_rank):
    import statistics

    lengths = {len(samples) for samples in samples_by_rank}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ValueError("every rank must have the same nonempty sample count")
    maximum = [max(cycle) for cycle in zip(*samples_by_rank, strict=True)]
    return {
        "per_iteration_max_rank_seconds": maximum,
        "median_seconds": statistics.median(maximum),
        "min_seconds": min(maximum),
        "max_seconds": max(maximum),
    }


MODEL_SHAPES = {
    "qwen3-30b-a3b": (2048, 768, 128, 8),
    "glm53-flash": (4096, 2048, 288, 8),
}


def add_shape_arguments(parser):
    parser.add_argument("--model", choices=list(MODEL_SHAPES), default="qwen3-30b-a3b")
    for name in ("width", "hidden", "experts", "top-k"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument(
        "--router-weight-dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Router weight storage; logits and gradient accumulation stay FP32",
    )


def requested_shape(args, world_size=1):
    dimensions = dict(
        zip(
            ("width", "hidden", "experts", "top_k"),
            MODEL_SHAPES[args.model],
            strict=True,
        )
    )
    dimensions.update(
        {
            name: getattr(args, name)
            for name in dimensions
            if getattr(args, name) is not None
        }
    )
    if args.tiny:
        dimensions.update(width=128, hidden=128, experts=max(world_size, 8), top_k=2)
    return Shape(
        **dimensions,
        lora_rank=args.lora_rank,
        lora_alpha=32.0 if args.lora_alpha is None else args.lora_alpha,
    )
