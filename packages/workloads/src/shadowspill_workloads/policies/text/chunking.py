"""Text policies select domains and boundaries using provider-owned contracts.

Unknown operators remain errors in selected regions. Sequence mixers retain
their complete inputs unless an explicit provider contract permits splitting.
"""

from dataclasses import dataclass

from shadowspill.integrations.mlops import operator_chunk_rules
from shadowspill.pytorch.chunking import (
    AxisChunks,
    AxisFrom,
    ChunkAxis,
    OperatorRule,
)
from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.task.inputs import TaskInputRole

_LINEAR_ATTENTION = "mlops.linear_attention_fla_gated_delta_rule_fwd.default"


def linear_attention_head_chunks(stage, chunk_size, variants):
    """Keep all tokens while slicing matched key/value head groups.

    This provider accepts packed T,H,D inputs and per-value-head gate state.
    Its sequence metadata stays shared. The shape checks refuse other layouts
    rather than treating an arbitrary attention stage as head independent.
    """
    matrices = {
        i: value
        for i, (value, role) in enumerate(
            zip(stage.inputs, stage.input_roles, strict=True)
        )
        if value is not None
        and role is not TaskInputRole.CONTROL
        and len(value.shape) == 3
    }
    if len(matrices) != 3:
        raise ValueError("linear-attention head chunks require three T,H,D inputs")
    tokens = {value.shape[0] for value in matrices.values()}
    heads = [value.shape[1] for value in matrices.values()]
    key_heads, value_heads = min(heads), max(heads)
    if len(tokens) != 1 or sorted(heads) != [key_heads, key_heads, value_heads]:
        raise ValueError("linear-attention inputs must align tokens and head groups")
    if key_heads <= 0 or value_heads % key_heads:
        raise ValueError("value heads must be a positive multiple of key heads")
    ratio = value_heads // key_heads
    token_count = tokens.pop()
    axes = {}
    for i, (value, role) in enumerate(
        zip(stage.inputs, stage.input_roles, strict=True)
    ):
        if value is None or role is TaskInputRole.CONTROL:
            continue
        if i in matrices:
            axes[i] = ChunkAxis(1, after=value.shape[1] // key_heads)
        elif value.shape == (token_count, value_heads):
            axes[i] = ChunkAxis(1, after=ratio)
        elif value.shape == (value_heads,):
            axes[i] = ChunkAxis(0, after=ratio)
        else:
            raise ValueError(
                f"unclassified linear-attention head input {i}: {value.shape}"
            )
    rules = (
        OperatorRule(
            _LINEAR_ATTENTION,
            "head-groups-1",
            (AxisFrom(2), ChunkAxis(2, after=ratio), ChunkAxis(2, after=ratio)),
        ),
        OperatorRule(
            "mlops.linear_attention_fla_gated_delta_rule_bwd.default",
            "head-groups-1",
            tuple(AxisFrom(i) for i in range(1, 8)),
        ),
    )
    return AxisChunks(chunk_size, axes, variants, rules)


def is_barrier(target):
    # Stacking per-stage statistics combines complete row reductions with
    # earlier stages' summaries. Keep that aggregation outside a row region.
    return target in {
        "aten.stack.default",
        # Packed sequence offsets and their token lookup use complete boundary
        # arrays. Keep their construction outside token-local execution loops.
        "aten.cumsum.default",
        "aten.searchsorted.Tensor",
        # Router histograms combine assignments from the complete microbatch.
        # Their shared destination must not be repeated inside token chunks.
        "aten.scatter_add.default",
        # Frozen compressed dense weights are shared by all token slices.
        # Decode once outside their loops, not once per chunk.
        "mlops_seqmoe.decode_weight.default",
    } or any(
        marker in target
        for marker in (
            # Keep the whole embedding prologue separate from token-local
            # projections/norms. Otherwise a frozen embedding can prevent their
            # flattened token/head statistics from acquiring a chunk domain.
            "embedding",
            "flash_attention",
            "scaled_dot_product",
            "linear_attention",
            "kimi_delta_attention",
            "causal_conv",
            "head_loss",
            "sparse_latent_attention",
            "sparse_latent_rope",
            "dsa_attention",
            "pooled_topk",
            # These providers embed complete-sequence lengths and nonlinear
            # auxiliary statistics. Output-axis rules alone cannot resize that
            # contract. Their explicit LoRA dispatch path has separate rules.
            "moe_builtin_grouped_gemm",
            "moe_sonicmoe",
        )
    )


@dataclass(frozen=True)
class SequenceBoundaries:
    """Isolate selected sequence operations inside the ordinary module stages."""

    def assign_stages(self, graph_module, module):
        targets = tuple(
            str(n.target) for n in graph_module.graph.nodes if is_barrier(str(n.target))
        )
        return IsolateOperations(targets).assign_stages(graph_module, module)


@dataclass(frozen=True)
class TextAxisChunks:
    """Explicit fixed-window row selection, including flattened batch/sequence axes."""

    sequence_length: int
    batch_size: int
    chunk_size: int
    variants: tuple[str, ...] = ("recompute",)
    linear_attention_head_chunk_size: int | None = None

    def configure_stage(self, stage):
        if stage.terminal:
            return None
        if (
            self.linear_attention_head_chunk_size is not None
            and _LINEAR_ATTENTION in stage.operators
        ):
            return linear_attention_head_chunks(
                stage, self.linear_attention_head_chunk_size, self.variants
            )
        if any(is_barrier(target) for target in stage.operators):
            return None
        # Control derivation and embedding stay in the model prologue.
        if any(
            "embedding" in target or "packed_sequence" in target
            for target in stage.operators
        ):
            return None
        axes = {}
        for i, (value, role) in enumerate(
            zip(stage.inputs, stage.input_roles, strict=True)
        ):
            if value is None or role in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}:
                continue
            shape = value.shape
            if (
                len(shape) >= 2
                and shape[0] == self.batch_size
                and shape[1] == self.sequence_length
            ):
                axes[i] = ChunkAxis(1)
            elif (
                len(shape) == 4
                and shape[0] == self.batch_size
                and shape[2] == self.sequence_length
            ):
                # Attention providers may return B,H,S,D instead of B,S,H,D.
                # The projection region must slice that boundary as well.
                axes[i] = ChunkAxis(2)
            elif shape and shape[0] == self.batch_size * self.sequence_length:
                axes[i] = ChunkAxis(0, before=self.batch_size)
        return (
            AxisChunks(
                self.chunk_size,
                axes,
                self.variants,
                operator_chunk_rules(self.batch_size),
            )
            if axes
            else None
        )
