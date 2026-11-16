"""One actual-batch selection algorithm used before upload and on the server."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from shadowspill_client import BatchConfig
from shadowspill_client._records import number
from shadowspill_client.placement import InputPlacement
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.payload import BatchSnapshot
from shadowspill_client.protocol.training import ResolvedTraining

from .coverage import CoverageCertificate
from .packing import PackedAssignment, PackingError, pack_batch
from .shapes import PackingSchedule
from .validation import ValidatedBatch, validate_batch


@dataclass(frozen=True)
class PackingCandidate:
    id: str
    batch: int
    schedule: PackingSchedule


def admissible_batches(
    validated: ValidatedBatch, batches: Sequence[BatchConfig]
) -> tuple[int, ...]:
    allowed = tuple(
        i
        for i, batch in enumerate(batches)
        if all(
            total <= number(batch.limits[name], name)
            for name, total in validated.capacities.items()
        )
    )
    if not allowed:
        raise ValueError("input capacities fit no requested batch option")
    return allowed


def validate_snapshot(
    snapshot: BatchSnapshot, config: ResolvedTraining, key: OperationKey
) -> ValidatedBatch:
    if key not in config.operations or key.outputs is None:
        raise ValueError("operation/output/mode was not prepared for this client")
    output = config.outputs[key.outputs].select(forward=key.operation == "forward")
    paths = {("inputs", *path) for path in output.model_inputs} | {
        ("loss_inputs", *path) for path in output.loss_inputs
    }
    if key.operation != "forward" and config.participation.path:
        paths.add(("loss_inputs", *config.participation.path))
    # This union permits one value-validation pass; selection still requires
    # an original batch option in its entirety, never a mix of its limits.
    limits = BatchConfig(
        limits={
            name: max(number(batch.limits[name], name) for batch in config.batches)
            for name in config.batches[0].limits
        }
    )
    validated = validate_batch(
        snapshot, config.inputs, limits, required_paths=tuple(sorted(paths))
    )
    admissible_batches(validated, config.batches)
    return validated


def select_packing(
    validated: ValidatedBatch,
    batches: Sequence[BatchConfig],
    placement: InputPlacement,
    candidates: Sequence[PackingCandidate],
    *,
    proofs: Mapping[int, CoverageCertificate],
    groups: Sequence[Sequence[int]] | None = None,
) -> tuple[PackingCandidate, PackedAssignment]:
    """Try the saved cost order, with optional constructive fallback proofs."""
    allowed = admissible_batches(validated, batches)
    for candidate in candidates:
        if candidate.batch not in allowed:
            continue
        limits = batches[candidate.batch]
        batch = replace(validated, config=limits)
        try:
            assignment = pack_batch(
                batch,
                validated.schema,
                limits,
                placement,
                candidate.schedule,
                groups=groups,
            )
        except PackingError:
            proof = proofs.get(candidate.batch)
            if proof is None or candidate.schedule != proof.schedule:
                continue
            assignment = proof.pack(batch, groups=groups)
        return candidate, assignment
    raise PackingError(
        "no packing found for this batch in the prepared layouts; "
        "submit a smaller batch or prepare additional layouts"
    )
