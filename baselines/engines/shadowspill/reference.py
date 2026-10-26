"""Tiny-model oracle access through bounded public tensor exports."""

import torch
import torch.distributed as dist

from baselines.common.reference import Access, Reference
from baselines.common.training import TrainingModel
from shadowspill.artifacts.tensors import TensorChunk


def assemble(chunks: list[tuple[TensorChunk, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Reconstruct physical storage, checking shard coverage and duplicate replicas."""
    grouped: dict[str, list[tuple[TensorChunk, torch.Tensor]]] = {}
    for request, value in chunks:
        grouped.setdefault(request.name, []).append((request, value))
    values = {}
    for name, entries in grouped.items():
        request = entries[0][0]
        owner = torch.empty(request.storage_elements, dtype=request.dtype)
        covered = 0
        for part, value in sorted(entries, key=lambda item: item[0].offset):
            start, end = part.offset, part.offset + value.numel()
            if start > covered or end > owner.numel():
                raise ValueError(f"incomplete tensor export: {name}")
            overlap = min(covered, end) - start
            if overlap > 0 and not torch.equal(
                owner[start : start + overlap], value[:overlap]
            ):
                raise ValueError(f"replicated tensor exports disagree: {name}")
            owner[start:end].copy_(value)
            covered = max(covered, end)
        if covered != owner.numel():
            raise ValueError(f"incomplete tensor export: {name}")
        values[name] = owner.as_strided(
            request.shape, request.stride, request.storage_offset
        )
    return values


class Snapshot(Access):
    exposes = ("first_moment", "second_moment", "master")

    def __init__(self, executor, model, device):
        super().__init__(None, device)
        self.executor = executor
        self.names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        self.values = {}

    def refresh(self):
        chunks: list[tuple[TensorChunk, torch.Tensor]] = []
        self.executor.export_tensors(
            lambda request, value: chunks.append((request, value.clone())),
            chunk_bytes=4 << 20,
        )
        if dist.is_initialized():
            ranks: list[list[tuple[TensorChunk, torch.Tensor]]] = [
                [] for _ in range(dist.get_world_size())
            ]
            dist.all_gather_object(ranks, chunks)
            chunks = [chunk for rank in ranks for chunk in rank]
        self.values = assemble(chunks)
        # Export writes shared storage once; expose either tied model name.
        aliases = {
            item.object_id: item.alias_group_id
            for item in self.executor.plan.program.objects
        }
        registrations = self.executor.plan.program.bindings["registrations"]
        owners = {
            aliases[item["object_id"]]: self.values[item["name"]]
            for item in registrations
            if item["name"] in self.values
        }
        for item in registrations:
            if item["name"] not in self.values:
                self.values[item["name"]] = owners[aliases[item["object_id"]]]

    def parameter(self, parameter):
        name = self.names[id(parameter)]
        return self.values.get("master." + name, self.values[name]).to(self.device)

    def buffer(self, name, buffer):
        return self.values[name].to(self.device)

    def gradient(self, parameter):
        return None

    def state(self, parameter, key):
        value = self.values.get(f"optimizer.{self.names[id(parameter)]}.{key}")
        return None if value is None else value.to(self.device)

    def evidence(self, model):
        counts: dict[str, dict[str, int]] = {"stepped": {}, "moments": {}}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            values = [("stepped", self.values.get("master." + name, self.values[name]))]
            values.extend(
                ("moments", self.values[f"optimizer.{name}.{key}"])
                for key in ("exp_avg", "exp_avg_sq")
            )
            for kind, value in values:
                key = str(value.dtype)
                counts[kind][key] = counts[kind].get(key, 0) + value.numel()
        if not counts["moments"]:
            raise ValueError("optimizer evidence is empty")
        return counts


def create(case, trainer, geometry, stack):
    if not case.settings.verify_reference:
        return None, None, None
    request = case.settings.request()
    model = TrainingModel(request, geometry, dtype=case.precision.model_dtype)
    access = Snapshot(trainer.executor, model, case.device)
    access.refresh()
    group = None
    if case.world > 1:
        group = dist.new_group(backend="nccl", device_id=case.device)
        stack.callback(dist.destroy_process_group, group)
    oracle = Reference(
        model,
        request,
        geometry,
        case.precision,
        case.settings,
        case.device,
        case.world,
        access=access,
        group=group,
    )
    return model, access, oracle
