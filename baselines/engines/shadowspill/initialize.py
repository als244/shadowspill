"""Fill only Executor-owned chunks, using the catalog initializer or checkpoint."""

from dataclasses import replace

from workloads.models.shared.routed_lora import FACTORS
from workloads.recipes.text.initialization import Initialize as CatalogInitialize


class Initialize:
    def __init__(self, settings):
        request = settings.request()
        self.generated = CatalogInitialize(
            request.entry.family,
            request.entry.config(**request.config),
            seed=settings.seed,
        )
        self.checkpoint = (
            None
            if settings.checkpoint is None
            else request.checkpoint(settings.checkpoint)
        )
        self.written, self.used = set(), set()

    def __call__(self, request, destination):
        request = replace(request, name=request.name.removeprefix("model."))
        generated = request.name.startswith("rotary.") or request.name.rsplit(".", 1)[
            -1
        ] in {"lora_a", "lora_b", *FACTORS}
        if self.checkpoint is None or generated:
            self.generated(request, destination)
        else:
            self.checkpoint.copy_chunk(
                request.name, request.shape, request.offset, destination
            )
            if request.name not in self.written:
                self.used.update(
                    self.checkpoint.sources(request.name, self.checkpoint.config).keys
                )
                self.written.add(request.name)

    def report(self):
        import torch.distributed as dist

        if self.checkpoint is None:
            raise ValueError("no checkpoint was requested")
        written, used = self.written, self.used
        if dist.is_initialized():
            parts: list[tuple[set[str], set[str]]] = [
                (set(), set()) for _ in range(dist.get_world_size())
            ]
            dist.all_gather_object(parts, (written, used))
            written = set().union(*(part[0] for part in parts))
            used = set().union(*(part[1] for part in parts))
        available = self.checkpoint.keys()
        unused = [
            key
            for key in available
            if key not in used and not key.startswith(self.checkpoint.ignored_prefixes)
        ]
        if unused:
            raise ValueError(f"checkpoint keys the model did not take: {unused[:5]}")
        return {"written": sorted(written), "unused": unused}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.checkpoint is not None:
            self.checkpoint.close()
