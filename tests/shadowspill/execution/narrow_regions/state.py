"""Bounded shard exports, full-tensor reconstruction and exact resume checks."""


def export(executor):
    import torch

    pieces = {}

    def collect(request, chunk):
        assert chunk.device.type == "cpu" and chunk.nbytes <= 4096
        assert torch.isfinite(chunk).all()
        record = pieces.setdefault(
            request.name,
            dict(
                shape=request.shape,
                storage_elements=request.storage_elements,
                region_offset=request.region_offset,
                region_elements=request.region_elements,
                chunks=[],
            ),
        )
        record["chunks"].append((request.offset, chunk.clone()))

    executor.export_tensors(collect, chunk_bytes=4096)
    return pieces


def same_pieces(left, right):
    import torch

    assert left.keys() == right.keys()
    count = 0
    for name in left:
        a, b = left[name], right[name]
        assert {k: v for k, v in a.items() if k != "chunks"} == {
            k: v for k, v in b.items() if k != "chunks"
        }
        for (offset_a, value_a), (offset_b, value_b) in zip(
            a["chunks"], b["chunks"], strict=True
        ):
            assert offset_a == offset_b and torch.equal(value_a, value_b), (
                name,
                offset_a,
            )
            count += value_a.numel()
    return count


def merge(ranks, expected):
    import torch

    names = {name for rank in ranks for name in rank["tensors"]}
    assert names == expected.keys(), (names, expected.keys())
    merged = {}
    for name, wanted in expected.items():
        value = torch.empty_like(wanted).reshape(-1)
        covered = torch.zeros(value.numel(), dtype=torch.bool)
        for rank in ranks:
            item = rank["tensors"].get(name)
            if item is None:
                continue
            assert item["storage_elements"] == wanted.numel()
            for offset, chunk in item["chunks"]:
                assert chunk.dtype == wanted.dtype
                region = slice(offset, offset + chunk.numel())
                overlap = covered[region]
                if overlap.any():
                    assert torch.equal(value[region][overlap], chunk[overlap]), (
                        name,
                        "replicated state differs",
                    )
                value[region] = chunk
                covered[region] = True
        assert covered.all(), (name, "missing shard coverage")
        merged[name] = value.reshape_as(wanted)
    return merged
