"""Independent NumPy inference from tiny downloaded model-weight fixtures."""

import json
import tarfile

import numpy as np


def read_weights(directory, initial):
    """Inspect one fully owned rank without the engine or a model framework."""
    with tarfile.open(directory / "ranks/rank-00000.ssp") as archive:
        manifest = json.loads(archive.extractfile("manifest.json").read())
        assert manifest["format"] == "shadowspill.weights" and manifest["version"] == 1
        assert not manifest["gathered_tensors"]
        assert "randomness" not in manifest["metadata"]
        tensors = {}
        endian = "<" if manifest["byteorder"] == "little" else ">"
        for row in manifest["tensors"]:
            raw = archive.extractfile(row["member"]).read()
            assert len(raw) == row["bytes"]
            dtype = row["dtype"].removeprefix("torch.")
            if dtype == "bfloat16":
                value = np.frombuffer(raw, dtype=endian + "u2")
                value = (value.astype(np.uint32) << 16).view(np.float32)
            else:
                value = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder(endian))
            tensors[row["name"]] = value
    weights, used = {}, set()
    for row in manifest["model_tensors"]:
        identity, view = row["tensor"]["identity"], row["view"]
        assert identity["role"] in {"parameter", "buffer", "master"}
        if identity["role"] != "parameter":
            continue
        name, source = identity["name"], row["archive"]
        if source in manifest["omitted_parameters"]:
            assert manifest["frozen_state_id"] is not None and name in initial
            weights[name] = np.array(initial[name], dtype=np.float32)
            continue
        source = manifest["derived_tensors"].get(source, source)
        value = tensors[source]
        used.add(source)
        assert view["shape"] == view["local_shape"] and not any(view["origin"])
        offset = view["storage_offset"] - view["region_offset"]
        weights[name] = np.ndarray(
            tuple(view["shape"]),
            dtype=value.dtype,
            buffer=value,
            offset=offset * value.itemsize,
            strides=tuple(stride * value.itemsize for stride in view["stride"]),
        ).copy()
    assert used == tensors.keys()  # This tiny model has no persistent buffers.
    return manifest, weights


def check_export(training, directory, oracle, initial, observation, Oracle, *, frozen):
    directory.mkdir(exist_ok=True)
    receipts = []
    for representation in ("master", "compute"):
        saved = training.export_weights(
            name="inference-" + representation,
            weights=representation,
            metadata={"cursor": 9},
        ).result(120)
        info = saved.manifest().result(30).info
        assert info.completed_updates == 3 and info.metadata["cursor"] == 9
        assert (info.frozen_base is not None) == frozen
        target = saved.download(directory / ("weights-" + representation)).result(60)
        manifest, restored = read_weights(target, initial)
        assert set(restored) == oracle.weights.keys()
        assert bool(manifest["omitted_parameters"]) == frozen
        for name, value in restored.items():
            np.testing.assert_allclose(
                value, oracle.weights[name], rtol=2e-4, atol=2e-5
            )
        inference = Oracle(restored)
        prediction, _, _ = inference.forward_backward(
            np.array(observation["outputs"]["noise"], dtype=np.float32)
        )
        np.testing.assert_allclose(
            prediction, observation["outputs"]["prediction"], rtol=2e-4, atol=2e-5
        )
        receipts.append(
            {"representation": representation, "reference": saved.reference.to_dict()}
        )
    return receipts
