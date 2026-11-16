"""The NumPy-only training oracle reads actual distributed weight archives."""

import io
import json
import tarfile
from copy import deepcopy

import numpy as np
import pytest

from engine_tests.checkpoint.test_conversion import WEIGHT, HostRank, family
from shadowspill.checkpoint import write_weights
from workload_tests._text_service_client import read_weights


@pytest.mark.parametrize("ranks,dim", [(1, 0), (2, 0), (4, 1)])
@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_real_archives_restore_model_shards_ties_and_frozen_base(
    tmp_path, ranks, dim, master, frozen
):
    (tmp_path / "ranks").mkdir()
    for state in family(ranks, dim, master=master, frozen=frozen):
        with (tmp_path / f"ranks/rank-{state.logical_rank:05}.ssp").open("wb") as sink:
            write_weights(
                HostRank(state, initialize=True),
                sink,
                chunk_bytes=17,
                frozen_state_id="frozen-base" if frozen else None,
            )
    expected = {"linear.weight": WEIGHT.numpy(), "head.weight": WEIGHT.numpy()}
    if frozen:
        expected.update({"base.weight": WEIGHT.numpy(), "base.tied": WEIGHT.numpy()})
    actual = read_weights(
        tmp_path, ["linear.weight"], expected, separate_masters=master
    )
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def archive(path, view, data):
    """Small explicit geometry, independent of the engine's logical-view helper."""
    manifest = {
        "byteorder": "little",
        "tensors": [{"name": "master", "member": "value", "dtype": "float32"}],
        "model_tensors": [
            {
                "tensor": {
                    "identity": {
                        "role": "master",
                        "name": "weight",
                        "component_path": [],
                    }
                },
                "archive": "master",
                "view": view,
            }
        ],
        "derived_tensors": {},
        "omitted_parameters": [],
    }
    with tarfile.open(path, "w") as result:
        for name, body in (
            ("manifest.json", json.dumps(manifest).encode()),
            ("value", np.asarray(data, dtype="<f4").tobytes()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            result.addfile(info, io.BytesIO(body))


def test_flat_storage_shards_keep_strides_offsets_and_check_replica_equality(tmp_path):
    ranks = tmp_path / "ranks"
    ranks.mkdir()
    # Transposed logical tensor over a storage slice which starts after a prefix.
    view = dict(
        shape=[3, 2],
        local_shape=[3, 2],
        origin=[0, 0],
        stride=[1, 3],
        storage_offset=1,
        storage_elements=7,
        region_offset=0,
        region_elements=4,
    )
    archive(ranks / "rank-00000.ssp", view, [-99, 1, 2, 3])
    other = {**view, "region_offset": 4, "region_elements": 3}
    archive(ranks / "rank-00001.ssp", other, [4, 5, 6])
    expected = {"weight": [[1, 4], [2, 5], [3, 6]]}
    np.testing.assert_array_equal(
        read_weights(tmp_path, ["weight"], expected)["weight"], expected["weight"]
    )
    archive(ranks / "rank-00002.ssp", deepcopy(other), [4, 5, 6])
    np.testing.assert_array_equal(
        read_weights(tmp_path, ["weight"], expected)["weight"], expected["weight"]
    )
    archive(ranks / "rank-00002.ssp", other, [4, 5, 7])
    with pytest.raises(AssertionError):
        read_weights(tmp_path, ["weight"], expected)
    (ranks / "rank-00002.ssp").unlink()
    (ranks / "rank-00001.ssp").unlink()
    with pytest.raises(AssertionError, match="incomplete logical weights"):
        read_weights(tmp_path, ["weight"], expected)
