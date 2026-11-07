"""The reference's kernel-selection artifact is a required numerical input."""

from types import SimpleNamespace

import pytest

from qualification.numerical import kernel_selection as kernels


def test_selection_is_bound_to_case_and_target(tmp_path, monkeypatch):
    request = SimpleNamespace(identity=lambda: "case-a")
    monkeypatch.setattr(kernels, "_target", lambda: {"device": "test"})
    monkeypatch.setattr(kernels, "_provider", lambda request: None)
    path = tmp_path / "kernel-selection.json"
    with kernels.capture(request) as selection:
        pass
    kernels.save(path, selection)
    restored = kernels.load(path, request)
    assert kernels.selection_digest(restored) == kernels.selection_digest(selection)
    with kernels.frozen(restored, request):
        pass
    with pytest.raises(RuntimeError, match="identity differs"):
        kernels.load(path, SimpleNamespace(identity=lambda: "case-b"))
    monkeypatch.setattr(kernels, "_target", lambda: {"device": "another"})
    with (
        pytest.raises(RuntimeError, match="target differs"),
        kernels.frozen(restored, request),
    ):
        pytest.fail("a different target entered numerical execution")


def test_missing_selection_is_not_an_implicit_autotune_request(tmp_path):
    with pytest.raises(RuntimeError, match="selection is missing"):
        kernels.load(
            tmp_path / "missing.json", SimpleNamespace(identity=lambda: "case")
        )
