from dataclasses import replace
from types import SimpleNamespace

from shadowspill.communication import backend
from shadowspill.pytorch.distributed import _timing_identity as timing
from shadowspill.task.profiles import ProfileEnvironment, ProfileKey


def _binding(monkeypatch):
    options = SimpleNamespace(
        is_high_priority_stream=False,
        config=SimpleNamespace(max_ctas=32, min_ctas=1, nvls_ctas=16, cta_policy=0),
    )
    group = SimpleNamespace(
        _get_backend=lambda device: SimpleNamespace(options=options)
    )
    bound = SimpleNamespace(groups=SimpleNamespace(groups={"participants": group}))
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(backend.torch.cuda.nccl, "version", lambda: (2, 29, 7))
    return bound, options


def test_collective_settings_change_cached_profile_identity(monkeypatch):
    bound, options = _binding(monkeypatch)
    environment = ProfileEnvironment(
        "torch", "cuda", "gpu", (9, 0), "compiler", "provider"
    )

    def key():
        return ProfileKey(
            "task",
            replace(
                environment,
                communication_timing_digest=timing.communication_timing_digest(
                    bound, device_ordinal=0
                ),
            ),
        ).digest

    original = key()
    options.config.max_ctas = 8
    fewer_blocks = key()
    options.is_high_priority_stream = True
    priority = key()
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    algorithm = key()
    assert len({original, fewer_blocks, priority, algorithm}) == 4


def test_output_paths_do_not_change_timing_identity(monkeypatch):
    bound, _ = _binding(monkeypatch)
    original = timing.communication_timing_digest(bound, device_ordinal=0)
    monkeypatch.setenv("NCCL_DEBUG_FILE", "/different/log/%p")
    assert timing.communication_timing_digest(bound, device_ordinal=0) == original


def test_host_only_groups_do_not_initialize_cuda(monkeypatch):
    bound = SimpleNamespace(groups=SimpleNamespace(groups={"control": object()}))
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA touched")),
    )
    assert timing.communication_timing_identity(bound) == {
        "groups": {"control": {"backend": "gloo"}},
        "nccl_version": None,
        "environment": {},
    }


def test_config_content_changes_identity_without_changing_path(monkeypatch, tmp_path):
    bound, _ = _binding(monkeypatch)
    config = tmp_path / "nccl.conf"
    monkeypatch.setenv("NCCL_CONF_FILE", str(config))
    config.write_text("NCCL_ALGO=Ring\n")
    original = timing.communication_timing_digest(bound, device_ordinal=0)
    config.write_text("NCCL_ALGO=Tree\n")
    assert timing.communication_timing_digest(bound, device_ordinal=0) != original


def test_non_distributed_profile_identity_remains_unchanged():
    environment = ProfileEnvironment(
        "torch", "cuda", "gpu", (9, 0), "compiler", "provider"
    )
    assert "communication_timing_digest" not in environment.identity()
