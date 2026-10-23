"""Explicit SM90 candidate policy for reproducible SonicMoE experiments."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

_active_mode = None
_destination = None


def expanded(original, epilogue=None, tune_coop=True, *, include_k=False):
    """Keep current candidates and broaden only Hopper tile/cluster geometry.

    Quack still applies its normal gather, varlen, epilogue and sink checks.
    This is a candidate space, not a claim every kernel compiles or is faster.
    """
    from quack.gemm_config import GemmConfig

    candidates = list(original(epilogue, tune_coop))
    seen = set(candidates)

    def add(candidate):
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    # Stock candidates omit the single-CTA cluster, including known measured
    # choices from mlops.expert_parallel.quack.experts.policy.
    for candidate in tuple(candidates):
        add(replace(candidate, cluster_m=1, cluster_n=1))
    tiles = [(64, n, True) for n in (32, 64, 96, 128, 192, 256)]
    tiles += [(128, n, True) for n in (32, 64)]
    if tune_coop:
        tiles += [(64, n, False) for n in (32, 64, 96, 128, 192, 256)]
        tiles += [(128, n, False) for n in (32, 64, 96, 128, 160, 192)]
    swaps = (False,) if epilogue in ("gated", "lse") else (False, True)
    for m, n, pingpong in tiles:
        for cm, cn in ((1, 1), (1, 2), (2, 1)):
            for swap in swaps:
                add(
                    GemmConfig(
                        tile_m=m,
                        tile_n=n,
                        pingpong=pingpong,
                        cluster_m=cm,
                        cluster_n=cn,
                        swap_ab=swap,
                        device_capacity=9,
                        is_dynamic_persistent=False,
                        use_tma_gather=False,
                    )
                )
    if include_k:
        anchors = {
            (64, 128, True),
            (128, 96, True),
            (128, 128, False),
            (256, 128, False),
        }
        for candidate in tuple(candidates):
            if (candidate.tile_m, candidate.tile_n, candidate.pingpong) in anchors:
                for k in (32, 64, 128, 256):
                    add(replace(candidate, tile_k=k))
    return candidates


def install(mode, destination):
    global _active_mode, _destination
    _destination = Path(destination)
    if _active_mode is not None:
        if _active_mode != mode:
            raise RuntimeError("candidate policy cannot change within shared workers")
        return
    if mode not in {"baseline", "expanded", "expanded-k"}:
        raise ValueError(f"unknown candidate policy {mode}")
    _active_mode = mode
    from mlops.quack_runtime import initialize

    initialize()
    import quack.gemm_config as gc
    from quack.autotuner import AutotuneConfig
    from quack.gemm_interface import gemm_tuned

    original = gc._get_sm90_configs

    def generator(epilogue=None, tune_coop=True):
        return expanded(original, epilogue, tune_coop, include_k=mode == "expanded-k")

    if gemm_tuned.cache:
        raise RuntimeError("candidate policy must be installed before any GEMM use")
    if mode != "baseline":
        gc._get_sm90_configs = generator
        gemm_tuned.configs = [AutotuneConfig(config=c) for c in gc.get_all_configs()]
    _install_audit(mode)


def _install_audit(mode):
    from quack.autotuner import Autotuner

    original = Autotuner.check_disk_cache
    rank = int(os.environ.get("RANK", "0"))

    def checked(self, key, configs, bench):
        original(self, key, configs, bench)
        if rank == 0:
            record = {
                "policy": mode,
                "kernel": self.fn.__name__,
                "key": str(key),
                "candidate_count": len(configs),
                "winner": asdict(self.cache[key].kwargs["config"]),
                "bench_seconds": self.bench_time,
                "timings": [
                    {"config": asdict(c.kwargs["config"]), "milliseconds": timings}
                    for c, timings in self.configs_timings.items()
                ],
            }
            with (_destination / "candidate-audit.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    Autotuner.check_disk_cache = checked
