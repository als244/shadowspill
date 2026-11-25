"""Bounded staging and immutable execution-lifetime settings."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._records import Record, Value, integer, mapping, number


@dataclass(frozen=True)
class Execution(Record):
    """Settings fixed for the lifetime of a run's workers.

    Parameters
    ----------
    seed : int, default 0
        Seed of each worker's Python, NumPy and framework random streams in a
        fresh run, at least zero. A restored checkpoint keeps its saved seed
        and state. Initial weights take their seed from `Weights`.
    max_staging_bytes : int, default 2 << 30
        Bound, in bytes (2 GiB by default), on host staging: the service's
        buffer for submitted inputs, shared by pending requests, and each
        worker's input, initialization and output copies. Creating the run
        counts the whole bound against host memory.
    max_pending_requests : int, default 2
        Maximum number of submitted requests not yet complete, at least one.
        A further submission waits for one to complete.
    idle_timeout_seconds : float, optional
        Idle time, in seconds, after which the run stops and releases its
        workers. Positive. It counts only while no request is queued or
        staging. Expiry saves no checkpoint and applies no pending
        gradients. None never expires.
    nsys : mapping, optional
        Nsight Systems capture on every worker, with keys ``start_step``
        (one-based optimizer update to start at, default 2), ``steps``
        (updates to capture, default 2), ``gpu_metrics`` (default True),
        ``gpu_metrics_frequency`` (in Hz, 10 to 200000, default 1000), and
        ``cuda_flush_interval_ms`` (default 60000). On CUDA 11+, a positive
        interval buffers CUDA activity between saves, and 0 saves on fill.
        None runs workers without a collector.
    reject_overbudget : bool, default False
        Whether a worker fails when its process holds more device memory
        than the budget, or more outside its execution pool than the
        headroom allows. False reports such overruns and continues. Pool
        bounds are enforced either way.
    """

    seed: int = 0
    max_staging_bytes: int = 2 << 30
    max_pending_requests: int = 2
    idle_timeout_seconds: float | None = None
    nsys: Mapping[str, Value] | None = None
    reject_overbudget: bool = False

    def __post_init__(self) -> None:
        integer(self.seed, "seed")
        integer(self.max_staging_bytes, "max_staging_bytes", minimum=1)
        integer(self.max_pending_requests, "max_pending_requests", minimum=1)
        if type(self.reject_overbudget) is not bool:
            raise TypeError("reject_overbudget must be a bool")
        if (
            self.idle_timeout_seconds is not None
            and number(self.idle_timeout_seconds, "idle_timeout_seconds") <= 0
        ):
            raise ValueError("idle_timeout_seconds must be positive or None")
        if self.nsys is not None:
            object.__setattr__(self, "nsys", mapping(self.nsys, "nsys"))
