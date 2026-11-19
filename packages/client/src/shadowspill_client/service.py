"""Public lightweight connection to an existing authenticated training service."""

from __future__ import annotations

import os
import ssl
from threading import Lock
from typing import Any, Self, TypeVar
from uuid import uuid4
from weakref import WeakSet

from ._async import CallFuture, ConnectionWork
from ._records import Record, integer
from .artifacts import ArtifactHandle, ArtifactTransfers
from .buffers import BufferBudget
from .captures import NsightCapture, get_nsys
from .checkpoints import CheckpointHandle
from .config import Resources
from .creation import TrainingFuture, create_training_client, get_training_client
from .downloads import CHECKPOINT_SCRATCH_BYTES, FILE_SCRATCH_BYTES
from .metrics import MetricQueries
from .plans import PlansHandle, get_plans
from .preparation import (
    PlanningFuture,
    PreparationFuture,
    get_planning,
    get_preparation,
    plan_training,
    prepare_training,
)
from .protocol.checkpoints import CHECKPOINT_MEDIA_TYPE, WEIGHTS_MEDIA_TYPE
from .protocol.codec import ProtocolError, encode
from .protocol.hardware import HardwareInventory
from .protocol.metrics import MetricPage
from .protocol.preview import TrainingDescription, TrainingPreview
from .protocol.requests import ArtifactRef, identifier
from .protocol.service import ServiceInfo
from .protocol.setup import TrainingSetup
from .protocol.training import ResolvedTraining
from .training import TrainingClient
from .transport.http import HTTP_SCRATCH_BYTES, HTTPTransport
from .transport.ssh import SSH, SSH_BYTES, SSHTunnel
from .uploads import UPLOAD_SCRATCH_BYTES
from .weights import WeightsHandle

R = TypeVar("R", bound=Record)


class ServiceClient:
    """Connection to a training service.

    Construction blocks until the service answers. After that, each call that
    contacts the service, on the connection or on a handle from it, returns a
    future at once and does its work on a background thread; while
    ``max_pending_requests`` such calls are outstanding, the next one blocks.
    No exchange is retried automatically. Closing the connection ends its
    local work only; runs and other accepted work continue on the service.
    The connection is a context manager that closes on exit.

    Parameters
    ----------
    base_url : str, optional
        Service URL: ``https://`` or loopback ``http://``, without credentials,
        query or fragment. Give exactly one of ``base_url`` and ``ssh``.
    ssh : SSH, optional
        Reach the service through an SSH forward that this connection starts
        and closes.
    token : str, optional
        Bearer credential, 32 to 512 visible ASCII characters. By default, the
        ``SHADOWSPILL_TOKEN`` environment variable.
    session_id : str, optional
        Session the connection works in: runs it creates belong to it, and
        lookups by run ID search it. Reconnect with the same ID to reach
        earlier runs. By default, a new random ID; the attribute
        ``session_id`` holds the ID in use.
    timeout : float, default 120
        Socket timeout of each HTTP exchange, in seconds.
    max_connections : int, default 4
        Number of calls worked on at once, each with its own HTTP exchange.
    max_pending_requests : int, default 16
        Number of calls that may be queued or in progress at once.
    ssl_context : ssl.SSLContext, optional
        TLS settings for an HTTPS ``base_url``. By default, Python's default
        certificate verification.

    Attributes
    ----------
    info : ServiceInfo
        The service's identity, read when connecting.
    artifacts : ArtifactTransfers
        Uploads into the service's artifact storage.

    Raises
    ------
    ValueError
        Both or neither of ``base_url`` and ``ssh`` are given, or an argument,
        such as the token, is invalid.
    RemoteError
        The service refuses the connection, for example for a wrong token.
    TransportError
        The service cannot be reached.
    RuntimeError
        OpenSSH cannot start the forward.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        ssh: SSH | None = None,
        token: str | None = None,
        session_id: str | None = None,
        timeout: float = 120,
        max_connections: int = 4,
        max_pending_requests: int = 16,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if (base_url is None) == (ssh is None):
            raise ValueError("provide exactly one of base_url or ssh")
        integer(max_connections, "max_connections", minimum=1)
        self.session_id = identifier(session_id or uuid4().hex, "session ID")
        self._work = ConnectionWork(max_pending_requests, max_connections)
        self._handles: WeakSet[TrainingClient] = WeakSet()
        self._handle_lock = Lock()
        self._memory = BufferBudget(
            max_connections
            * (
                HTTP_SCRATCH_BYTES
                + max(FILE_SCRATCH_BYTES, UPLOAD_SCRATCH_BYTES)
                + CHECKPOINT_SCRATCH_BYTES
            )
            + SSH_BYTES,
            3 * max_connections + 1,
        )
        self._tunnel = None if ssh is None else SSHTunnel(ssh, memory=self._memory)
        try:
            if self._tunnel is not None:
                base_url = self._tunnel.start()
            assert base_url is not None
            self._transport = HTTPTransport(
                base_url,
                token=token
                if token is not None
                else os.environ.get("SHADOWSPILL_TOKEN", ""),
                memory=self._memory,
                timeout=timeout,
                ssl_context=ssl_context,
            )
            self.info = self._request("GET", "/v1/info", ServiceInfo)
            self.artifacts = ArtifactTransfers(self)
        except BaseException:
            self.close()
            raise

    def _request(
        self, method: str, path: str, kind: type[R], value: Record | None = None
    ) -> R:
        self._work.check()
        result = self._transport.request(method, path, value).value
        if not isinstance(result, kind):
            raise ProtocolError(f"service response requires {kind.__name__}")
        return result

    @staticmethod
    def _setup(
        config: ResolvedTraining | None, setup: dict[str, Any]
    ) -> TrainingSetup | ResolvedTraining:
        if config is not None:
            if setup or not isinstance(config, ResolvedTraining):
                raise ValueError(
                    "config requires resolved settings without overlapping keywords"
                )
            settings: TrainingSetup | ResolvedTraining = config
        else:
            settings = TrainingSetup(**setup)
        # Bound the control tree before retaining it in the connection queue.
        encode(settings)
        return settings

    def list_hardware(self) -> CallFuture[HardwareInventory]:
        """Read the hardware the service can assign, without reserving any.

        Returns
        -------
        CallFuture of HardwareInventory
            The service's nodes and devices, each device marked available
            unless the service has claimed it.
        """
        return self._work.submit(
            lambda: self._request("GET", "/v1/hardware", HardwareInventory)
        )

    def _describe(self, request: TrainingDescription) -> TrainingPreview:
        return self._request("POST", "/v1/training/describe", TrainingPreview, request)

    def describe_training(
        self,
        *,
        config: ResolvedTraining | None = None,
        resources: Resources | None = None,
        **setup: Any,
    ) -> CallFuture[TrainingPreview]:
        """Resolve a run's settings and propose its workers, without starting work.

        The service fills defaults from its installed definitions and selects
        devices. It builds nothing and reserves no hardware; busy devices can
        appear among the proposed workers.

        Parameters
        ----------
        config : ResolvedTraining, optional
            Settings resolved earlier, such as the ``config`` of a
            `TrainingPreview`, to check again. Cannot be combined with
            ``setup``.
        resources : Resources, optional
            Hardware selection. By default, the service chooses.
        **setup
            The run's settings: the keywords ``model`` through ``prepare`` of
            `create_training_client`.

        Returns
        -------
        CallFuture of TrainingPreview
            The resolved settings, preparation estimates and proposed workers.
            It fails with `RemoteError` if the service rejects the settings.

        Raises
        ------
        TypeError
            A required setting is missing, a keyword is unknown, or a value has
            the wrong type.
        ValueError
            ``config`` is combined with ``setup``, or a value is invalid.
        """
        request = TrainingDescription.request(self._setup(config, setup), resources)
        encode(request)
        return self._work.submit(lambda: self._describe(request))

    def artifact(self, reference: ArtifactRef) -> ArtifactHandle:
        """Attach an artifact reference to this connection.

        The call does not contact the service.

        Parameters
        ----------
        reference : ArtifactRef
            The stored artifact.

        Returns
        -------
        ArtifactHandle
            A `CheckpointHandle` for a checkpoint, a `WeightsHandle` for a
            weights export, otherwise an `ArtifactHandle`.

        Raises
        ------
        TypeError
            ``reference`` is not an `ArtifactRef`.
        """
        if not isinstance(reference, ArtifactRef):
            raise TypeError("artifact requires an immutable server reference")
        if reference.media_type == CHECKPOINT_MEDIA_TYPE:
            return CheckpointHandle(reference, self)
        if reference.media_type == WEIGHTS_MEDIA_TYPE:
            return WeightsHandle(reference, self)
        return ArtifactHandle(reference, self)

    def create_training_client(self, **setup: Any) -> TrainingFuture:
        """Create a training run and attach to it once it is ready.

        The service builds, measures and plans the run unless ``plans`` is
        given, then starts its workers and initializes its state or loads
        ``checkpoint``.

        Parameters
        ----------
        **setup
            The run's settings:

            model : `Model`
                The installed model, its configuration and initial weights.
            objective : `Objective`
                What training minimizes.
            batch : `BatchConfig` or sequence of `BatchConfig`
                Limits of one submitted batch; a sequence prepares each.
            memory : `Memory`
                Device and spill budgets of each worker.
            optimizer : `Optimizer`, optional
                Update rule and its state.
            parallelism : dict, optional
                A declaration such as
                `~shadowspill_client.parallelism.data_parallel` returns. By
                default, the model definition's, or a single worker.
            execution : `Execution`, optional
                Run-time settings. By default, ``Execution()``.
            operations : mapping of str to sequence, optional
                Compute operations to prepare (``"step"``,
                ``"forward_backward"``, ``"forward"``), each mapped to output
                set names or `Operation` records. By default, ``"step"`` with
                the objective's default outputs.
            prepare : `Preparation`, optional
                Build, profiling and planning choices. By default, the
                preparation preset's.

            Or, in place of those settings:

            config : `ResolvedTraining`, optional
                Settings resolved earlier, such as the ``config`` of a
                `TrainingPreview`.

            And options of this creation:

            plans : `ArtifactRef` or `ArtifactHandle`, optional
                Saved Plans to run, skipping build, profiling and planning.
            checkpoint : `ArtifactRef` or `ArtifactHandle`, optional
                Checkpoint to start from. With ``plans`` and no settings, the
                run uses the settings saved with them.
            resources : `Resources`, optional
                Hardware selection, such as the ``resources`` of a
                `TrainingPreview`. By default, the service chooses.
            wait_for_resources : bool, default False
                Wait for occupied devices instead of failing.
            name : str, optional
                Run name, at most 256 characters.
            labels : mapping of str to str, optional
                At most 16 run labels.
            progress : bool, default False
                Print the run's preparation and startup logs to standard
                output while waiting.

        Returns
        -------
        TrainingFuture
            Its ``run_id`` is set at once. It completes with a
            `TrainingClient` attached as the run's writer, and fails with
            `CreationError` if the run ends before it is ready.

        Raises
        ------
        TypeError
            A required setting is missing, a keyword is unknown, or a value has
            the wrong type.
        ValueError
            ``config`` is combined with other settings, or a value is invalid.
        """
        return create_training_client(self, **setup)

    def prepare_training(self, **setup: Any) -> PreparationFuture:
        """Build and measure a run's cases, without planning or training.

        The service builds and profiles every case and measures communication,
        then releases the devices it used.

        Parameters
        ----------
        **setup
            The keywords of `create_training_client` other than ``plans`` and
            ``checkpoint``. The model needs no initial weights.

        Returns
        -------
        PreparationFuture
            Its ``run_id`` is set at once. It completes with a
            `PreparationHandle` when measurement finishes, and fails with
            `CreationError` if the job fails or stops.

        Raises
        ------
        TypeError
            A required setting is missing, a keyword is unknown, or a value has
            the wrong type.
        ValueError
            ``config`` is combined with other settings, or a value is invalid.
        """
        return prepare_training(self, **setup)

    def plan_training(self, **setup: Any) -> PlanningFuture:
        """Build, measure and plan a run, without starting training workers.

        The service releases the devices it used once planning finishes. Pass
        the resulting Plans as ``plans`` to `create_training_client` to start
        the run without repeating this work.

        Parameters
        ----------
        **setup
            The keywords of `create_training_client` other than ``plans`` and
            ``checkpoint``.

        Returns
        -------
        PlanningFuture
            Its ``run_id`` is set at once. It completes with a `PlansHandle`
            when planning finishes, and fails with `CreationError` if the job
            fails or stops.

        Raises
        ------
        TypeError
            A required setting is missing, a keyword is unknown, or a value has
            the wrong type.
        ValueError
            ``config`` is combined with other settings, or a value is invalid.
        """
        return plan_training(self, **setup)

    def get_planning(self, run_id: str, *, progress: bool = False) -> PlanningFuture:
        """Rejoin a planning job of this session, without repeating it.

        Parameters
        ----------
        run_id : str
            The job's run ID, the ``run_id`` of its `PlanningFuture`.
        progress : bool, default False
            Print the job's logs to standard output while waiting.

        Returns
        -------
        PlanningFuture
            Completes with a `PlansHandle` when planning finishes, and fails
            with `CreationError` if the job failed or stopped.

        Raises
        ------
        ValueError
            ``run_id`` is not a valid run ID.
        """
        return get_planning(self, run_id, progress=progress)

    def get_preparation(
        self, run_id: str, *, progress: bool = False
    ) -> PreparationFuture:
        """Rejoin a measurement job of this session, without repeating it.

        Parameters
        ----------
        run_id : str
            The job's run ID, the ``run_id`` of its `PreparationFuture`.
        progress : bool, default False
            Print the job's logs to standard output while waiting.

        Returns
        -------
        PreparationFuture
            Completes with a `PreparationHandle` when measurement finishes, and
            fails with `CreationError` if the job failed or stopped.

        Raises
        ------
        ValueError
            ``run_id`` is not a valid run ID.
        """
        return get_preparation(self, run_id, progress=progress)

    def get_metrics(
        self,
        run_id: str,
        *,
        after: int = 0,
        through: int | None = None,
        limit: int = 100,
    ) -> CallFuture[MetricPage]:
        """Read one page of a run's recorded metrics.

        This works without a training client, including after the run's
        workers or the service have stopped.

        Parameters
        ----------
        run_id : str
            The run, in this connection's session.
        after : int, default 0
            Journal position after which the page starts.
        through : int, optional
            Last journal position the page may include. Pin it to read one
            fixed snapshot over several pages. By default, the journal's
            current end.
        limit : int, default 100
            Maximum number of records, 1 to 128.

        Returns
        -------
        CallFuture of MetricPage
            The page, in completion order.

        Raises
        ------
        ValueError
            ``run_id`` is invalid, ``limit`` is out of range, or ``through``
            is before ``after``.
        """
        return MetricQueries(self, self.session_id, run_id).query(
            after=after, through=through, limit=limit
        )

    def get_training_client(
        self, run_id: str, *, read_only: bool = False, progress: bool = False
    ) -> TrainingFuture:
        """Reattach to a training run of this session.

        Parameters
        ----------
        run_id : str
            The run's ID, the ``run_id`` of its `TrainingFuture`.
        read_only : bool, default False
            Attach for inspection only, at once and in any run state. A
            read-only client cannot submit operations and leaves the current
            writer in place.
        progress : bool, default False
            Print the run's preparation and startup logs to standard output
            while waiting.

        Returns
        -------
        TrainingFuture
            Completes with a `TrainingClient`. Unless ``read_only``, it first
            waits for the run to be ready, then attaches as the run's writer,
            which fences any earlier writer; it fails with `CreationError` if
            the run failed, stopped or closed.

        Raises
        ------
        ValueError
            ``run_id`` is not a valid run ID.
        """
        return get_training_client(self, run_id, read_only=read_only, progress=progress)

    def get_plans(self, run_id: str, *, name: str) -> CallFuture[PlansHandle]:
        """Look up Plans saved under a name in a run of this session.

        This works after the run's workers have closed.

        Parameters
        ----------
        run_id : str
            The run the Plans were saved in.
        name : str
            The name given to `PlansHandle.save`.

        Returns
        -------
        CallFuture of PlansHandle
            The saved Plans. It fails with `RemoteError` if the run has no
            Plans of that name.

        Raises
        ------
        ValueError
            ``run_id`` or ``name`` is invalid.
        """
        return get_plans(self, run_id, name=name)

    def get_nsys(self, run_id: str, *, rank: int = 0) -> CallFuture[NsightCapture]:
        """Read one worker's Nsight Systems capture of a run of this session.

        This works after the run's workers have closed.

        Parameters
        ----------
        run_id : str
            The run.
        rank : int, default 0
            The worker's rank.

        Returns
        -------
        CallFuture of NsightCapture
            The capture's status and, when present, its report.

        Raises
        ------
        ValueError
            ``run_id`` or ``rank`` is invalid.
        """
        return get_nsys(self, run_id, rank=rank)

    def _retain_handle(self, handle: TrainingClient) -> None:
        with self._handle_lock:
            self._work.check()
            self._handles.add(handle)

    def close(self) -> None:
        """Close the connection, and the SSH forward it started.

        Blocks until running calls end. Calls still queued, and calls polling
        the service, fail with `ConnectionError`; later calls that contact the
        service raise it, including on training clients from this connection.
        Work the service accepted continues, and runs stay open. Closing again
        does nothing.

        Raises
        ------
        RuntimeError
            OpenSSH cannot cancel the forward on a shared control connection.
        """
        self._work.stop()
        with self._handle_lock:
            for handle in self._handles:
                handle._disconnect()
        self._work.close()
        if self._tunnel is not None:
            self._tunnel.close()
        self._memory.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
