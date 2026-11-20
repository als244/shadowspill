"""Admitted saved Plans, one worker, one outstanding asynchronous invocation."""

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Self

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.artifacts.tensors import export_selection
from shadowspill.communication.binding.ranks import RankAssignment
from shadowspill.memory import SpillPool
from shadowspill.parallelism import GroupRef
from shadowspill.pytorch.invocation import InvocationResult, ReusableCompletionEvent
from shadowspill.pytorch.numerics import validate_numerical_settings
from shadowspill.runtime.plan.annotations import TaskBoundaryAnnotations
from shadowspill.runtime.plan.objects import PlanObjects
from shadowspill.runtime.plan.tracing import TraceCapacities

from .dispatch import dispatch
from .distributed.checks import check_external_budget, check_invocation, coordinated
from .distributed.sequence import independent_plans
from .inputs import stage_inputs
from .paths import ExecutionPath
from .results import TrainingResult, collect_outputs
from .tensor_copies import reconstruct_replicas
from .tensors import TensorChunk, tensor_descriptors, visit_tensors
from .tracing import InvocationTrace
from .worker import create_paths


class Executor:
    """A worker that runs saved Plans and owns their persistent state.

    Constructing one admits ``plan`` and every Plan in ``plans`` on one
    device: it opens the device and spill pools, loads the task code and
    binds each Plan's storage, without filling it. Then `initialize` fills the
    persistent state, `run` invokes a Plan, and `close` releases everything;
    as a context manager it closes on exit. Construct it in a process whose
    device has no other allocator; it needs neither the model nor the build
    caches. Every worker of a distributed Plan constructs its Executor after
    the process-group rendezvous, and calls `initialize`, `run` and
    `discard_gradients` together with the others.

    Parameters
    ----------
    plan : Plan
        The default Plan, run when `run` names none.
    plans : mapping of str to Plan, optional
        Additional Plans by name, such as evaluation, backward or
        optimizer-only Plans, sharing persistent state with ``plan``. All
        must share the build target, parallelism and memory budgets; state is
        matched by name and layout. Admission checks that every Plan fits
        together.
    device : int, default 0
        Process-local GPU ordinal, the one the Plans' code was compiled for.
    rank_map : mapping of int to int, optional
        Launcher rank to logical rank, for every worker; required for a
        Plan with more than one rank.
    external_headroom_bytes : int, optional
        Device memory reserved outside the execution pool for what libraries
        allocate lazily, in bytes. By default the headroom the Plans
        were profiled with, or 512 MiB when unrecorded; more than that is
        refused.
    parameter_norms : bool, default False
        Collect each parameter's weight and gradient norms during runs of a
        step Plan, in `TrainingResult.parameter_metrics`.
    spill : SpillPool, optional
        The spill pool configuration. Its capacity must cover the Plans'
        spill requirement and fit their spill budget. By default a pinned host
        pool of the requirement plus 5%, up to the budget.
    reject_overbudget : bool, default False
        Refuse, rather than report, device memory use beyond the reservations
        outside the execution pool and process use beyond the budget. The
        pool's own bounds are enforced either way.

    Attributes
    ----------
    plan : Plan
        The default Plan.
    plans : mapping of str to Plan
        The additional Plans, by name, read-only.
    device : torch.device
        The execution device.
    runtime : Runtime
        The runtime that holds the pools.

    Raises
    ------
    ValueError
        If a Plan is simulation-only, was built for another device, toolchain
        or build, was measured for another spill backend or pool, cannot share
        state or memory with the others, ``external_headroom_bytes`` exceeds
        its reservation, or ``parameter_norms`` is set for a Plan that is not
        a step.
    TypeError
        If ``parameter_norms`` or ``reject_overbudget`` is not a bool, or
        ``spill`` is not a `SpillPool`.
    """

    def __init__(
        self,
        plan: Plan,
        *,
        plans: Mapping[str, Plan] | None = None,
        device: int = 0,
        rank_map: Mapping[int, int] | None = None,
        external_headroom_bytes: int | None = None,
        parameter_norms: bool = False,
        spill: SpillPool | None = None,
        reject_overbudget: bool = False,
    ) -> None:
        if type(reject_overbudget) is not bool:
            raise TypeError("reject_overbudget must be a bool")
        self.parameter_norms = parameter_norms
        world_size = (
            1
            if plan.program.parallelism is None
            else plan.program.parallelism.world_size
        )
        # Distributed assignment is validated collectively during startup, so
        # one invalid map cannot leave peers entering NCCL alone.
        self._assignment = (
            RankAssignment.resolve(1, rank_map) if world_size == 1 else None
        )
        self.plan = plan
        self.plans = MappingProxyType(dict(plans or {}))
        self.device = torch.device("cuda", device)
        self._pending: InvocationResult[Any] | None = None
        self._closed = False
        self._plan_released = False
        self._initialized = False
        self._failure: BaseException | None = None
        (
            self.runtime,
            self._paths,
            self._named_paths,
            self._streams,
            self._distributed,
            self._observations,
            self._providers,
        ) = create_paths(
            plan,
            device,
            external_headroom_bytes,
            self.plans,
            rank_map,
            parameter_norms,
            spill,
            reject_overbudget,
        )
        self._independent: frozenset[str] = frozenset()
        if self._distributed is not None:
            self._assignment = self._distributed.groups.assignment
            self._independent = independent_plans(self._named_paths)
        self._completion = ReusableCompletionEvent(
            self.runtime._runtime_handle, torch.device("cuda", device)
        )

    @property
    def spill_capacity_bytes(self) -> int:
        """Capacity of the worker's spill pool, in bytes; at most the spill budget."""
        return self.runtime.pools["spill"].capacity

    @property
    def logical_rank(self) -> int:
        """This worker's logical rank in the Plans, whatever its launcher rank."""
        return 0 if self._distributed is None else self._distributed.groups.logical_rank

    def initialize(
        self,
        fill_tensor: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
        restore: bool | Collection[str] = False,
        replicated_copies: Mapping[str, tuple[str, GroupRef]] | None = None,
    ) -> None:
        """Fill the persistent state, which every run needs first.

        Each persistent tensor this worker owns is filled in chunks:
        ``fill_tensor`` fills those without captured initial values, such as
        the model's parameters and buffers, and the rest take their captured
        values: gradients start at zero, masters copy their compute weights,
        and optimizer state, controls and constants take the values captured
        at build. Calling it again fills everything again; a failure leaves the
        executor uninitialized.

        Parameters
        ----------
        fill_tensor : callable
            Called as ``fill_tensor(chunk, buffer)`` with a `TensorChunk` and
            a one-dimensional buffer of its elements, which it fills before
            returning and keeps no reference to.
        chunk_bytes : int, default 64 << 20
            Largest chunk, in bytes.
        restore : bool or collection of str, default False
            True asks ``fill_tensor`` for every tensor, captured values
            included, as when restoring saved state; a collection asks for the
            tensors of those canonical names.
        replicated_copies : mapping of str to tuple, optional
            Replicated tensors to rebuild after filling, as destination name
            to ``(source name, GroupRef)``: the destination is assembled from
            the shards of the source the group's members own, in the group's
            order. ``fill_tensor`` may leave such a destination unfilled.
            Every worker passes it, even one with an empty mapping.

        Raises
        ------
        RuntimeError
            If the executor is closed or an earlier run failed.
        TypeError
            If ``restore`` is neither a bool nor a collection of names.
        ValueError
            If ``restore`` names a tensor `tensor_chunks` does not list, or
            ``chunk_bytes`` is not positive.
        """
        self.synchronize()
        self._initialized = False

        def fill() -> None:
            if not isinstance(restore, bool | Collection):
                raise TypeError(
                    "restore requires a boolean or collection of tensor names"
                )
            selected = (
                restore
                if isinstance(restore, bool)
                else export_selection(self.tensor_chunks(), restore)
            )
            assert selected is not None
            path = self._paths[0]
            visit_tensors(
                path.storage,
                fill_tensor,
                chunk_bytes,
                initialize=True,
                restore=selected,
            )
            path.controls.initialize(
                fill_tensor, chunk_bytes=chunk_bytes, restore=selected
            )
            for name, extra in self._named_paths.items():
                visit_tensors(
                    extra.storage,
                    fill_tensor,
                    chunk_bytes,
                    initialize=True,
                    restore=selected,
                    skip_shared=True,
                    prefix=f"plans.{name}.",
                )
                extra.controls.initialize(
                    fill_tensor,
                    chunk_bytes=chunk_bytes,
                    restore=selected,
                    skip_shared=True,
                    prefix=f"plans.{name}.",
                )

        coordinated(self._distributed, fill)
        if replicated_copies is not None:
            coordinated(
                self._distributed,
                lambda: reconstruct_replicas(
                    {
                        "": self._paths[0].storage,
                        **{
                            f"plans.{name}.": path.storage
                            for name, path in self._named_paths.items()
                        },
                    },
                    replicated_copies,
                    self._distributed,
                    chunk_bytes,
                ),
            )
        self._initialized = True

    def export_tensors(
        self,
        write_chunk: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
        names: Collection[str] | None = None,
    ) -> None:
        """Read the persistent state out in bounded chunks.

        An outstanding run is waited for first. Checkpoint formats belong to
        the caller.

        Parameters
        ----------
        write_chunk : callable
            Called as ``write_chunk(chunk, buffer)`` with a `TensorChunk` and
            a one-dimensional buffer of its elements, which it consumes before
            returning and keeps no reference to.
        chunk_bytes : int, default 64 << 20
            Largest chunk, in bytes.
        names : collection of str, optional
            Canonical names of the tensors to export; the others are not
            read. By default every tensor.

        Raises
        ------
        RuntimeError
            If the executor is closed, an earlier run failed, or it is not
            initialized.
        TypeError
            If ``names`` is a string or holds something else.
        ValueError
            If ``names`` holds a name `tensor_chunks` does not list, or
            ``chunk_bytes`` is not positive.
        """
        self.synchronize()
        if not self._initialized:
            raise RuntimeError("initialize persistent tensors before exporting them")
        selected = export_selection(self.tensor_chunks(), names)
        visit_tensors(
            self._paths[0].storage,
            write_chunk,
            chunk_bytes,
            initialize=False,
            names=selected,
        )
        self._paths[0].controls.export(
            write_chunk, chunk_bytes=chunk_bytes, names=selected
        )
        for name, path in self._named_paths.items():
            visit_tensors(
                path.storage,
                write_chunk,
                chunk_bytes,
                initialize=False,
                skip_shared=True,
                prefix=f"plans.{name}.",
                names=selected,
            )
            path.controls.export(
                write_chunk,
                chunk_bytes=chunk_bytes,
                skip_shared=True,
                prefix=f"plans.{name}.",
                names=selected,
            )

    def tensor_chunks(self) -> Iterator[TensorChunk]:
        """Describe each persistent tensor this worker owns, without reading it.

        This inventory is the training state a checkpoint binds to.

        Yields
        ------
        TensorChunk
            One chunk spanning each tensor's owned region.
        """
        yield from tensor_descriptors(self._paths[0].storage)
        yield from self._paths[0].controls.descriptors()
        for name, path in self._named_paths.items():
            yield from tensor_descriptors(
                path.storage, skip_shared=True, prefix=f"plans.{name}."
            )
            yield from path.controls.descriptors(
                skip_shared=True, prefix=f"plans.{name}."
            )

    def run(
        self,
        *,
        plan: str | None = None,
        args: Sequence[tuple[Any, ...]] | None = None,
        kwargs: Sequence[Mapping[str, Any]] | None = None,
        input_loader: Callable[[int], tuple[tuple[Any, ...], Mapping[str, Any]]]
        | None = None,
        hyperparams: Mapping[str, Any] | None = None,
        accumulate_gradients: bool = False,
        collect_parameter_norms: bool | None = None,
        profiler_annotations: bool = False,
        runtime_trace: bool = False,
        trace_capacity: TraceCapacities | None = None,
    ) -> InvocationResult[Any]:
        """Invoke a Plan once, without waiting for it to finish.

        It waits for the previous run, checks and stages the inputs, then
        dispatches every task. Every worker of a Plan that issues collectives
        calls it with the same ``plan``, ``accumulate_gradients`` and norm
        setting. The outputs belong to the caller, who releases any device
        outputs it holds before `close`.

        Parameters
        ----------
        plan : str, optional
            Name of an additional Plan. By default the default Plan.
        args : sequence of tuple, optional
            Positional arguments of each example, one tuple per example the
            Plan was built for, each of the built signature. An
            optimizer-only Plan takes none.
        kwargs : sequence of mapping, optional
            Keyword arguments of each example, matching ``args``.
        input_loader : callable, optional
            Called with each example's index and returning its
            ``(args, kwargs)``, instead of ``args`` and ``kwargs``.
        hyperparams : mapping of str to object, optional
            Values of captured runtime controls for this run only, such as
            ``"optimizer_group.0.lr"`` or the gradient scale; the others keep
            their captured defaults.
        accumulate_gradients : bool, default False
            Add this run's gradients to the stored ones instead of replacing
            them. Only for Plans of forward/backward Programs.
        collect_parameter_norms : bool, optional
            Whether this run collects parameter norms. By default it does for
            the default Plan when the executor collects them.
        profiler_annotations : bool, default False
            Open a named profiler range around each task.
        runtime_trace : bool, default False
            Record a runtime trace of this run, read from the result's
            ``diagnostics``.
        trace_capacity : TraceCapacities, optional
            Event capacities of that trace; only with ``runtime_trace``. By
            default sized from the Plan.

        Returns
        -------
        InvocationResult
            The run's handle. Its ``result()`` waits, and returns the model's
            outputs for a forward Plan or a `TrainingResult` for any other.

        Raises
        ------
        RuntimeError
            If the executor is closed, an earlier run failed, or it is not
            initialized.
        ValueError
            If ``plan`` names no admitted Plan, the examples differ in number
            or signature from the build, a control is unknown or misshapen,
            the Plan has no accumulating path, norms are asked for without
            ``parameter_norms`` or of an additional Plan, or
            ``trace_capacity`` comes without ``runtime_trace``.
        TypeError
            If ``profiler_annotations``, ``runtime_trace`` or
            ``collect_parameter_norms`` has another type.
        """
        self.synchronize()
        observe = (
            self.parameter_norms and plan is None
            if collect_parameter_norms is None
            else collect_parameter_norms
        )
        # Workers meet for every invocation of a Plan that issues collectives.
        distributed = None if plan in self._independent else self._distributed
        if distributed is not None:
            check_invocation(distributed, plan, accumulate_gradients, observe)

        def prepare() -> ExecutionPath:
            if type(profiler_annotations) is not bool:
                raise TypeError("profiler_annotations must be a bool")
            if type(runtime_trace) is not bool:
                raise TypeError("runtime_trace must be a bool")
            if trace_capacity is not None and (
                not runtime_trace
                or not isinstance(trace_capacity, TraceCapacities)
                or type(trace_capacity.events) is not int
                or type(trace_capacity.allocation_events) is not int
                or min(trace_capacity.events, trace_capacity.allocation_events) < 1
            ):
                raise ValueError(
                    "explicit trace capacity requires a positive traced call"
                )
            if type(observe) is not bool:
                raise TypeError("collect_parameter_norms must be a bool or None")
            if observe and (self._observations is None or plan is not None):
                raise ValueError(
                    "enable parameter_norms on Executor and select its training Plan"
                )
            if not self._initialized:
                raise RuntimeError("initialize persistent tensors before run")
            if plan is not None and plan not in self._named_paths:
                raise ValueError(f"unknown admitted Plan: {plan}")
            path = self._paths[0] if plan is None else self._named_paths[plan]
            if accumulate_gradients:
                if path.accumulation is None:
                    raise ValueError(
                        "this Plan has no gradient-accumulation invocation path"
                    )
                path = path.accumulation
            validate_numerical_settings(path.numerical_contracts)
            path.bridge.require_initial_residency(path.plan.schedule.initial_residency)
            path.controls.apply(hyperparams or {})
            stage_inputs(path.storage, args, kwargs, input_loader)
            return path

        path = coordinated(distributed, prepare)
        observations = self._observations if observe else None
        annotations = (
            TaskBoundaryAnnotations(path.bridge) if profiler_annotations else None
        )
        trace = InvocationTrace(path, trace_capacity) if runtime_trace else None
        try:
            if trace is not None:
                trace.start()
            if annotations is not None:
                annotations.set_enabled(True)
            parameter_metrics = {} if observations is None else observations.begin()
            with self._streams.invocation() as streams:
                for task in path.tasks:
                    with torch.cuda.stream(streams[task.task.resource]):
                        dispatch(
                            task,
                            path.storage,
                            path.controls,
                            observations,
                            annotations,
                            trace,
                        )
            payload = collect_outputs(path.storage, path.acquisition)
            if observations is not None:
                assert isinstance(payload, TrainingResult)
                payload = replace(payload, parameter_metrics=parameter_metrics)
            wait = self._completion.record()
            if trace is not None:
                trace.dispatched()

            def finish() -> None:
                wait()
                path.bridge.wait_until_idle()
                if trace is not None:
                    trace.finish()
                if annotations is not None:
                    annotations.set_enabled(False)
                if observations is not None:
                    observations.finish(parameter_metrics)
                if distributed is not None:
                    coordinated(
                        distributed, lambda: check_external_budget(self.runtime)
                    )

            self._pending = InvocationResult(
                payload,
                finish,
                on_resolved=lambda result: None,
                on_failure=self._failed,
                diagnostics=None if trace is None else lambda: trace.diagnostics,
            )
            return self._pending
        except BaseException as error:
            if trace is not None:
                try:
                    trace.cancel()
                except BaseException as cleanup:
                    error.add_note(f"Trace cleanup also failed: {cleanup}")
            self._failed(error)
            raise

    def _failed(self, error: BaseException) -> None:
        self._failure = error

    def discard_gradients(self) -> None:
        """Release the stored gradients without applying them.

        Each persistent gradient is released once; the Plans and the other
        state remain. The next backward must then start fresh gradients
        (``accumulate_gradients=False``); the caller keeps track of that. An
        outstanding run is waited for first.

        Raises
        ------
        RuntimeError
            If the executor is closed, an earlier run failed, or it is not
            initialized.
        ValueError
            If gradient storage overlaps other persistent state.
        """
        self.synchronize()
        if not self._initialized:
            raise RuntimeError("initialize persistent tensors before discarding")

        def release() -> None:
            gradients: dict[int, tuple[PlanObjects, str]] = {}
            other_state = set()
            for path in (*self._paths, *self._named_paths.values()):
                objects = path.bridge.objects
                for binding in program_state(
                    path.plan.program, self.logical_rank
                ).bindings:
                    alias = binding.alias_group_id
                    if alias is None or not objects.requires_storage(alias):
                        continue
                    identity = objects.runtime_object_id(alias)
                    if binding.identity.role == "gradient":
                        gradients.setdefault(identity, (objects, alias))
                    else:
                        other_state.add(identity)
            if gradients.keys() & other_state:
                raise ValueError("gradient storage overlaps other persistent state")
            for objects, alias in gradients.values():
                objects.release_value(alias)

        try:
            coordinated(self._distributed, release)
        except BaseException as error:
            self._failed(error)
            raise

    def synchronize(self) -> None:
        """Wait for the outstanding run, if any.

        Raises
        ------
        RuntimeError
            If the executor is closed or an earlier run failed.
        Exception
            If the outstanding run failed: its own failure, after which the
            executor is unusable.
        """
        if self._closed or self._plan_released:
            raise RuntimeError("Executor is closed")
        if self._failure is not None:
            raise RuntimeError(
                "Executor is unusable after an execution failure"
            ) from self._failure
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def close(self) -> None:
        """Release the admitted Plans and close the runtime.

        An outstanding run is waited for, unless an earlier run failed: then
        it is dropped, the device is drained, and every release step still
        runs. Calling it again after it has closed does nothing.

        Raises
        ------
        RuntimeConfigurationError
            If device outputs the caller still holds reference the pools. The
            runtime stays open, and close can be retried once they are
            released.
        Exception
            If the outstanding run failed: its failure, raised once everything
            is released.
        """
        if self._closed:
            return
        found: BaseException | None = None
        if self._failure is None and self._pending is not None:
            try:
                self._pending.result()
            except BaseException as error:
                found = error
        try:
            with ExitStack() as cleanup:
                # The runtime closes last, whatever the steps before it raise.
                cleanup.callback(self.runtime.close)
                if not self._plan_released:
                    self._plan_released = True
                    if self._distributed is not None:
                        cleanup.callback(self._distributed.close)
                    cleanup.callback(self._providers.close)
                    additional = [
                        part
                        for path in self._named_paths.values()
                        for part in (path, path.accumulation)
                        if part is not None
                    ]
                    for path in (*self._paths, *additional):
                        cleanup.callback(path.close, self.runtime)
                    if self._observations is not None:
                        cleanup.callback(self._observations.close)
                    cleanup.callback(self._completion.release)
                    cleanup.callback(self._settle)
        except BaseException as error:
            if self._failure is None:
                raise
            if not self.runtime._closed:
                # A refused runtime stays open for a retry, which releases no
                # plan twice. ExitStack drops the exception a caller's
                # `finally` is unwinding from the chain, so the failure is
                # named here.
                raise error from self._failure
        self._closed = True
        if found is not None:
            raise found

    def _settle(self) -> None:
        """Drop the outstanding invocation; after a failure, drain the device.

        A runtime close stops waiting for outstanding work once a failure is
        latched, and the device must be done with memory before it is released.
        """
        self._pending = None
        if self._failure is None:
            return
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as error:
            self._failure.add_note(
                f"Synchronizing the execution device during close also failed: {error}"
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Executor cleanup also failed: {cleanup}")
