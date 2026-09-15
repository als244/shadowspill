"""One plan described to the adapter: its objects, tasks, actions and layout."""

from __future__ import annotations

import ctypes


class PlanDescription(ctypes.Structure):
    """The pools and routes one plan is created against."""

    _fields_ = [
        ("plan_id", ctypes.c_uint64),
        ("execution_pool_id", ctypes.c_uint32),
        ("spill_pool_id", ctypes.c_uint32),
        ("fetch_route_id", ctypes.c_uint32),
        ("evict_route_id", ctypes.c_uint32),
    ]


class ObjectDescription(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("initial_version", ctypes.c_uint64),
        ("initial_pool_id", ctypes.c_uint32),
        ("retain_spill_copy", ctypes.c_uint8),
        ("initially_resident", ctypes.c_uint8),
    ]


class ObjectBinding(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("authoritative_version", ctypes.c_uint64),
        ("pointer", ctypes.c_void_p),
    ]


class ObjectUpdate(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("version_delta", ctypes.c_uint64),
    ]


class RuntimeAction(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
        ("trace_label", ctypes.c_char_p),
    ]


class TaskPublicationDescription(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
    ]


class TaskAllocationContractStep(ctypes.Structure):
    _fields_ = [
        ("allocation_ordinal", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("alignment_bytes", ctypes.c_uint64),
        ("operation", ctypes.c_uint8),
        ("required", ctypes.c_uint8),
    ]


class TaskDescription(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("trace_label", ctypes.c_char_p),
        ("input_object_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("input_count", ctypes.c_uint32),
        ("updates", ctypes.POINTER(ObjectUpdate)),
        ("update_count", ctypes.c_uint32),
        ("publications", ctypes.POINTER(TaskPublicationDescription)),
        ("publication_count", ctypes.c_uint32),
        ("actions", ctypes.POINTER(RuntimeAction)),
        ("action_count", ctypes.c_uint32),
        ("allocation_contract_steps", ctypes.POINTER(TaskAllocationContractStep)),
        ("allocation_contract_step_count", ctypes.c_uint32),
        ("enforce_allocation_contract", ctypes.c_uint8),
        ("maximum_requested_allocation_bytes", ctypes.c_uint64),
        ("maximum_charged_allocation_bytes", ctypes.c_uint64),
        ("live_requested_allocation_limit_bytes", ctypes.c_uint64),
        ("live_charged_allocation_limit_bytes", ctypes.c_uint64),
        ("dynamic_scratch_maximum_allocation_bytes", ctypes.c_uint64),
        ("dynamic_scratch_live_limit_bytes", ctypes.c_uint64),
    ]


class FixedPlacementDescription(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("ordinal", ctypes.c_uint64),
        ("object_id", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("alignment_bytes", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
    ]


class FixedDependencyDescription(ctypes.Structure):
    _fields_ = [
        ("predecessor_task_id", ctypes.c_uint64),
        ("predecessor_action_ordinal", ctypes.c_uint64),
        ("successor_task_id", ctypes.c_uint64),
        ("successor_ordinal", ctypes.c_uint64),
        ("successor_kind", ctypes.c_uint8),
    ]


class FixedLayoutDescription(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("slice_bytes", ctypes.c_uint64),
        ("placements", ctypes.POINTER(FixedPlacementDescription)),
        ("placement_count", ctypes.c_uint64),
        ("dependencies", ctypes.POINTER(FixedDependencyDescription)),
        ("dependency_count", ctypes.c_uint64),
    ]


class ObjectSnapshot(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("authoritative_version", ctypes.c_uint64),
        ("execution_version", ctypes.c_uint64),
        ("spill_version", ctypes.c_uint64),
        ("residency", ctypes.c_uint8),
        ("spill_current", ctypes.c_uint8),
        ("has_spill_lease", ctypes.c_uint8),
        ("execution_pointer", ctypes.c_void_p),
        ("spill_pointer", ctypes.c_void_p),
        ("retired_generation", ctypes.c_uint64),
        ("retired_execution_pointer", ctypes.c_void_p),
    ]


class ObjectLocationSnapshot(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("authoritative_version", ctypes.c_uint64),
        ("version", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("pool_id", ctypes.c_uint32),
        ("current", ctypes.c_uint8),
        ("has_lease", ctypes.c_uint8),
        ("pointer", ctypes.c_void_p),
    ]
