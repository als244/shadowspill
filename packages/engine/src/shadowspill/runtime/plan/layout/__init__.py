"""Project certified layouts and completion dependencies into runtime records."""

from .projection import DynamicTaskAllocationPolicy, project_runtime_fixed_layout

__all__ = ["DynamicTaskAllocationPolicy", "project_runtime_fixed_layout"]
