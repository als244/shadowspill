"""Bounded task timing for correctness checks that do not judge throughput."""

from shadowspill.task.profiling import ProfilingOptions

# Exercise exact-task warmups, allocation probes and a fixed sample count.
# Tiny test kernels need no sustained-clock conditioning or duration floor.
CORRECTNESS_PROFILING = ProfilingOptions(
    conditioning_seconds=0,
    conditioning_wall_seconds=0,
    measurement_seconds=0,
    measurement_wall_seconds=0,
)
