"""Untimed per-phase memory measurements, separate from runtime samples."""

from __future__ import annotations

import threading


class DeviceMemorySampler:
    """Sample total device usage, including allocations outside PyTorch.

    NVML polling can miss short-lived peaks. Preserve its interval/sample count
    and describe the result as a sampled lower bound, not an exact high-water
    mark. Exact PyTorch allocator peaks are reported independently.
    """

    def __init__(self, torch, interval_seconds=0.001):
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread = None
        self.peak = 0
        self.samples = 0
        self.error = None
        self.nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        except Exception as error:
            self.error = repr(error)

    def _sample(self):
        try:
            used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
            self.peak = max(self.peak, used)
            self.samples += 1
        except Exception as error:
            self.error = repr(error)
            self.stop_event.set()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        if self.error is None:
            self._sample()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self._sample()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
        return {
            "sampled_device_peak_bytes": self.peak if self.samples else None,
            "device_samples": self.samples,
            "device_sampling_interval_seconds": self.interval,
            "device_peak_is_sampled_lower_bound": True,
            "device_sampling_error": self.error,
        }


def snapshot(torch):
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "device_used_bytes": total - free,
        "device_total_bytes": total,
    }


def measure_phase(torch, operation):
    torch.cuda.synchronize()
    before = snapshot(torch)
    torch.cuda.reset_peak_memory_stats()
    sampler = DeviceMemorySampler(torch)
    sampler.start()
    try:
        result = operation()
        torch.cuda.synchronize()
    finally:
        sampled = sampler.finish()
    return result, {
        "before": before,
        "after": snapshot(torch),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        **sampled,
    }


def measure(torch, implementation, inputs, gradient):
    """One warmed forward/backward pass; no memory instrumentation is timed."""
    implementation.zero_grad()
    inputs.grad = None
    output, forward = measure_phase(torch, lambda: implementation.forward(inputs))
    _, backward = measure_phase(
        torch, lambda: implementation.backward(output, gradient)
    )
    return output, {"forward": forward, "backward": backward}
