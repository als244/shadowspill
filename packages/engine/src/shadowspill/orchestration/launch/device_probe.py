"""Short-lived CUDA/NVML observation child; never imported to probe a controller."""

import sys

from shadowspill.profiling.pool_identity import device_uuid, visible_device_count

from .device_inventory import CODEC, MAX_DEVICES, DeviceInventory, DeviceObservation


def observe() -> DeviceInventory:
    import pynvml  # type: ignore[import-untyped]

    count = visible_device_count()
    if count > MAX_DEVICES:
        raise ValueError("visible CUDA device count exceeds the inventory bound")
    pynvml.nvmlInit()
    try:
        devices = []
        for ordinal in range(count):
            identifier = device_uuid(ordinal)
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(identifier)
            except pynvml.NVMLError_NotFound:
                # CUDA's UUID bytes identify a MIG compute instance as well.
                # Use its canonical modern prefix, never the parent GPU/index.
                identifier = "MIG-" + identifier.removeprefix("GPU-")
                handle = pynvml.nvmlDeviceGetHandleByUUID(identifier)
            actual = pynvml.nvmlDeviceGetUUID(handle)
            if actual != identifier:
                raise ValueError("NVML identity differs from the visible CUDA UUID")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                DeviceObservation(
                    actual, pynvml.nvmlDeviceGetName(handle), memory.total, memory.free
                )
            )
        return DeviceInventory(pynvml.nvmlSystemGetDriverVersion(), devices)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    sys.stdout.buffer.write(CODEC.encode(observe()))
    sys.stdout.buffer.flush()
