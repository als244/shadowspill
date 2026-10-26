"""Explicit physical-cap deductions and hypothetical host-copy rates."""

import argparse

from shadowspill.planner.program_inputs import TransferBandwidths


def execution_budget(dump, *, execution_gib=None, physical_gib=None):
    """Resolve a physical cap with the capture's measured runtime carve-out."""
    if execution_gib is not None and physical_gib is not None:
        raise ValueError("execution and physical budget overrides are alternatives")
    if execution_gib is not None:
        execution = int(execution_gib * 2**30)
        physical = None
    elif physical_gib is not None:
        physical = int(physical_gib * 2**30)
        fixed = dump["physical_budget_bytes"] - dump["execution_budget_bytes"]
        execution = physical - fixed
    else:
        execution = dump["execution_budget_bytes"]
        physical = dump.get("physical_budget_bytes")
    if execution <= 0:
        raise ValueError("budget leaves no execution capacity after runtime overhead")
    return execution, physical


def transfer_bandwidths(value):
    """Parse decimal GB/s without changing the captured transfer latencies."""
    try:
        rates = tuple(int(float(part.strip()) * 1e9) for part in value.split(","))
        if len(rates) == 2:
            rates = (rates[0], rates[0], rates[1], rates[1])
        if len(rates) != 4:
            raise ValueError("expected two or four rates")
        return TransferBandwidths(
            *rates,
            provenance=f"tools.pressurefit_trace plan --transfer-bandwidths {value}",
        )
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(
            "use positive finite GB/s rates: FETCH,EVICT or "
            "FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT"
        ) from error
