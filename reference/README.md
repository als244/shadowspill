# Readable algorithm references

This tree contains non-installed Python implementations used for explanation
and differential testing. Production imports never select these modules.

- `python/simulator/` is the readable event-scheduling oracle for the compiled
  simulator.
- `python/pressurefit/` contains the readable residency, action-emission, and
  PressureFit algorithms plus narrow compiled-component differential helpers.
- `python/admission/` holds the readable oracles for physical admission: the
  fixed-offset placement and the schedule-to-lease replay, both of which the
  planner library now implements. They define what the compiled versions must
  reproduce and are the baselines their speedups are measured against.

The wheel includes only `src/shadowspill`. Missing or ABI-incompatible compiled
planner and simulator libraries therefore fail immediately instead of changing
the production algorithm.
