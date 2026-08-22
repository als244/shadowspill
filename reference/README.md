# Readable algorithm references

This tree contains non-installed Python implementations used for explanation
and differential testing. Production imports never select these modules.

- `python/simulator/` is the readable event-scheduling oracle for the compiled
  simulator.
- `python/pressurefit/` contains the readable residency, action-emission, and
  PressureFit algorithms plus narrow compiled-component differential helpers.
- `python/admission/` is the readable oracle for physical admission: fixed
  offset placement today, and the schedule-to-lease replay once that moves to
  the planner library.

The wheel includes only `src/shadowspill`. Missing or ABI-incompatible compiled
planner and simulator libraries therefore fail immediately instead of changing
the production algorithm.
