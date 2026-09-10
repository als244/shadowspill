# Events

The runtime's synchronization objects live in `csrc/src/runtime/sync/`. They
are built on the backend's event and stream calls and own every event the
runtime uses, so a backend never pools, seals, or measures anything itself.

## Event leases

An `EventLease` is a runtime record that owns one backend event and tracks
the completion it stands for: a generation, the object or allocation whose
release it protects, a reference count, and whether the backend has reported
it complete. Leases come from an `EventPool`: a free list of records whose
backend events are created once and kept across leases, so acquiring and
releasing a lease is a list operation and never a driver call.

## Reserving and sealing

Cold plan adoption calls `shadowspill_runtime_reserve_event_leases()`, which
grows the pool to the plan's requirement and creates the backend events up
front, then seals it. Before sealing, a request the free list cannot serve is
given a record outside the pool that creates its own event. After sealing that
path is gone: a request that finds no free lease is refused and counted, so no
steady-state step ever pays a driver call the plan did not reserve.

The runtime statistics expose the pool's capacity, current and peak use,
`event_lease_sealed`, `event_lease_growth_rejections`, and
`event_lease_driver_creates` -- every backend event the pool has created, which
after sealing only a further reservation can raise. A nonzero rejection count
is the signal worth watching: it means a plan asked for more completions than
it reserved.

## Completion tracking

Each stream the runtime records completions on has a FIFO of leases. The
worker queries only the head of each FIFO with `query_event`, follows an
already-complete head immediately, and drains immediately completed
successors, so the number of driver queries stays close to the number of
completions. A completed lease lets its owner, an object residency, a task
allocation, or a retiring range, move on; the lease itself returns to the
pool with its event.

## The timing pool

Traced transfers are measured with timing events, the kind that carries a
device timestamp. They come from a second pool the runtime keeps apart from
the dependency pool, reserved when a trace is prepared with a fixed number of
events per lane, and sealed like the first. A stream interval takes two
leases from it, records the first before a copy and the second after, and
reads both against the step's origin event with `elapsed_nanoseconds`; a
pool that runs out leaves later intervals unmeasured, never a transfer
failed. An untraced step never touches this pool. How the intervals become a
timeline is in [timelines](timelines.md).

## Idle wake-up

A dedicated condition variable lets callers wait for the worker at explicit
boundaries (`shadowspill_runtime_wait_idle()`) without polling; the worker
signals it when it has nothing queued and nothing pending.
