"""Everything in the planner that speaks to libshadowspill.

These modules own the ctypes surface and the projections that feed it: an
indexed form of admission facts, the operations a schedule implies, the
lifetimes those operations give each lease, where the leases are placed, and
one call that evaluates a whole candidate portfolio.

Nothing here decides anything. The decisions are in ``pressurefit``.
"""
