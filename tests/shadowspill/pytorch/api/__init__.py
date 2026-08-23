"""Public end-to-end entry points.

These run first, and have to. Installing ShadowSpill's allocator needs a
process where CUDA has not been initialized, so a test that runs after
anything touching the device skips instead of running. pytest collects
directories before their sibling files and sorts by name, so this package
sorts ahead of every other one under ``pytorch``, and the ``test_00_``/
``test_01_`` prefixes order the two within it.

Do not move these, and do not add a package under ``pytorch`` that sorts
before ``api``.
"""
