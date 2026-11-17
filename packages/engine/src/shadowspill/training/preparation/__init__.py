"""Resolve training setup and compose the existing engine phases.

``setup`` resolves installed definitions. ``configuration`` validates requested
layouts and policies; ``capture`` binds them to the build frontend. ``phases``
launches build, profile, communication and planning jobs. ``planning`` selects
compatible Plan families; ``artifacts`` saves and validates their evidence.
``offline`` exposes measured inputs for local planning and hypothetical sweeps.
Worker initialization and training commands belong to their own packages.
"""
