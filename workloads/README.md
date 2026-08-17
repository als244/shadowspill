# Workloads

This directory contains model and data definitions used by benchmarking and
qualification. They are clients of the public ShadowSpill API and are not
installed as part of the `shadowspill` package.

Core planning, runtime, and lowering code must never import `workloads`.
