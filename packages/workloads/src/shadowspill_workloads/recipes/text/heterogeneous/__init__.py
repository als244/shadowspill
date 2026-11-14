"""Data parallelism over machines whose GPUs differ: its own section of a config.

The normal runner assumes the ranks of a group run one Plan, so they must
hold the same hardware, since a Plan's executables and profiles belong to
the GPU that built them. A ``heterogeneous`` section lifts that: every node
builds, profiles and plans for its own GPUs in a run directory of its own,
nodes/<index>/, with its own budgets and its own microbatch geometry; the
ranks agree on what they exchange, checked when the workers open. A step is
shared in units of ``max_seq_len`` tokens: ``shares`` gives each rank's
proportion of the step's units, in order, and the step's units must divide by
their total; a rank concatenates as many units as its geometry holds into one
microbatch, every rank's loss its share of the step's mean. See README.md
beside this package.
"""
