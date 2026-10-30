# Heterogeneous data parallelism

Ranks whose GPUs differ cannot run one Plan: a Plan's executables and
profiles belong to the hardware that built them. This recipe lets such
ranks train as one data-parallel group anyway. It is reached from the same
command as every other run, by a section of the config:

```json
{"checkpoint": {"family": "llama3", "directory": "/models/Llama-3.2-1B"},
 "data": "data/llama3", "optimizer_args": {"lr": 1e-5},
 "max_seq_len": 1024, "max_tokens_per_step": 8192, "steps": 100,
 "heterogeneous": {
   "nodes": [
     {"address": "192.168.50.23", "ranks": 1,
      "env": {"NCCL_SOCKET_IFNAME": "enp114s0f1np1", "GLOO_SOCKET_IFNAME": "enp114s0f1np1"}},
     {"address": "192.168.50.32", "ranks": 1, "ssh": "tubingen",
      "repo": "/home/shein/Documents/shadowspill", "python": "/home/shein/miniconda3/envs/shadowspill/bin/python",
      "env": {"NCCL_SOCKET_IFNAME": "enp4s0f0np0", "GLOO_SOCKET_IFNAME": "enp4s0f0np0"}}],
   "shares": [3, 1]}}
```

## What the section means

| Field | Meaning |
| --- | --- |
| `nodes` | The machines, the first the one the runner is launched on; the others reached by `ssh` with the repository (`repo`), its interpreter (`python`) and the config's paths in place. `ranks` per node, one per GPU; `env` for the node's network settings, whose interface names, logging and install paths may differ between nodes while every other NCCL setting must agree. |
| `shares` | One positive integer per rank: how many of every step's units it takes, in rank order (equal by default). The ranks of one node take equal shares. |
| `port` | The rendezvous port at the first node's address (default 29400). |

A step is `max_tokens_per_step / max_seq_len` units of `max_seq_len` tokens,
packed once and deterministically; the shares divide the step's units among
the ranks consecutively. Every node builds, profiles and plans in its own
run directory, `nodes/<index>/`, for its own share of the step with its own
budgets, so each searches the microbatch geometry its hardware fits; at
execution a rank concatenates as many units as its geometry holds into one
microbatch, and every rank's loss is its share of the step's mean, the
step's trained-token count being every rank's normalizer. Held-out
evaluation takes `eval_microbatches` steps' worth of units, each rank its
share, normalized over all of them; a rank evaluates its share in as many
calls as its geometry needs, since the evaluation Plan issues no collectives.

The ranks agree on what they exchange, the parallelism, the tensors with
their placement, shards and geometry, checked when the workers open; what
each built, profiled or budgeted is its own. Calibration is a collective over
every rank, so the nodes walk the union of their candidates together: for a
candidate a node did not build, it measures its own Program of the same kind,
training or evaluation, whose exchange is the same, and keeps nothing; each
node keeps the calibrations of its own candidates. A rank writes its records and
checkpoints under its node's `rank-xxxxx/`; the distributed logger's
aggregate under `nodes/00/aggregate/` sums the ranks' shares into the step's
figures; remote nodes' rank directories are copied back to the first node
after execution, and `resume=step_XXXXXXXX` makes every rank load its own
save. The normal runner's `ranks` and `nodes` are refused beside this
section: it carries its own.

Every rank keeps whole weights and gradients in its own spill storage and
only the optimizer state is sharded, unless `build.parallelism` names
another layout. Ranks pair collectives by the order they issue them, and
each node's planner orders sharded weight fetches for its own GPU, while a
step's gradient reductions and weight publications are tasks that run in
program order on every rank. Workers compare the order their Plans issue
collectives in when they open, and refuse a difference by naming the first
collective that differs.
