"""Explain task waits from saved input dependencies and admission edges.

Colors alone never imply a dependency. A transfer belongs to a task's
explanation only through its input alias or an explicit memory-reuse edge.
Other copies can delay those transfers by occupying the same lane.
"""

from __future__ import annotations

from collections import defaultdict

from tools.pressurefit_trace.data import short_task


class TaskWaits:
    def __init__(self, data, admission):
        self.data = data
        self.details = {}
        self.task_reuse = defaultdict(list)
        self.action_reuse = defaultdict(list)
        for edge in admission.get("reuse_dependencies", ()):
            predecessor = edge["predecessor_action_index"]
            if edge.get("successor_task_id") is not None:
                self.task_reuse[edge["successor_task_id"]].append(predecessor)
            else:
                self.action_reuse[edge["successor_action_index"]].append(predecessor)
        self.by_alias = defaultdict(list)
        self.by_lane = defaultdict(list)
        self.consumers = defaultdict(list)
        for i, task in enumerate(data.tasks):
            inputs = {data.object_alias[o] for o in task["inputs"]}
            inputs.update(data.object_alias[m["object_id"]] for m in task["mutations"])
            for alias in inputs:
                self.consumers[alias].append(i)
        for action in data.transfers:
            self.by_alias[action["alias_group_id"]].append(action)
            self.by_lane[action["transfer"]["direction"]].append(action)
        self.task_leases = defaultdict(list)
        self.alias_leases = defaultdict(list)
        for lease in data.leases:
            self.task_leases[lease["task_id"]].append(lease)
            self.alias_leases[lease["alias_group_id"]].append(lease)

    def requirements(self, task):
        def aliases(objects):
            return {self.data.object_alias[o] for o in objects}

        inputs = aliases(task["inputs"])
        outputs = aliases(task["outputs"])
        mutated = aliases(m["object_id"] for m in task["mutations"])

        def total(group):
            return sum(self.data.facts.alias_size[a] for a in group)

        return {
            "inputs": {"bytes": total(inputs), "aliases": len(inputs)},
            "outputs": {"bytes": total(outputs), "aliases": len(outputs)},
            "mutated": {"bytes": total(mutated), "aliases": len(mutated)},
            "workspace_bytes": task["workspace_bytes"],
            "distinct_object_bytes": total(inputs | outputs | mutated),
        }

    def reuse_purposes(self, task, action):
        """Which task allocations overlap this eviction's source addresses?"""
        end = action["transfer"]["end_ns"]
        sources = [
            p
            for p in self.alias_leases[action["alias_group_id"]]
            if p["predicted_end_ns"] == end and p["bytes"] > 0
        ]
        return sorted(
            {
                p["purpose"]
                for p in self.task_leases[task["task_id"]]
                if p["predicted_start_ns"] == task["start_ns"]
                and p["bytes"] > 0
                and any(
                    p["offset"] < s["offset"] + s["bytes"]
                    and s["offset"] < p["offset"] + p["bytes"]
                    for s in sources
                )
            }
        )

    def detail(self, index):
        if index in self.details:
            return self.details[index]
        task = self.data.tasks[index]
        ready, start = task["ready_ns"], task["start_ns"]
        inputs = {self.data.object_alias[o] for o in task["inputs"]}
        inputs.update(self.data.object_alias[m["object_id"]] for m in task["mutations"])
        inputs_waiting = [
            a["index"]
            for alias in sorted(inputs)
            for a in self.by_alias[alias]
            if a["transfer"]["direction"] == "fetch"
            and ready < a["transfer"]["end_ns"] <= start
        ]
        reuse_waiting = [
            i
            for i in self.task_reuse[task["task_id"]]
            if ready < self.data.actions[i]["transfer"]["end_ns"] <= start
        ]
        blockers = [
            {"action": i, "relation": "required_input", "purposes": []}
            for i in inputs_waiting
        ] + [
            {
                "action": i,
                "relation": "storage_reuse",
                "purposes": self.reuse_purposes(task, self.data.actions[i]),
            }
            for i in reuse_waiting
        ]
        blockers.sort(
            key=lambda b: (
                -self.data.actions[b["action"]]["transfer"]["end_ns"],
                b["relation"] != "required_input",
                b["action"],
            )
        )
        critical = blockers[0]["action"] if blockers else None
        # Include upstream storage dependencies of a required fetch. These
        # are not input aliases of the task and must be labeled separately.
        upstream = []
        for i in inputs_waiting:
            fetch = self.data.actions[i]
            for predecessor in self.action_reuse[i]:
                action = self.data.actions[predecessor]
                if (
                    fetch["trigger_ns"]
                    < action["transfer"]["end_ns"]
                    <= fetch["transfer"]["start_ns"]
                ):
                    upstream.append({"action": predecessor, "successor_action": i})
        direct = set(inputs_waiting) | set(reuse_waiting)
        ahead = []
        if critical is not None:
            action = self.data.actions[critical]
            t = action["transfer"]
            if "lane-busy" in t["stall_reasons"]:
                ahead = [
                    a["index"]
                    for a in self.by_lane[t["direction"]]
                    if a["index"] != critical
                    and a["transfer"]["end_ns"] > action["trigger_ns"]
                    and a["transfer"]["start_ns"] < t["start_ns"]
                    and a["transfer"]["end_ns"] <= t["start_ns"]
                ]
        later = []
        for i in ahead:
            action = self.data.actions[i]
            consumer = next(
                (
                    t
                    for t in self.consumers[action["alias_group_id"]]
                    if t >= action["boundary"]
                ),
                None,
            )
            if consumer is not None and consumer > index:
                later.append(i)
        result = {
            "index": index,
            "task_id": task["task_id"],
            "label": short_task(task),
            "boundary": index,
            "ready_ns": ready,
            "start_ns": start,
            "end_ns": task["end_ns"],
            "stall_ns": start - ready,
            "stall_reasons": task["stall_reasons"],
            "requirements": self.requirements(task),
            "blocking_transfers": blockers,
            "critical_action": critical,
            "input_fetch_bytes": sum(
                self.data.actions[i]["bytes"] for i in inputs_waiting
            ),
            "storage_reuse_bytes": sum(
                self.data.actions[i]["bytes"] for i in reuse_waiting
            ),
            "upstream_storage_dependencies": upstream,
            "lane_predecessors": ahead,
            "lane_predecessor_bytes": sum(
                self.data.actions[i]["transfer_bytes"] for i in ahead
            ),
            "unrelated_lane_predecessor_bytes": sum(
                self.data.actions[i]["transfer_bytes"] for i in ahead if i not in direct
            ),
            "later_use_lane_predecessors": later,
            "later_use_predecessor_bytes": sum(
                self.data.actions[i]["transfer_bytes"] for i in later
            ),
            "evidence": (
                "saved input aliases, transfer timings, and admission reuse edges"
            ),
        }
        self.details[index] = result
        return result
