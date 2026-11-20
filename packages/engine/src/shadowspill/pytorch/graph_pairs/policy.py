"""Independent saved-state cuts for the variants a checkpoint region offers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import torch

from ..capture.retention import RetentionPolicy


@dataclass(frozen=True, slots=True)
class GraphPairPolicy:
    """The retention policy of each graph pair a checkpoint stage offers.

    A stage offers a ``save`` and a ``recompute`` variant, each with its own
    saved-state cut; planning chooses among the measured variants and decides
    where retained values reside. `build_forward_backward` and `build_step`
    take one as ``graph_pair_policy``.

    Parameters
    ----------
    save : RetentionPolicy, default RetentionPolicy()
        Policy of the ``save`` variant. The default retains by cost.
    recompute : RetentionPolicy, default RetentionPolicy(default="recompute")
        Policy of the ``recompute`` variant. The default regenerates from the
        checkpoint inputs.

    Raises
    ------
    TypeError
        If either is not a `RetentionPolicy`.
    """

    save: RetentionPolicy = field(default_factory=RetentionPolicy)
    recompute: RetentionPolicy = field(
        default_factory=lambda: RetentionPolicy(default="recompute")
    )

    def __post_init__(self) -> None:
        for name in ("save", "recompute"):
            if not isinstance(getattr(self, name), RetentionPolicy):
                raise TypeError(f"{name} must be a RetentionPolicy")

    def variant(self, name: str) -> RetentionPolicy:
        """Return the policy of one variant.

        Parameters
        ----------
        name : {"save", "recompute"}
            The variant.

        Returns
        -------
        RetentionPolicy
            That variant's policy.

        Raises
        ------
        ValueError
            If ``name`` is neither variant.
        """
        if name == "save":
            return self.save
        if name == "recompute":
            return self.recompute
        raise ValueError(f"unknown graph-pair variant {name}; choose save or recompute")

    def identity(self) -> dict[str, object]:
        """Return both policies' configurations, as `RetentionPolicy.identity`.

        Returns
        -------
        dict
            The ``"save"`` and ``"recompute"`` identities.
        """
        return {"save": self.save.identity(), "recompute": self.recompute.identity()}

    def operator_classes(
        self, graph_module: torch.fx.GraphModule
    ) -> dict[str, dict[str, str]]:
        """Classify every custom operator in ``graph_module`` under each policy.

        Parameters
        ----------
        graph_module : torch.fx.GraphModule
            A traced graph.

        Returns
        -------
        dict
            The ``"save"`` and ``"recompute"`` results of
            `RetentionPolicy.operator_classes`.
        """
        return {
            "save": self.save.operator_classes(graph_module),
            "recompute": self.recompute.operator_classes(graph_module),
        }

    def digest(self, graph_module: torch.fx.GraphModule) -> str:
        """Return the SHA-256 digest of `capture_identity`.

        The graph-pair store keys a stage's graph pairs by it.

        Parameters
        ----------
        graph_module : torch.fx.GraphModule
            The stage's traced graph.

        Returns
        -------
        str
            The digest, in hexadecimal.

        Raises
        ------
        CaptureError
            If `RetentionPolicy.capture_identity` raises it for either variant.
        """
        encoded = json.dumps(
            self.capture_identity(graph_module),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    def capture_identity(self, graph_module: torch.fx.GraphModule) -> dict[str, object]:
        """Return what both policies decide for ``graph_module``.

        Combines each variant's `RetentionPolicy.capture_identity`.

        Parameters
        ----------
        graph_module : torch.fx.GraphModule
            The stage's traced graph.

        Returns
        -------
        dict
            The JSON-compatible description.

        Raises
        ------
        CaptureError
            If `RetentionPolicy.capture_identity` raises it for either variant.
        """
        variants = {
            name: self.variant(name).capture_identity(graph_module)
            for name in ("save", "recompute")
        }
        result = {
            **self.identity(),
            "operators": {
                name: identity["operators"] for name, identity in variants.items()
            },
        }
        scoped = {
            name: identity["scoped"]
            for name, identity in variants.items()
            if "scoped" in identity
        }
        if scoped:
            result["scoped"] = scoped
        choices = {
            name: identity["forward_choices"]
            for name, identity in variants.items()
            if "forward_choices" in identity
        }
        if choices:
            result["forward_choices"] = choices
        return result


__all__ = ["GraphPairPolicy"]
