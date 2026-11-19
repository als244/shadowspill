"""Model and computation references; construction happens on the server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .._records import Record, Value, integer, mapping, names, number, record, text


@dataclass(frozen=True)
class Weights(Record):
    """Initial model weights: random, pretrained or a stored artifact.

    Construct one with `random`, `pretrained` or `artifact`. For a pretrained
    source, the installed resolver that accepts its reference pins the
    revision and metadata during setup.

    Parameters
    ----------
    kind : str
        Where the weights come from: ``"random"``, ``"pretrained"`` or
        ``"artifact"``.
    seed : int, optional
        Seed of random weights, at least zero. Required for ``"random"`` and
        refused otherwise.
    reference : str, optional
        For ``"pretrained"``, a reference an installed resolver accepts, such
        as a Hub repository ID or an absolute path on the server. For
        ``"artifact"``, the ID of an artifact in the caller's storage.
        Refused for ``"random"``.
    revision : str, optional
        Revision of a pretrained source, such as a Hub commit. When omitted,
        setup pins the revision the resolver finds. Refused for other kinds.
    metadata : mapping, default {}
        Source metadata, such as a pretrained checkpoint's file manifest. For
        a pretrained source, setup fills it from the resolver, and values
        given must agree with what the resolver finds. Refused for
        ``"random"``.

    Raises
    ------
    ValueError
        If a field is missing, or is given for a kind that refuses it.
    """

    kind: Literal["random", "pretrained", "artifact"]
    seed: int | None = None
    reference: str | None = None
    revision: str | None = None
    metadata: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", mapping(self.metadata, "weight metadata"))
        if self.kind == "random":
            integer(self.seed, "seed")
            if self.reference is not None or self.revision is not None:
                raise ValueError("random initialization has no reference or revision")
            if self.metadata:
                raise ValueError("random initialization has no referenced metadata")
        elif self.kind in ("pretrained", "artifact"):
            text(self.reference, "reference")
            if self.seed is not None:
                raise ValueError("referenced weights cannot also select a random seed")
            if self.revision is not None:
                if self.kind != "pretrained":
                    raise ValueError("only pretrained references accept a revision")
                text(self.revision, "revision")
        else:
            raise ValueError(f"unknown weights kind: {self.kind}")

    @classmethod
    def random(cls, *, seed: int) -> Weights:
        """Select random initial weights.

        Parameters
        ----------
        seed : int
            Seed, at least zero. The seed alone decides the values, whatever
            the parallelism, the number of workers or which worker owns which
            parameters.

        Returns
        -------
        Weights
            Random weights from ``seed``.
        """
        return cls("random", seed=seed)

    @classmethod
    def pretrained(cls, reference: str, *, revision: str | None = None) -> Weights:
        """Select pretrained weights.

        Parameters
        ----------
        reference : str
            A reference an installed resolver accepts, such as a Hub
            repository ID or an absolute path on the server.
        revision : str, optional
            Revision to load, such as a Hub commit. When omitted, setup pins
            the revision the resolver finds.

        Returns
        -------
        Weights
            The pretrained source.
        """
        return cls("pretrained", reference=reference, revision=revision)

    @classmethod
    def artifact(cls, artifact_id: str) -> Weights:
        """Select weights stored as an artifact.

        A weights export supplies the model values only: optimizer state
        starts fresh and the update count starts at zero.

        Parameters
        ----------
        artifact_id : str
            ID of an artifact in the caller's storage, such as the ``id`` of
            the weights `TrainingClient.export_weights` produces.

        Returns
        -------
        Weights
            The artifact source.
        """
        return cls("artifact", reference=artifact_id)


@dataclass(frozen=True)
class LoRA(Record):
    """Low-rank adapters added to a model.

    The model definition adds them during setup. The supplied text models
    then train the adapters and freeze the base parameters, unless
    `Model` ``trainable`` selects otherwise.

    Parameters
    ----------
    rank : int
        Adapter rank, at least one.
    alpha : float, optional
        Scaling numerator: each adapter's update is scaled by
        ``alpha / rank``. Positive; None means ``rank``.
    dropout : float, default 0.0
        Adapter dropout probability, in ``[0, 1)``. The supplied text models
        require zero.
    targets : sequence of str, optional
        Module-name patterns to adapt, such as ``"blocks.*.attn.wq"``.
        Nonempty and without duplicates when given. None uses the
        definition's default targets.
    """

    rank: int
    alpha: float | None = None
    dropout: float = 0.0
    targets: Sequence[str] | None = None

    def __post_init__(self) -> None:
        integer(self.rank, "rank", minimum=1)
        alpha = self.rank if self.alpha is None else self.alpha
        if number(alpha, "alpha") <= 0:
            raise ValueError("alpha must be positive")
        object.__setattr__(self, "alpha", float(alpha))
        if not 0 <= number(self.dropout, "dropout") < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.targets is not None:
            targets = names(self.targets, "targets")
            if not targets:
                raise ValueError("explicit LoRA targets must not be empty")
            object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class Model(Record):
    """An installed model definition and its training settings.

    The server constructs the model; the client only names it.

    Parameters
    ----------
    definition : str, optional
        Name of an installed model definition, such as ``"llama3"``, or its
        exact versioned ID. May be omitted only with pretrained
        ``initialize``, whose resolver then selects the definition.
    config : mapping, default {}
        Settings the definition accepts, such as ``{"preset": "tiny"}``, as
        JSON values.
    initialize : Weights, optional
        Initial weights. A preview or preparation may omit them; creating a
        run needs them unless it restores a checkpoint.
    lora : LoRA, optional
        Low-rank adapters to add, for a definition that supports them.
    implementation : str, optional
        One of the implementations the definition offers, such as
        ``"mlops_ep"``. None uses the definition's default.
    precision : mapping, default {}
        Dtype settings. ``"parameter_dtypes"`` maps parameter and buffer name
        patterns to ``"float16"``, ``"bfloat16"``, ``"float32"`` or
        ``"float64"``, overriding the stored dtype of each floating match;
        each pattern must match. Other keys belong to the definition, such as
        ``{"compute_dtype": "bfloat16"}``.
    trainable : sequence of str, optional
        Parameter names or glob patterns to train; every other parameter is
        frozen. Selection follows adapter construction, so include adapter
        patterns such as ``"*.lora_*"`` to keep training them. Each pattern
        must match, and an empty sequence freezes every parameter. None keeps
        the definition's selection.

    Raises
    ------
    ValueError
        If neither ``definition`` nor pretrained ``initialize`` is given.
    """

    definition: str | None = None
    config: Mapping[str, Value] = field(default_factory=dict)
    initialize: Weights | None = None
    lora: LoRA | None = None
    implementation: str | None = None
    precision: Mapping[str, Value] = field(default_factory=dict)
    trainable: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.definition is not None:
            text(self.definition, "definition")
        if self.initialize is not None:
            object.__setattr__(
                self, "initialize", record(Weights, self.initialize, "initialize")
            )
        if self.definition is None and (
            self.initialize is None or self.initialize.kind != "pretrained"
        ):
            raise ValueError("select an installed definition or a pretrained model")
        if self.lora is not None:
            object.__setattr__(self, "lora", record(LoRA, self.lora, "lora"))
        if self.implementation is not None:
            text(self.implementation, "implementation")
        object.__setattr__(self, "config", mapping(self.config, "config"))
        object.__setattr__(self, "precision", mapping(self.precision, "precision"))
        if self.trainable is not None:
            object.__setattr__(self, "trainable", names(self.trainable, "trainable"))

    @classmethod
    def from_pretrained(
        cls,
        reference: str,
        *,
        revision: str | None = None,
        lora: LoRA | None = None,
        implementation: str | None = None,
        config: Mapping[str, Value] | None = None,
        precision: Mapping[str, Value] | None = None,
        trainable: Sequence[str] | None = None,
    ) -> Model:
        """Select a model by a pretrained reference.

        During setup, the installed resolver that accepts ``reference``
        selects the model definition and pins the revision and metadata.

        Parameters
        ----------
        reference : str
            Pretrained source, such as a Hub repository ID or an absolute path
            on the server.
        revision : str, optional
            Revision to load. When omitted, setup pins the revision the
            resolver finds.
        lora : LoRA, optional
            Low-rank adapters to add.
        implementation : str, optional
            Implementation the definition offers, as in `Model`.
        config : mapping, optional
            Definition settings, as in `Model`. The resolver fills settings
            left out from the source's metadata and refuses to change any
            given.
        precision : mapping, optional
            Dtype settings, as in `Model`.
        trainable : sequence of str, optional
            Parameter names or glob patterns to train, as in `Model`.

        Returns
        -------
        Model
            The model, initialized from ``reference``.
        """
        return cls(
            initialize=Weights.pretrained(reference, revision=revision),
            lora=lora,
            implementation=implementation,
            config=config or {},
            precision=precision or {},
            trainable=trainable,
        )


@dataclass(frozen=True)
class Objective(Record):
    """An installed objective definition and its settings.

    Parameters
    ----------
    definition : str
        Name of an installed objective definition, such as
        ``"cross_entropy"``, or its exact versioned ID.
    config : mapping, default {}
        Structural settings the definition accepts, such as
        ``{"reference_penalty": "k3"}``. They are fixed for the run.
    defaults : mapping, default {}
        Default values of the objective's per-call controls, such as
        ``{"clip_ratio": 0.2}``. A call's ``loss_params`` override them for
        that call only.
    """

    definition: str
    config: Mapping[str, Value] = field(default_factory=dict)
    defaults: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text(self.definition, "definition")
        object.__setattr__(self, "config", mapping(self.config, "config"))
        object.__setattr__(self, "defaults", mapping(self.defaults, "defaults"))

    @classmethod
    def sum(
        cls,
        terms: Mapping[str, Objective],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> Objective:
        """Combine installed objectives into one weighted sum.

        The terms share one model evaluation; setup checks that their
        evaluation, inputs and participation are compatible. Each term's
        controls and results carry its name as a prefix, as in
        ``tokens.loss_sum``, and its coefficient is the control
        ``<name>.weight``, which ``loss_params`` can override per call. The
        sum is not averaged.

        Parameters
        ----------
        terms : mapping of str to Objective
            Terms by name. Names are Python identifiers, and no term may
            itself be a sum.
        weights : mapping of str to float, optional
            Coefficient of each named term. Terms left out have coefficient
            one.

        Returns
        -------
        Objective
            The sum, as the definition ``"shadowspill/sum@1"``.

        Raises
        ------
        ValueError
            If ``terms`` is empty, a name is not an identifier, a term is a
            sum, or ``weights`` names an unknown term or holds a value that is
            not a finite number.
        TypeError
            If a term is not an `Objective`.
        """
        if not isinstance(terms, Mapping) or not terms:
            raise ValueError("an objective sum requires named terms")
        for name, term in terms.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("objective term names must be identifiers")
            if not isinstance(term, Objective):
                raise TypeError("objective terms must be Objective settings")
            if term.definition == "shadowspill/sum@1":
                raise ValueError("flatten nested objective sums into named terms")
        weights = {} if weights is None else weights
        if set(weights) - terms.keys():
            raise ValueError("objective weights reference unknown terms")
        return cls(
            "shadowspill/sum@1",
            config={"terms": {name: term.to_dict() for name, term in terms.items()}},
            defaults={
                f"{name}.weight": number(value, f"{name}.weight")
                for name, value in weights.items()
            },
        )


@dataclass(frozen=True)
class Optimizer(Record):
    """An installed optimizer definition and its settings.

    Per-update controls, such as the learning rate, are passed on each update
    as ``optim``.

    Parameters
    ----------
    definition : str
        Name of an installed optimizer definition, such as ``"adamw"``, or its
        exact versioned ID.
    defaults : mapping, default {}
        Default values of the per-update controls, such as ``{"lr": 0.001}``.
        A call's ``optim`` values override them for that call only.
    config : mapping, default {}
        Structural settings the definition accepts, such as
        ``{"maximize": True}``.
    groups : mapping, default {}
        Parameter-group settings, for a definition that supports several
        groups.
    precision : mapping, default {}
        Dtypes and rounding of gradients, optimizer state and master weights,
        as the definition accepts them, such as
        ``{"gradient_dtype": "float32", "opt_state_dtype": "float32"}``.
    """

    definition: str
    defaults: Mapping[str, Value] = field(default_factory=dict)
    config: Mapping[str, Value] = field(default_factory=dict)
    groups: Mapping[str, Value] = field(default_factory=dict)
    precision: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text(self.definition, "definition")
        for name in ("defaults", "config", "groups", "precision"):
            object.__setattr__(self, name, mapping(getattr(self, name), name))
