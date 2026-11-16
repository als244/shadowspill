"""Resolve supported Hub architectures through the public metadata interface."""

from dataclasses import asdict, replace
from importlib import import_module

from shadowspill.training.definitions import ModelReferenceDefinition
from shadowspill_client._records import plain
from shadowspill_workloads.initialization import hub, hub_files, local_files
from shadowspill_workloads.models.families import checkpoint_families, families

from .source import definition_source

# These settings select the stored tensors or their layout. Other supported
# overrides (e.g. rotary capacity/frequencies or top-k) do not reshape weights.
WEIGHT_GEOMETRY = {
    "n_layers",
    "d_model",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "d_ff",
    "vocab_size",
    "tied_embeddings",
    "n_experts",
    "d_ff_expert",
    "d_ff_shared",
    "full_attention_interval",
    "lin_k_heads",
    "lin_v_heads",
    "lin_k_head_dim",
    "lin_v_head_dim",
    "lin_conv_kernel",
    "attention_gate",
}


def architecture(metadata, overrides):
    text = metadata.get("text_config", metadata)
    if not isinstance(text, dict):
        raise ValueError("pretrained text_config must be a JSON object")
    kind = text.get("model_type", metadata.get("model_type"))
    supported = checkpoint_families(training=True)
    if kind not in supported:
        raise ValueError(f"unsupported pretrained model type: {kind!r}")
    registration = supported[kind]
    family = registration.name
    module = import_module(f"shadowspill_workloads.models.{family}.config")
    defaults = {}
    if "max_position_embeddings" in text:
        capacity = text["max_position_embeddings"]
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("pretrained maximum sequence length must be positive")
        defaults["max_seq_len"] = capacity
    original = module.from_huggingface(metadata, **defaults)
    if registration.validate_pretrained is not None:
        registration.validate_pretrained(text, original)
    values = asdict(original)
    for name in WEIGHT_GEOMETRY & overrides.keys():
        if name in values and values[name] != overrides[name]:
            raise ValueError(f"pretrained override changes checkpoint geometry: {name}")
    return family, asdict(replace(original, **overrides))


def resolve(requested):
    weights = requested.initialize
    revision, metadata = hub.config(weights.reference, weights.revision)
    return _resolved(
        requested, revision, metadata, hub_files.describe(weights.reference, revision)
    )


def resolve_local(requested):
    weights = requested.initialize
    if weights.metadata and set(weights.metadata) != {"files"}:
        raise ValueError("local checkpoint requires a resolved file manifest")
    reference, metadata, manifest = local_files.describe(
        weights.reference, weights.metadata.get("files")
    )
    revision = local_files.revision(manifest)
    if weights.revision is not None and weights.revision != revision:
        raise ValueError("local checkpoint differs from its pinned content revision")
    requested = replace(requested, initialize=replace(weights, reference=reference))
    return _resolved(requested, revision, metadata, [asdict(file) for file in manifest])


def _resolved(requested, revision, metadata, manifest):
    weights = requested.initialize
    config = plain(requested.config)
    overrides = config.get("architecture", {})
    if not isinstance(overrides, dict):
        raise ValueError("pretrained architecture overrides must be a mapping")
    family, values = architecture(metadata, overrides)
    return replace(
        requested,
        definition=f"shadowspill/{family}@1",
        config={**config, "architecture": values},
        initialize=replace(
            weights,
            revision=revision,
            metadata={"files": manifest},
        ),
    )


def register(registry):
    source = definition_source(
        (
            "definitions/pretrained.py",
            "initialization/hub.py",
            "initialization/hub_files.py",
            "initialization/weight_files.py",
            "initialization/local_files.py",
            "initialization/headers.py",
            "models/families.py",
            "models/registration.py",
            "definitions/text_metadata.py",
            *(
                path
                for family in families()
                for path in (
                    f"models/{family.name}/registration.py",
                    *(
                        (f"models/{family.name}/config.py",)
                        if family.training and family.checkpoint_types
                        else ()
                    ),
                )
            ),
            "models/shared/qwen_moe/config.py",
        ),
        providers=("huggingface_hub", "httpx", "httpcore", "httpx2", "httpcore2"),
    )
    for name, accepts, resolve_reference in (
        ("huggingface", hub.accepts, resolve),
        ("local_safetensors", local_files.accepts, resolve_local),
    ):
        registry.add_model_reference(
            ModelReferenceDefinition(
                f"shadowspill/{name}@1", accepts, resolve_reference
            ),
            source=source,
        )
