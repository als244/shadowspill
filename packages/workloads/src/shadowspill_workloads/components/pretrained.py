"""Resolve supported Hub architectures through the public metadata interface."""

from dataclasses import asdict, replace
from importlib import import_module

from shadowspill.training.components import ModelReferenceComponent
from shadowspill_client._records import plain
from shadowspill_workloads.initialization import headers, hub, hub_files, local_files
from shadowspill_workloads.models.families import checkpoint_families, families
from shadowspill_workloads.models.families import family as family_definition

from .source import component_source

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
    "hidden_size",
    "num_hidden_layers",
    "first_k_dense_replace",
    "intermediate_size",
    "moe_intermediate_size",
    "num_local_experts",
    "n_shared_experts",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "linear_num_heads",
    "linear_head_dim",
    "linear_conv_kernel_dim",
    "hc_mult",
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
    manifest = hub_files.describe(weights.reference, revision)
    from shadowspill_workloads.initialization.weight_files import quantization_file

    if recipe := quantization_file(metadata):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        file = hub_files.metadata(weights.reference, revision, recipe)
        with TemporaryDirectory(prefix="shadowspill-weight-recipe-") as root:
            metadata[recipe] = hub.read_config(
                hub_files.fetch(weights.reference, revision, file, Path(root))
            )
        manifest = [*manifest, asdict(file)]
    return _resolved(requested, revision, metadata, manifest)


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
    config = {**config, "architecture": values}
    describe = family_definition(family).pretrained_config
    if describe is not None:
        from pathlib import Path

        from shadowspill_workloads.initialization.weight_files import files

        inventory = files(manifest)
        tensors = (
            headers.read(Path(weights.reference), inventory)
            if local_files.accepts(weights.reference)
            else headers.from_hub(weights.reference, revision, inventory)
        )
        described = describe(metadata, tensors, values)
        for name, value in described.get("architecture", {}).items():
            if name in overrides and overrides[name] != value:
                raise ValueError(f"checkpoint storage differs from requested {name}")
        for name, value in described.items():
            if name != "architecture" and name in config and config[name] != value:
                raise ValueError(f"checkpoint {name} differs from requested storage")
        config.update(described)
    return replace(
        requested,
        name=f"shadowspill/{family}@1",
        config=config,
        initialize=replace(
            weights,
            revision=revision,
            metadata={"files": manifest},
        ),
    )


def register(registry):
    source = component_source(
        (
            "components/pretrained.py",
            "initialization/hub.py",
            "initialization/hub_files.py",
            "initialization/weight_files.py",
            "initialization/local_files.py",
            "initialization/headers.py",
            "models/shared/glm/initialization/metadata.py",
            "models/families.py",
            "models/registration.py",
            "components/text_metadata.py",
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
            ModelReferenceComponent(
                f"shadowspill/{name}@1", accepts, resolve_reference
            ),
            source=source,
        )
