"""Vision geometry, HF-reference arithmetic and static capture on CPU."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("mlops")
from workloads.mlops.glm53_flash.vision import (
    VisionConfig,
    VisionModel,
    prepare_geometry,
    prepare_images,
)


def tiny_config():
    return VisionConfig(
        depth=2,
        hidden_size=128,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=64,
        projection_intermediate_size=128,
        patch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("grid", ([[1, 4, 4]], [[1, 4, 4], [2, 2, 4]]))
def test_vision_matches_hf_output_gradients_and_export(grid):
    configuration = pytest.importorskip(
        "transformers.models.glm5_next.configuration_glm5_next"
    )
    modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    geometry = pytest.importorskip("transformers.vision_utils")
    c = tiny_config()
    hf = configuration.Glm5NextVisionConfig(
        depth=c.depth,
        hidden_size=c.hidden_size,
        num_heads=c.num_heads,
        intermediate_size=c.intermediate_size,
        out_hidden_size=c.out_hidden_size,
        projection_intermediate_size=c.projection_intermediate_size,
        patch_size=c.patch_size,
        temporal_patch_size=c.temporal_patch_size,
        spatial_merge_size=c.spatial_merge_size,
    )
    hf._attn_implementation = "sdpa"
    torch.manual_seed(81)
    reference = modeling.Glm5NextVisionModel(hf).eval()
    model = VisionModel(c, device="cpu").eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    positions, boundaries = prepare_geometry(grid)
    torch.testing.assert_close(
        positions, geometry.get_vision_position_ids(torch.tensor(grid), 2)
    )
    torch.testing.assert_close(
        torch.tensor(boundaries, dtype=torch.int32),
        geometry.get_vision_cu_seqlens(torch.tensor(grid)),
    )
    x = torch.randn(boundaries[-1], 24, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = model(x, positions, boundaries)
    expected = reference(y, torch.tensor(grid)).pooler_output
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
    parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            parameters[name].grad,
            rtol=1e-4,
            atol=1e-6,
            msg=lambda message, name=name: name + ": " + message,
        )
    exported = torch.export.export(
        model, (x.detach(), positions, boundaries), strict=True
    ).module()
    torch.testing.assert_close(
        exported(x.detach(), positions, boundaries),
        actual,
        rtol=1e-5,
        atol=1e-6,
    )


def test_image_positions_follow_expanded_placeholders_across_sequences():
    pixels = torch.randn(32, 24)
    tokens = torch.tensor([8, 9, 9, 9, 9, 3, 4, 9, 9, 9, 9, 5])
    images = prepare_images(pixels, [[1, 4, 4], [1, 4, 4]], tokens, image_token_id=9)
    assert images.boundaries == (0, 16, 32)
    assert images.token_indices.tolist() == [1, 2, 3, 4, 7, 8, 9, 10]
    assert images.pixels is pixels
    with pytest.raises(ValueError, match="placeholders"):
        prepare_images(pixels, [[1, 4, 4], [1, 4, 4]], tokens[:-3], image_token_id=9)
    with pytest.raises(ValueError, match="Patch count"):
        prepare_images(pixels[:-1], [[1, 4, 4], [1, 4, 4]], tokens, image_token_id=9)


@pytest.mark.parametrize(
    "grid,merge", (([[1, 3, 4]], 2), ([[0, 4, 4]], 2), ([], 2), ([[1, 4, 4]], 0))
)
def test_invalid_vision_geometry_fails_before_capture(grid, merge):
    with pytest.raises(ValueError):
        prepare_geometry(grid, merge_size=merge)


@pytest.mark.parametrize(
    "name",
    (
        "depth",
        "intermediate_size",
        "out_hidden_size",
        "projection_intermediate_size",
        "in_channels",
    ),
)
def test_invalid_vision_dimensions_fail_before_allocation(name):
    with pytest.raises(ValueError, match="positive"):
        replace(tiny_config(), **{name: 0})


def test_image_input_pytree_survives_serialization():
    from torch.utils._pytree import (
        tree_flatten,
        tree_unflatten,
        treespec_dumps,
        treespec_loads,
    )

    images = prepare_images(
        torch.ones(16, 24),
        [[1, 4, 4]],
        torch.tensor([9, 9, 9, 9, 1]),
        image_token_id=9,
    )
    leaves, spec = tree_flatten([images])
    restored = tree_unflatten(leaves, treespec_loads(treespec_dumps(spec)))[0]
    assert type(restored) is type(images)
    assert restored.boundaries == images.boundaries
    assert restored.pixels is images.pixels
    torch.testing.assert_close(restored.token_indices, images.token_indices)


def test_rotary_initialization_matches_hf_at_checkpoint_head_width():
    from workloads.mlops.glm53_flash.vision import rotary_embeddings, rotary_frequencies

    configuration = pytest.importorskip(
        "transformers.models.glm5_next.configuration_glm5_next"
    )
    modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    config = configuration.Glm5NextVisionConfig(hidden_size=1024, num_heads=16)
    reference = modeling.Glm5NextVisionRotaryEmbedding(config)
    positions, _ = prepare_geometry([[1, 32, 32]])
    actual = rotary_embeddings(positions, rotary_frequencies(64, 10000.0))
    expected = reference(torch.empty(len(positions), 1024), positions)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
