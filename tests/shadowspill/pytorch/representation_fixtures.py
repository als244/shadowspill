"""Small integer/scaled tensor for representation tests; only imports PyTorch."""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.utils._pytree import tree_map


class ScaledWeight(torch.Tensor):
    def __repr__(self):
        return f"ScaledWeight(shape={tuple(self.shape)}, device={self.device})"

    @staticmethod
    def __new__(cls, payload, scale):
        value = torch.Tensor._make_wrapper_subclass(
            cls,
            payload.shape,
            strides=payload.stride(),
            device=payload.device,
            dtype=torch.float32,
        )
        value.payload, value.scale = payload, scale
        return value

    def __tensor_flatten__(self):
        return ["payload", "scale"], None

    @staticmethod
    def __tensor_unflatten__(components, metadata, outer_size, outer_stride):
        return ScaledWeight(components["payload"], components["scale"])

    def dense(self):
        return self.payload.float() * self.scale

    @classmethod
    def __torch_dispatch__(cls, op, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten = torch.ops.aten
        if op is aten.detach.default:
            x = args[0]
            return return_and_correct_aliasing(
                op, args, kwargs, cls(x.payload.detach(), x.scale.detach())
            )
        if op is aten.clone.default:
            x = args[0]
            return cls(x.payload.clone(), x.scale.clone())
        if (
            op is aten.empty_like.default
            and kwargs.get("dtype", args[0].dtype) == args[0].dtype
        ):
            x = args[0]
            return cls(torch.empty_like(x.payload), torch.empty_like(x.scale))
        if op is aten._to_copy.default:
            first = args[0]
            if kwargs.get("dtype", first.dtype) != first.dtype:
                return first.dense().to(**kwargs)
            options = {key: value for key, value in kwargs.items() if key != "dtype"}
            return cls(
                aten._to_copy.default(first.payload, **options),
                aten._to_copy.default(first.scale, **options),
            )
        if op is aten.copy_.default:
            target, source = args[:2]
            if isinstance(target, cls):
                if isinstance(source, cls):
                    target.payload.copy_(source.payload)
                    target.scale.copy_(source.scale)
                else:
                    scale = (source.float().abs().amax() / 127).clamp_min(1e-8)
                    target.payload.copy_(
                        (source.float() / scale).round().clamp(-127, 127).to(torch.int8)
                    )
                    target.scale.copy_(scale)
                return target
            return target.copy_(source.dense())
        if op in (aten.t.default, aten.transpose.int):
            x = args[0]
            return return_and_correct_aliasing(
                op, args, kwargs, cls(op(x.payload, *args[1:], **kwargs), x.scale)
            )
        if op is aten._assert_tensor_metadata.default:
            with torch._C._DisableTorchDispatch():
                return op(*args, **kwargs)

        def plain(value):
            return value.dense() if isinstance(value, cls) else value

        return op(*tree_map(plain, args), **tree_map(plain, kwargs))


def model():
    result = torch.nn.Linear(8, 4, bias=False)
    result.weight = torch.nn.Parameter(
        ScaledWeight(
            torch.arange(-16, 16, dtype=torch.int8).reshape(4, 8), torch.tensor(0.025)
        )
    )
    return result
