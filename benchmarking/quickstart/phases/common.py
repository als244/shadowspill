"""Small, CPU-only request/index helpers; no model or runtime imports."""

import importlib
import json
from hashlib import sha256
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def experiment(request, *, logical_rank=None):
    module, _, name = request["factory"].partition(":")
    factory = importlib.import_module(module)
    for component in name.split("."):
        factory = getattr(factory, component)
    arguments = dict(request.get("factory_args", {}))
    if logical_rank is not None:
        arguments["logical_rank"] = logical_rank
    return factory(device=request["device"], **arguments)


def paths(request):
    root = Path(request["output_dir"])
    return root, read(root / "programs" / "index.json")


def plan_identity(plan):
    from shadowspill.artifacts._values import encoded

    return sha256(encoded(plan.to_dict())).hexdigest()


def reference(factory):
    if callable(factory):
        if factory.__module__ == "__main__" or "<" in factory.__qualname__:
            raise ValueError(
                "quickstart workers need an importable module:function factory"
            )
        factory = factory.__module__ + ":" + factory.__qualname__
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("factory must name an importable module:function")
    return factory
