#!/usr/bin/env bash
set -euo pipefail

readonly script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly project_root="$(cd "${script_root}/../.." && pwd)"

environment_name="baselines"
python_executable=""
torch_backend="auto"
expert_parallel=""
conda_executable=""
skip_shadowspill=0

# The versions this environment is verified with. PyTorch's own pin lives in
# scripts/setup.sh (2.13); these are the baseline systems on top of it.
readonly deepspeed_version="0.19.7"
readonly torchtitan_version="0.3.0"
readonly torchao_version="0.18.0"

usage() {
  cat <<'EOF'
Usage: baselines/scripts/setup.sh [--env NAME | --python PATH]
                                  [--torch-backend BACKEND]
                                  [--expert-parallel quack|te|both]
                                  [--skip-shadowspill]

Create the environment every baseline harness runs in: ShadowSpill (built
from this checkout) with mlops and its providers, through scripts/setup.sh,
and then DeepSpeed, torchtitan and torchao at the versions verified with
PyTorch 2.13. torchtune is not installed: its import needs an older torchao
than the LoRA and FP8 cases use. By default a Conda environment named
`baselines` is created with Python 3.12; --python installs into an existing
interpreter instead. --expert-parallel passes through to mlops. The
Megatron-Core/Bridge and NeMo AutoModel recipe harnesses have their own
pinned environments and scripts. --skip-shadowspill leaves step 1 out when
the environment already holds ShadowSpill and mlops from an earlier run.

The ShadowSpill build directory is shared with any other environment built
from this checkout at the same Python and PyTorch (build/<wheel tag>); the
libraries are identical, so the sharing is harmless.
EOF
}

while (($#)); do
  case "$1" in
    --env)
      [[ $# -ge 2 ]] || { echo "--env requires a name" >&2; exit 2; }
      environment_name="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a path" >&2; exit 2; }
      python_executable="$2"
      shift 2
      ;;
    --torch-backend)
      [[ $# -ge 2 ]] || { echo "--torch-backend requires a value" >&2; exit 2; }
      torch_backend="$2"
      shift 2
      ;;
    --expert-parallel)
      [[ $# -ge 2 ]] || { echo "--expert-parallel requires quack, te, or both" >&2; exit 2; }
      expert_parallel="$2"
      shift 2
      ;;
    --skip-shadowspill)
      skip_shadowspill=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${python_executable}" ]]; then
  conda_executable="$(command -v conda || true)"
  if [[ -z "${conda_executable}" ]]; then
    echo "setup needs conda to create the ${environment_name} environment, or --python" >&2
    exit 1
  fi
  conda_root="$("${conda_executable}" info --base)"
  python_executable="${conda_root}/envs/${environment_name}/bin/python"
  if [[ -x "${python_executable}" ]]; then
    echo "[0/4] Conda environment ${environment_name} exists; installing into it"
  else
    echo "[0/4] Creating Conda environment ${environment_name} with Python 3.12"
    "${conda_executable}" create --yes --quiet --name "${environment_name}" python=3.12
  fi
fi
if [[ ! -x "${python_executable}" ]]; then
  echo "Python interpreter is not executable: ${python_executable}" >&2
  exit 1
fi

uv_executable="$(command -v uv || true)"
if [[ -z "${uv_executable}" ]]; then
  echo "setup needs uv on PATH, as scripts/setup.sh does" >&2
  exit 1
fi

if [[ "${skip_shadowspill}" == 1 ]]; then
  echo "[1/4] Keeping the ShadowSpill and mlops already in the environment"
else
  echo "[1/4] ShadowSpill, mlops and the providers through scripts/setup.sh"
  "${project_root}/scripts/setup.sh" \
    --python "${python_executable}" \
    --torch-backend "${torch_backend}" \
    ${expert_parallel:+--expert-parallel "${expert_parallel}"}
fi

echo "[2/4] DeepSpeed ${deepspeed_version}, torchtitan ${torchtitan_version}, torchao ${torchao_version}"
# PyTorch is pinned to the exact build step 1 installed (its local version
# names the CUDA build) so the resolver cannot move it. uv's torch-backend
# mode is not used here: it confines every PyTorch-family package, torchdata
# included, to PyTorch's index, where torchtune's torchdata pin is absent.
# Instead PyTorch's index for this CUDA build is an extra index, so torchao's
# CUDA wheel comes from it, and the index strategy lets every other package
# resolve from PyPI.
torch_version="$("${python_executable}" -c 'import torch; print(torch.__version__)')"
torch_cuda_tag="$("${python_executable}" -c 'import torch; v = torch.version.cuda; print("cu" + v.replace(".", "") if v else "cpu")')"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --extra-index-url "https://download.pytorch.org/whl/${torch_cuda_tag}" \
  --index-strategy unsafe-best-match \
  "torch==${torch_version}" \
  "deepspeed==${deepspeed_version}" \
  "torchtitan==${torchtitan_version}" \
  "torchao==${torchao_version}" \
  "pyyaml"

echo "[3/4] Verifying the baseline systems import beside PyTorch ${torch_version}"
"${python_executable}" - <<'PY'
from importlib import import_module
from importlib.metadata import version

import torch

release = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if release != (2, 13):
    raise RuntimeError(f"expected PyTorch 2.13, found {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a GPU; check the NVIDIA driver")
print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)})")

for module_name, package_name in (
    ("shadowspill", "shadowspill"),
    ("mlops", "mlops"),
    ("deepspeed", "deepspeed"),
    ("torchtitan", "torchtitan"),
    ("torchao", "torchao"),
    ("yaml", "pyyaml"),
):
    import_module(module_name)
    print(f"{package_name}: {version(package_name)}")

# The pieces the harnesses call, so a missing one fails here, not in a sweep.
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # noqa: E402
    checkpoint_wrapper,
    offload_wrapper,
)
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard  # noqa: E402

from torchao.float8 import Float8LinearConfig  # noqa: E402
from torchao.optim import AdamW4bit, AdamW8bit, CPUOffloadOptimizer  # noqa: E402

import deepspeed.runtime.zero  # noqa: E402

for name in (checkpoint_wrapper, offload_wrapper, CPUOffloadPolicy, fully_shard,
             Float8LinearConfig, AdamW4bit, AdamW8bit, CPUOffloadOptimizer,
             deepspeed.runtime.zero):
    assert name is not None
print("harness entry points: available")
PY

echo "[4/4] Recording the environment"
"${python_executable}" - <<'PY' > "${project_root}/baselines/scripts/environment.lock.json"
import importlib.metadata, json, platform, sys

import torch

packages = sorted(
    (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
)
print(json.dumps({
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "packages": dict(packages),
}, indent=2))
PY
echo "Baselines setup is complete: ${python_executable}"
echo "Activate it with: conda activate ${environment_name}"
