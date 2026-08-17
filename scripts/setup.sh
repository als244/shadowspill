#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly default_environment="${project_root}/.venv"

python_executable=""
torch_backend="auto"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--python PATH] [--torch-backend BACKEND]

Create a complete ShadowSpill development environment.  By default the script
creates .venv with Python 3.12 and automatically selects the installed GPU's
PyTorch backend.  --python installs into an existing virtual or Conda
environment instead.
EOF
}

while (($#)); do
  case "$1" in
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

uv_executable="$(command -v uv || true)"
bootstrap_dir=""
if [[ -z "${uv_executable}" ]]; then
  bootstrap_python=""
  bootstrap_python="$(command -v python3 || command -v python || true)"
  if [[ -z "${bootstrap_python}" ]]; then
    echo "setup requires uv or a Python interpreter with venv support" >&2
    exit 1
  fi

  bootstrap_dir="$(mktemp -d)"
  "${bootstrap_python}" -m venv "${bootstrap_dir}"
  "${bootstrap_dir}/bin/python" -m pip install --quiet "uv>=0.9,<1" >&2
  uv_executable="${bootstrap_dir}/bin/uv"
  trap 'rm -rf -- "${bootstrap_dir}"' EXIT
fi
readonly uv_executable

if [[ -z "${python_executable}" ]]; then
  "${uv_executable}" venv --allow-existing --python 3.12 "${default_environment}"
  python_executable="${default_environment}/bin/python"
fi

if [[ ! -x "${python_executable}" ]]; then
  echo "Python interpreter is not executable: ${python_executable}" >&2
  exit 1
fi

echo "[1/4] Installing PyTorch 2.13 with backend '${torch_backend}'"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  "torch>=2.13,<2.14"

echo "[2/4] Building and installing ShadowSpill"
torch_cmake_prefix="$(
  "${python_executable}" -c '
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("torch")
if spec is None or spec.submodule_search_locations is None:
    raise RuntimeError("installed PyTorch package cannot be located")
print(Path(next(iter(spec.submodule_search_locations))) / "share" / "cmake")
'
)"
"${uv_executable}" pip install \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  --config-setting "cmake.define.CMAKE_PREFIX_PATH=${torch_cmake_prefix}" \
  --config-setting "cmake.define.Python3_EXECUTABLE=${python_executable}" \
  --editable "${project_root}[pytorch,dev]"

echo "[3/4] Verifying PyTorch and the accelerator backend"
"${python_executable}" - <<'PY'
import torch

version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if version != (2, 13):
    raise RuntimeError(f"expected PyTorch 2.13, found {torch.__version__}")
if torch.version.cuda is None:
    raise RuntimeError(f"expected a CUDA-enabled PyTorch build, found {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch has a CUDA backend but cannot access a GPU; check the NVIDIA driver"
    )

print(f"PyTorch: {torch.__version__}")
print(f"CUDA backend: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
PY

echo "[4/4] Verifying ShadowSpill's installed compiled components"
"${python_executable}" - <<'PY'
import ctypes

import torch

from shadowspill._libraries import resolve_library
from shadowspill.planner._capi import load_planner_library, planner_library_path
from shadowspill.pytorch.runtime_adapter.allocator import (
    _REQUIRED_STORAGE_OPERATIONS,
)
from shadowspill.simulator._capi import load_simulator_library, simulator_library_path

libraries = (
    "libshadowspill_simulator.so",
    "libshadowspill_runtime.so",
    "libshadowspill_backend_mock.so",
    "libshadowspill_backend_cuda.so",
    "libshadowspill_planner.so",
    "libshadowspill_pytorch.so",
)
missing = [filename for filename in libraries if resolve_library(filename) is None]
if missing:
    raise RuntimeError(f"missing compiled ShadowSpill libraries: {', '.join(missing)}")

for filename in libraries:
    print(f"{filename}: {resolve_library(filename)}")

# Planning requires these libraries. Load and ABI-check them here so setup
# cannot complete with a missing or incompatible compiled backend.
load_planner_library()
load_simulator_library()
print(f"PressureFit backend: compiled C ({planner_library_path()})")
print(f"Simulator backend: compiled C ({simulator_library_path()})")

adapter = resolve_library("libshadowspill_pytorch.so")
assert adapter is not None
ctypes.CDLL(str(adapter), mode=ctypes.RTLD_GLOBAL)
missing_operations = [
    name
    for name in _REQUIRED_STORAGE_OPERATIONS
    if not hasattr(torch.ops.shadowspill, name)
]
if missing_operations:
    raise RuntimeError(
        "PyTorch adapter is missing canonical storage operations: "
        + ", ".join(missing_operations)
    )
print("PyTorch storage adapter: available")
PY

echo "ShadowSpill setup is complete."
if [[ "${python_executable}" == "${default_environment}/bin/python" ]]; then
  echo "Activate it with: source .venv/bin/activate"
fi
