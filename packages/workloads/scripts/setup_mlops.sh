#!/usr/bin/env bash
# Install mlops, the operation library the workloads models and their AdamW
# component use, with its implementation providers, into an environment that
# already has PyTorch and ShadowSpill. A checkout named mlops beside this
# repository is used as it is, with a warning when it is not at the pinned
# revision below or has uncommitted changes; otherwise the pinned revision is
# cloned into the cache and installed. mlops's own setup builds its providers.
#
#   setup_mlops.sh --python PATH [--torch-backend BACKEND] [--expert-parallel quack|te|both]
set -euo pipefail

readonly mlops_revision="678250b8536848a5b83a85517d5ea048fa98e859"
readonly repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python_executable=""
torch_backend="auto"
expert_parallel=""
while (($#)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a path" >&2; exit 2; }
      python_executable="$2"; shift 2 ;;
    --torch-backend)
      [[ $# -ge 2 ]] || { echo "--torch-backend requires a value" >&2; exit 2; }
      torch_backend="$2"; shift 2 ;;
    --expert-parallel)
      [[ $# -ge 2 ]] || { echo "--expert-parallel requires quack, te, or both" >&2; exit 2; }
      case "$2" in
        quack|te|both) expert_parallel="$2" ;;
        *) echo "--expert-parallel must be quack, te, or both" >&2; exit 2 ;;
      esac
      shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -x "${python_executable}" ]] || { echo "--python must name an executable interpreter" >&2; exit 2; }

mlops_checkout="$(cd "${repository_root}/.." && pwd)/mlops"
if [[ -x "${mlops_checkout}/scripts/setup.sh" ]]; then
  # A developer's working copy is installed as it is; say when that is not
  # the pinned revision, so an installation is never taken for the pin.
  actual="$(git -C "${mlops_checkout}" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ "${actual}" != "${mlops_revision}" ]]; then
    echo "warning: installing ${mlops_checkout} at ${actual}, not the pinned ${mlops_revision}" >&2
  elif [[ -n "$(git -C "${mlops_checkout}" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "warning: installing ${mlops_checkout} at the pinned revision with uncommitted changes" >&2
  fi
else
  mlops_checkout="${XDG_CACHE_HOME:-$HOME/.cache}/shadowspill/sources/mlops-${mlops_revision}"
  if [[ ! -d "${mlops_checkout}/.git" ]]; then
    mkdir -p "$(dirname "${mlops_checkout}")"
    git clone --filter=blob:none --no-checkout \
      https://github.com/als244/mlops.git "${mlops_checkout}"
  fi
  git -C "${mlops_checkout}" cat-file -e "${mlops_revision}^{commit}" 2>/dev/null || \
    git -C "${mlops_checkout}" fetch origin "${mlops_revision}"
  git -C "${mlops_checkout}" checkout --detach "${mlops_revision}"
fi
echo "  through ${mlops_checkout}/scripts/setup.sh, whose own steps follow"
"${mlops_checkout}/scripts/setup.sh" \
  --python "${python_executable}" \
  --torch-backend "${torch_backend}" \
  ${expert_parallel:+--expert-parallel "${expert_parallel}"}
