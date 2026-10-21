#!/usr/bin/env bash
set -euo pipefail

readonly script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

environment_name="nvidia-training"
python_executable=""
sources=""
cuda_archs=""
jobs=""
nccl_ep=0
# uv reads a download for at most 30 s by default, too short for PyTorch's
# CUDA wheels from a slow mirror; a caller's own value is kept.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

# The pinned revisions the recipe harnesses run (megatron_core/README.md,
# automodel/README.md), and the PyTorch every baseline environment runs.
readonly bridge_revision="6dea15d7fd499975ebf91fbeaf8c2b740a73d679"
readonly megatron_core_revision="ce852bfb93f364f64a556602fc71a5d9ed2c377e"
readonly automodel_revision="b203f3f9f9f3140c660942a088ff36bbc3a5e856"
# NVIDIA Resiliency Extension v0.7.0, the version Megatron-Core pins.
readonly resiliency_revision="ebf0352a82ca5699148ea10bec2b8eedd121e382"
# APEX, whose CUDA extensions NVIDIA's PyTorch containers ship.
readonly apex_revision="575968bc1f9127ccd61003a681472a83af4ff1a1"
# AutoModel's optional kernels (its dev group), at the revisions its
# [tool.uv.sources] pins: Cut Cross-Entropy for FusedLinearCrossEntropy, and
# Liger, which its Hugging Face models take by default.
readonly cce_revision="87a86aba72cfd2f0d8abecaf81c13c4528ea07d8"
readonly liger_revision="1f1a4b8d6a6c3c1ddb3573a46480c41cef25bde6"
# The attention and convolution kernels AutoModel's defaults and Transformer
# Engine take where they import. FlashAttention 3 (compute capability 9.0)
# is PyTorch's CUDA 13 build, as mlops installs it. FlashAttention 2 publishes
# no wheel for PyTorch 2.13, so it is a third-party build
# (mjun0812/flash-attention-prebuild-wheels), pinned by its digest, at the
# newest version Transformer Engine accepts (below 2.8.4). causal-conv1d
# (v1.7.0, the revision below) publishes none for PyTorch 2.13 either; it is
# Astral's build of that revision for PyTorch 2.13 and CUDA 13.0
# (astral-sh-build/build-causal-conv1d), pinned by its digest, compiled for
# the architectures its setup.py lists, and is built from its source on a
# platform that build does not cover.
readonly causal_conv1d_revision="cd81f0413cad2fc1e6f17e785ac39f59aae690cd"
readonly causal_conv1d_x86_64="https://github.com/astral-sh-build/build-causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu.13.0.torch.2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
readonly causal_conv1d_x86_64_sha256="b8b7d03b1cafb306f9e89d8d195ece8c296417d8be00b55c4b809b018f77a99b"
readonly causal_conv1d_aarch64="https://github.com/astral-sh-build/build-causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu.13.0.torch.2.13-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"
readonly causal_conv1d_aarch64_sha256="07a5da6040fa8ab211990880d0b4be67628e0201d4a7bb29bf72babaa877546c"
readonly flash_attn_3_x86_64="https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
readonly flash_attn_3_x86_64_sha256="44a019ca428d0866fc73ca49b8af02eb4b6a37ae8a1a6b1a07917c383a187fe6"
readonly flash_attn_3_aarch64="https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_34_aarch64.whl"
readonly flash_attn_3_aarch64_sha256="39ec3c28580b547d1721be227dc9803113fc54a30b95cc697e8061db2fc7bb5b"
readonly flash_attn_2_x86_64="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3%2Bcu130torch2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
readonly flash_attn_2_x86_64_sha256="48181bc891c9907812403656361af7a9bf0666a057ae67b3791ef74745814207"
readonly torch_version="2.13.0"
# Where 9.0 or 10.0 is a target, NCCL at the release Transformer Engine's
# NCCL expert-parallel library (Megatron's ncclep dispatcher) is validated
# against, in place of the 2.29.7 PyTorch 2.13's wheel pins: a newer release
# of the same library, whose interface PyTorch's build keeps working with.
readonly nccl_version="2.30.7"

usage() {
  cat <<'EOF'
Usage: baselines/scripts/setup_nvidia_training.sh [--env NAME | --python PATH]
         [--sources DIR] [--cuda-archs LIST] [--jobs N]

Create the environment the recipe harnesses run in: Megatron-Core and
Megatron Bridge, and NeMo AutoModel, at their pinned revisions, with the
Transformer Engine revision Bridge pins built against PyTorch 2.13. It is
separate from the `baselines` environment: these revisions pin transformers,
cuDNN Frontend and quack at versions mlops's stack does not take, so mlops
is not installed here and the qualified harnesses' environment is left
unchanged. Run without options it needs nothing else, and compiles for the
GPUs the machine shows, or for the list --cuda-archs names. By default a
Conda environment named `nvidia-training` is created with Python 3.12
(--python installs into an existing interpreter instead).

--sources is where the pinned checkouts are cloned, or found. By default it
is src/ inside the environment, beside its bin/ and lib/: Megatron-Core,
Bridge and AutoModel are installed editable, so the environment imports them
from their checkouts, and removing the environment removes them. The
checkouts are Megatron-Bridge with its Megatron-LM submodule, Automodel,
nvidia-resiliency-ext, which is built from its v0.7.0 source because its
wheels since 0.5.0 need glibc 2.39 and Bridge imports it as it loads, and
APEX, whose CUDA extensions are built because NVIDIA's containers ship them:
Megatron-Core's own output layer fuses its weight-gradient accumulation
through APEX's (Transformer Engine fuses its layers' itself), and Bridge
enables the fusion whenever Transformer Engine is present.

The versions of everything the pins leave open come from NVIDIA's own lock
files at these revisions (nvidia_versions.py beside this script): Bridge's
uv.lock, and AutoModel's for what Bridge's does not pin or AutoModel's
requirements exclude, except where PyTorch 2.13 pins its own (triton and its
CUDA libraries), since the locks are for other PyTorch releases. The
resolution prefers these versions and departs from one only where a
requirement forces it (the two locks, made at different times, disagree on
most packages they share); a package it moves from Bridge's version is
preferred at AutoModel's, and what still departs is listed in
<prefix>/environment.departures. A pre-release is installed only where a
lock pins it. NCCL is PyTorch's own (2.29.7) unless the list holds 9.0 or
10.0: then Transformer Engine is built with its NCCL expert-parallel library
(Megatron's ncclep dispatcher), with NCCL 2.30.7, the release that library
is validated against, over PyTorch's pin (a newer release of the same
library and interface), laid out in <prefix>/nccl-home for that build; it
compiles some kernels when first used, with the environment's own nvcc. A
Transformer Engine build that fails with that library is done again without
it, and later runs do not try it again until PyTorch, the toolkit or the
list changes.

Where the list holds 9.0 or 10.0, DeepEP is built for those targets, as
Megatron Bridge's CI image builds it, for Megatron's deepep and hybridep
dispatchers: at the commit that image pins (docker/Dockerfile.ci in Bridge's
checkout), with its patch, against the NVSHMEM release it installs over
PyTorch's pin. HybridEP dispatches across nodes, through DOCA GPUNetIO built
from DeepEP's NCCL submodule against the machine's rdma-core, whose headers
and libraries are linked into <prefix>/rdma-core; on a machine without them
it dispatches within a node. HybridEP compiles its kernels when first used,
with the environment's own nvcc. A DeepEP build that fails stops nothing
else: the checks and the record say why.

The compilers are the environment's own: NVIDIA's CUDA toolkit wheels at the
release PyTorch's CUDA wheels come from, laid out in <prefix>/cuda-home,
which holds the toolkit alone, and the cuDNN and NCCL headers of PyTorch's
wheels on CPATH. No toolkit needs to be installed, every machine compiles
with the same one, and no build takes a CUDA or NCCL the machine has.
Transformer Engine and the extensions built through PyTorch (APEX's) are
compiled for the architectures of the GPUs this machine shows (86 for an RTX
3090), or for --cuda-archs's list, such as "90" or "80;90", where the
machine that builds is not the one that trains (a cluster's login node
building for its compute nodes); with no GPU visible and no list, each build
takes its own (Transformer Engine's, and PyTorch's for APEX).
fast-hadamard-transform compiles a fixed list of its own, and causal-conv1d
is a wheel built for the list its setup.py names (or, on a platform without
one, built here for that list). When PyTorch, the toolkit or the list
changes, what is compiled here is rebuilt rather than taken from uv's cache,
which tells builds apart by their source alone. --jobs caps the parallel
compilations; by default one per CPU the process may use, at most one per 6
GiB of the memory it may take, a cgroup's limits (a cluster's login node, a
container) counted. uv is the one on PATH or in ~/.local/bin, or is
installed into the environment.

The checks at the end run on the machine's GPU when it has one and report
what they find without changing what is installed. The resolved
environment, with the revisions, the toolkit and the architectures, is
recorded inside the environment, in <prefix>/environment.lock.json.
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
    --sources)
      [[ $# -ge 2 ]] || { echo "--sources requires a directory" >&2; exit 2; }
      sources="$2"
      shift 2
      ;;
    --cuda-archs)
      [[ $# -ge 2 ]] || { echo "--cuda-archs requires a list" >&2; exit 2; }
      cuda_archs="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs requires a count" >&2; exit 2; }
      jobs="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Nothing the caller's shell sets (a CUDA or cuDNN module, an MPI build of
# Transformer Engine, architectures, compiler flags or a package's build
# switches of its own) changes what is built: the variables the builds read
# are this script's. Della's CUDA module exports LDFLAGS with its own
# toolkit's libraries, which a makefile appends to.
unset CUDA_HOME CUDA_PATH CUDNN_PATH CUDNN_HOME NCCL_HOME MPI_HOME NVSHMEM_HOME \
  CUBLASMP_HOME CUSOLVERMP_HOME XLA_HOME TORCH_CUDA_ARCH_LIST \
  CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CUDAFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS \
  NVCC_CCBIN NVCC_THREADS CUDACXX CUDAHOSTCXX CUDAToolkit_ROOT CMAKE_CUDA_COMPILER \
  CMAKE_CUDA_HOST_COMPILER CMAKE_CUDA_ARCHITECTURES NVSHMEM_DIR RDMA_CORE_HOME \
  HYBRID_EP_MULTINODE DISABLE_SM90_FEATURES USE_NIXL NIXL_HOME DOCA_HOME UCX_HOME ONLY_FUNCS
for variable in $(compgen -v NVTE_) $(compgen -v CAUSAL_CONV1D_) \
  $(compgen -v FAST_HADAMARD_TRANSFORM_) $(compgen -v APEX_); do
  unset "${variable}"
done
# A search path the caller's shell sets (a CUDA module's LIBRARY_PATH or
# CPATH) keeps only the directories that hold no CUDA toolkit, so that a
# link or an include takes the environment's toolkit and not the machine's:
# della's module put CUDA 13.1's libcudadevrt.a before it, which the 13.0
# linker refuses. Directories of other libraries, a driver's among them,
# stay.
holds_toolkit() {
  compgen -G "$1/libcudart.so*" >/dev/null || [[ -e "$1/libcudadevrt.a" || -e "$1/cuda_runtime.h" ]]
}
for variable in LIBRARY_PATH LD_LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
  [[ -n "${!variable:-}" ]] || continue
  kept=()
  IFS=: read -r -a entries <<< "${!variable}"
  for entry in "${entries[@]}"; do
    if [[ -n "${entry}" && -d "${entry}" ]] && ! holds_toolkit "${entry}"; then
      kept+=("${entry}")
    fi
  done
  if ((${#kept[@]})); then
    export "${variable}=$(IFS=:; echo "${kept[*]}")"
  else
    unset "${variable}"
  fi
done

if [[ -z "${python_executable}" ]]; then
  conda_executable="$(command -v conda || true)"
  for candidate in "${CONDA_EXE:-}" "${HOME}/miniconda3/bin/conda" \
    "${HOME}/miniforge3/bin/conda" "${HOME}/anaconda3/bin/conda"; do
    [[ -n "${conda_executable}" ]] && break
    [[ -n "${candidate}" && -x "${candidate}" ]] && conda_executable="${candidate}"
  done
  if [[ -z "${conda_executable}" ]]; then
    echo "setup needs conda to create the ${environment_name} environment, or --python" >&2
    exit 1
  fi
  # The environment where conda keeps it: under the base's envs/, or where
  # this user's environments live (~/.conda/envs beside a shared base).
  environment_python() {
    "${conda_executable}" env list | awk -v name="${environment_name}" \
      '$1 == name { print $NF "/bin/python"; exit }'
  }
  python_executable="$(environment_python)"
  if [[ -n "${python_executable}" && -x "${python_executable}" ]]; then
    echo "[0/8] Conda environment ${environment_name} exists; installing into it"
  else
    echo "[0/8] Creating Conda environment ${environment_name} with Python 3.12"
    "${conda_executable}" create --yes --quiet --name "${environment_name}" python=3.12
    python_executable="$(environment_python)"
  fi
fi
if [[ ! -x "${python_executable}" ]]; then
  echo "Python interpreter is not executable: ${python_executable}" >&2
  exit 1
fi
environment_prefix="$("${python_executable}" -c 'import sys; print(sys.prefix)')"
sources="${sources:-${environment_prefix}/src}"
uv_executable="$(command -v uv || true)"
if [[ -z "${uv_executable}" && -x "${HOME}/.local/bin/uv" ]]; then
  uv_executable="${HOME}/.local/bin/uv"
fi
if [[ -z "${uv_executable}" ]]; then
  "${python_executable}" -m pip install --quiet uv
  uv_executable="$(dirname "${python_executable}")/uv"
fi
if [[ -z "${jobs}" ]]; then
  # One compilation per CPU this process may use, at most one per 6 GiB of
  # the memory it may take; a cgroup's limits (a cluster's login node, a
  # container) count as well as the machine's.
  jobs="$("${python_executable}" - <<'PY'
import os


def read(path):
    try:
        with open(path) as handle:
            return handle.read().split()
    except OSError:
        return None


cpus = len(os.sched_getaffinity(0))
memory = next(
    int(line.split()[1]) * 1024 for line in open("/proc/meminfo") if line.startswith("MemAvailable:")
)
for line in open("/proc/self/cgroup"):
    hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
    if hierarchy == "0":
        # cgroup v2: the limits of this group and of every group above it.
        directory = "/sys/fs/cgroup" + path.rstrip("/")
        while directory.startswith("/sys/fs/cgroup"):
            quota = read(f"{directory}/cpu.max")
            if quota and quota[0] != "max":
                cpus = min(cpus, max(1, int(quota[0]) // int(quota[1])))
            limit, used = read(f"{directory}/memory.max"), read(f"{directory}/memory.current")
            if limit and used and limit[0] != "max":
                memory = min(memory, int(limit[0]) - int(used[0]))
            directory = os.path.dirname(directory)
    elif "cpu" in controllers.split(","):
        for root in ("/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpu"):
            quota = read(f"{root}{path}/cpu.cfs_quota_us")
            period = read(f"{root}{path}/cpu.cfs_period_us")
            if quota and period and int(quota[0]) > 0:
                cpus = min(cpus, max(1, int(quota[0]) // int(period[0])))
                break
    elif "memory" in controllers.split(","):
        limit = read(f"/sys/fs/cgroup/memory{path}/memory.limit_in_bytes")
        used = read(f"/sys/fs/cgroup/memory{path}/memory.usage_in_bytes")
        if limit and used:
            memory = min(memory, int(limit[0]) - int(used[0]))
print(max(1, min(cpus, memory // (6 << 30))))
PY
)"
fi

echo "[1/8] The pinned sources under ${sources}"
mkdir -p "${sources}"
checkout() {
  local url="$1" directory="$2" revision="$3"
  if [[ ! -d "${directory}/.git" ]]; then
    git clone --quiet "${url}" "${directory}"
  fi
  git -C "${directory}" fetch --quiet origin "${revision}" 2>/dev/null || true
  git -C "${directory}" checkout --quiet "${revision}"
}
checkout https://github.com/NVIDIA-NeMo/Megatron-Bridge.git \
  "${sources}/Megatron-Bridge" "${bridge_revision}"
git -C "${sources}/Megatron-Bridge" submodule update --init --quiet 3rdparty/Megatron-LM
megatron_lm="${sources}/Megatron-Bridge/3rdparty/Megatron-LM"
if [[ "$(git -C "${megatron_lm}" rev-parse HEAD)" != "${megatron_core_revision}" ]]; then
  echo "Megatron-LM is not at ${megatron_core_revision}" >&2
  exit 1
fi
checkout https://github.com/NVIDIA-NeMo/Automodel.git \
  "${sources}/Automodel" "${automodel_revision}"
checkout https://github.com/NVIDIA/nvidia-resiliency-ext.git \
  "${sources}/nvidia-resiliency-ext" "${resiliency_revision}"
checkout https://github.com/NVIDIA/apex.git "${sources}/apex" "${apex_revision}"

echo "[2/8] PyTorch ${torch_version}, its CUDA toolkit and the build tools"
# The architectures of the GPUs this machine shows, unless --cuda-archs
# names others (a login node building for its compute nodes); with neither,
# each build's own list: Transformer Engine's, and PyTorch's for APEX. They
# are read from the driver, before PyTorch is installed, since what DeepEP
# needs installed follows from them.
if [[ -z "${cuda_archs}" ]]; then
  cuda_archs="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d ' .' | sort -u | paste -sd ';')" || cuda_archs=""
fi
# The list's targets of 9.0 and 10.0 (both, where no list is named, as
# Transformer Engine's own list holds them) get the expert-parallel
# libraries: Transformer Engine's NCCL EP (Megatron's ncclep dispatcher) and
# DeepEP (Megatron's and AutoModel's deepep and hybridep), the latter at the
# commit Bridge's CI image builds, against the NVSHMEM release it installs.
expert_targets="$(tr ';' '\n' <<< "${cuda_archs:-90;100}" | grep -E '^(90|100)[a-z]?$' \
  | paste -sd ';')" || expert_targets=""
if [[ -n "${expert_targets}" ]]; then
  nccl_ep=1
fi
deepep_archs="$(tr ';' '\n' <<< "${expert_targets}" \
  | sed -E 's/^([0-9]+)([0-9])([a-z]?)$/\1.\2\3/' | paste -sd ';')" || deepep_archs=""
bridge_image="${sources}/Megatron-Bridge/docker/Dockerfile.ci"
deepep_commit="$(sed -n 's/^ARG DEEPEP_COMMIT=\([0-9a-f]*\).*/\1/p' "${bridge_image}" 2>/dev/null)" \
  || deepep_commit=""
deepep_nvshmem="$(grep -o 'nvidia-nvshmem-cu13==[0-9.]*' "${bridge_image}" 2>/dev/null | head -1)" \
  || deepep_nvshmem=""
if [[ -z "${deepep_commit}" || -z "${deepep_nvshmem}" ]]; then
  deepep_archs=""
fi
# PyTorch's index for its CUDA 13.0 build is an extra index, and the index
# strategy lets every other package resolve from PyPI.
readonly torch_index="https://download.pytorch.org/whl/cu130"
preferences="$(mktemp)"
pins="$(mktemp)"
overrides="$(mktemp)"
requirements="$(mktemp)"
resolved="$(mktemp)"
wheels="$(mktemp -d)"
trap 'rm -rf "${preferences}" "${pins}" "${overrides}" "${requirements}" "${resolved}" "${wheels}"' EXIT
# Wheels built here from pinned checkouts, which the resolution and every
# install find beside the indexes.
local_wheels="${wheels}/local"
mkdir -p "${local_wheels}"
# What every install takes: the overrides, and the wheels built here.
installs=(--override "${overrides}" --find-links "${local_wheels}")
# What replaces a requirement wherever it appears: where Transformer Engine
# builds its NCCL EP library, NCCL at the release above over PyTorch's pin;
# where DeepEP is built, NVSHMEM at
# the release Bridge's image builds it against; and, of Bridge's overrides,
# the fast-hadamard-transform pin resolution needs (PyPI has no 1.1.0).
{
  if ((nccl_ep)); then
    echo "nvidia-nccl-cu13==${nccl_version}"
  fi
  if [[ -n "${deepep_archs}" ]]; then
    echo "${deepep_nvshmem}"
  fi
  echo "fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0"
} > "${overrides}"
"${uv_executable}" pip install --python "${python_executable}" \
  --extra-index-url "${torch_index}" --index-strategy unsafe-best-match \
  "${installs[@]}" \
  "torch==${torch_version}" ninja packaging pybind11 setuptools wheel
torch_cuda="$("${python_executable}" -c 'import torch; print(torch.version.cuda)')"
torch_build="$("${python_executable}" -c 'import torch; print(torch.__version__)')"
# The compilers of the cuda-toolkit release PyTorch's CUDA wheels come from,
# with the headers the builds include beside what PyTorch's CUDA wheels
# bring: the compiler's own (crt, nvvm), CCCL's, the profiler's
# (cuda_profiler_api.h) and NVML's.
toolkit_requirement="$("${python_executable}" - <<'PY'
from importlib.metadata import requires

from packaging.requirements import Requirement

for text in requires("torch") or ():
    requirement = Requirement(text)
    if requirement.name == "cuda-toolkit":
        print(f"cuda-toolkit[nvcc,cccl,crt,nvvm,profiler,nvml]{requirement.specifier}")
        break
PY
)"
if [[ -z "${toolkit_requirement}" ]]; then
  echo "PyTorch ${torch_version} names no cuda-toolkit release to compile with" >&2
  exit 1
fi
"${uv_executable}" pip install --python "${python_executable}" \
  --extra-index-url "${torch_index}" --index-strategy unsafe-best-match \
  "${toolkit_requirement}"
# The wheels keep the toolkit in nvidia/cu<major> and each library under its
# versioned name only; cuda-home lays it out as builds expect it, with lib64
# and the plain names a link takes (-lcudart) beside the versioned ones. It
# is a tree of links, directories made and files linked, so that a build
# that walks it (nvidia-resiliency-ext's search for CUPTI) sees every file.
toolkit="$("${python_executable}" -c "import nvidia, os; print(next(os.path.join(p, 'cu${torch_cuda%%.*}') for p in nvidia.__path__ if os.path.isdir(os.path.join(p, 'cu${torch_cuda%%.*}'))))")"
cuda_home="${environment_prefix}/cuda-home"
rm -rf "${cuda_home}"
mkdir -p "${cuda_home}"
for part in bin include nvvm; do
  cp -rs "${toolkit}/${part}" "${cuda_home}/${part}"
done
cp -rs "${toolkit}/lib" "${cuda_home}/lib64"
for entry in "${cuda_home}/lib64/"*.so.*; do
  name="$(basename "${entry}")"
  plain="${name%%.so.*}.so"
  [[ -e "${cuda_home}/lib64/${plain}" ]] || ln -s "${name}" "${cuda_home}/lib64/${plain}"
done
ln -s lib64 "${cuda_home}/lib"
# The toolkit holds no NCCL, so a build that compiles a NCCL of its own
# (DeepEP's) sees only its own headers. Where Transformer Engine builds its
# NCCL EP library, NCCL is laid out in nccl-home, NCCL_HOME for that build,
# which links -lnccl; without it, that build takes the first NCCL the
# machine has.
nccl="$("${python_executable}" -c 'import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])')"
nccl_home="${environment_prefix}/nccl-home"
rm -rf "${nccl_home}"
if ((nccl_ep)); then
  mkdir -p "${nccl_home}"
  cp -rs "${nccl}/include" "${nccl_home}/include"
  cp -rs "${nccl}/lib" "${nccl_home}/lib"
  ln -s libnccl.so.2 "${nccl_home}/lib/libnccl.so"
fi
nvcc_release="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
if [[ "${nvcc_release}" != "${torch_cuda}" ]]; then
  echo "the toolkit wheels are CUDA ${nvcc_release}; PyTorch's CUDA is ${torch_cuda}" >&2
  exit 1
fi
wheel_headers="$("${python_executable}" -c 'import os, nvidia; print(":".join(os.path.join(nvidia.__path__[0], name, "include") for name in ("cudnn", "nccl")))')"
torch_archs="$("${python_executable}" -c 'import torch; archs = torch.cuda.get_arch_list() or torch._C._cuda_getArchFlags().split(); print(";".join(arch.removeprefix("sm_") for arch in archs if arch.startswith("sm_")))')"
engine_archs="${cuda_archs}"
extension_archs="${cuda_archs:-${torch_archs}}"
torch_arch_list="$(tr ';' '\n' <<< "${extension_archs}" | sed -E 's/^([0-9]+)([0-9])([a-z]?)$/\1.\2\3/' | paste -sd ';')"
# The builds compile and link against the environment's headers and its
# toolkit's libraries, where a link a compiler driver does not make itself
# (a shared library g++ links) looks for libcudart_static and libcudadevrt.
build_environment=(
  CUDA_HOME="${cuda_home}" CUDA_PATH="${cuda_home}" PATH="${cuda_home}/bin:${PATH}"
  CPATH="${wheel_headers}" LIBRARY_PATH="${cuda_home}/lib64" MAX_JOBS="${jobs}"
  TORCH_CUDA_ARCH_LIST="${torch_arch_list}"
)
if ((nccl_ep)); then
  build_environment+=(NCCL_HOME="${nccl_home}")
fi
if [[ -n "${engine_archs}" ]]; then
  build_environment+=(NVTE_CUDA_ARCHS="${engine_archs}")
fi
echo "CUDA ${torch_cuda} toolkit at ${cuda_home}; Transformer Engine for" \
  "${engine_archs:-its own architectures}, extensions for ${extension_archs}; ${jobs} jobs"
# A package compiled here (Transformer Engine, fast-hadamard-transform, APEX,
# DeepEP, and causal-conv1d where it is built) against this PyTorch, with this
# toolkit, for these architectures is kept; any other, in this environment or
# uv's cache, is rebuilt (uv keys a build by its source alone).
readonly build_identity="torch==${torch_build} ${toolkit_requirement} archs=${engine_archs:-own}/${extension_archs}"
readonly builds_record="${environment_prefix}/.cuda-builds"
# A package's identity is the build's, or the one the second argument names.
rebuild_flags() {
  if ! grep -qxF "$1 ${2:-${build_identity}}" "${builds_record}" 2>/dev/null; then
    "${uv_executable}" cache clean --quiet "$1" >&2 || true
    printf '%s\n' --reinstall-package "$1"
  fi
}
record_build() {
  { grep -v "^$1 " "${builds_record}" 2>/dev/null || true
    echo "$1 ${2:-${build_identity}}"; } > "${builds_record}.new"
  mv "${builds_record}.new" "${builds_record}"
}
# The versions NVIDIA's locks pin, to prefer, except PyTorch's own pins, the
# packages this script installs from its own pins and those an override
# sets (NCCL, NVSHMEM), since a preference is a constraint no override
# lifts; then, at those
# versions, what the packages built here without isolation declare they
# build with:
# Transformer Engine's setuptools, cmake, wheel, pybind11, ninja, pip and
# cuDNN Frontend (its JAX build's jax and flax aside), Megatron-Core's
# packaging, the rest's setuptools. Bridge declares setuptools below 80 and
# Megatron-Core and AutoModel at least 80, so one build of the three runs
# beside a setuptools the other declares against; it is the locks'.
readonly own_pins=(--keep torch --keep nvidia-resiliency-ext --keep apex
  --keep causal-conv1d --keep flash-attn --keep flash-attn-3 --keep nvidia-nccl-cu13
  --keep nvidia-nvshmem-cu13)
"${python_executable}" "${script_root}/nvidia_versions.py" prefer \
  "${sources}/Megatron-Bridge/uv.lock" "${sources}/Automodel/uv.lock" \
  "${own_pins[@]}" > "${preferences}"
echo "torch==${torch_version}" > "${pins}"
"${uv_executable}" pip install --python "${python_executable}" \
  --constraint "${preferences}" --constraint "${pins}" "${installs[@]}" \
  setuptools wheel ninja "pybind11[global]" pip packaging cmake nvidia-cudnn-frontend

echo "[3/8] NVIDIA Resiliency Extension from its checkout"
# Built into a wheel of its own: PyPI's wheels of this release need glibc
# 2.39, so a resolution that saw only the indexes would find none for an
# older system.
rm -rf "${sources}/nvidia-resiliency-ext/build"
env "${build_environment[@]}" "${uv_executable}" build --quiet --wheel \
  --python "${python_executable}" --out-dir "${local_wheels}" \
  "${sources}/nvidia-resiliency-ext"
"${uv_executable}" pip install --python "${python_executable}" \
  --constraint "${preferences}" --constraint "${pins}" "${installs[@]}" \
  "${local_wheels}"/nvidia_resiliency_ext-*.whl
resiliency="$("${python_executable}" -c 'from importlib.metadata import version; print(version("nvidia-resiliency-ext"))')"
echo "nvidia-resiliency-ext==${resiliency}" >> "${pins}"

echo "[4/8] Megatron-Core, Megatron Bridge and NeMo AutoModel from their checkouts, and AutoModel's optional kernels"
# One resolution for the three, holding PyTorch where step 2 put it.
# AutoModel's transformers pin lies inside Bridge's range and
# Megatron-Core's cuDNN Frontend pin inside AutoModel's. uv applies a
# checkout's [tool.uv.sources] when it installs it, so Transformer Engine
# comes from the revision Bridge pins and is built here, but not the rest
# of its [tool.uv] table: of Bridge's overrides, resolution needs its
# fast-hadamard-transform pin (PyPI has no 1.1.0), and of its settings
# NVIDIA's index (nvidia-modelopt); the versions are the locks' (step 2).
# The two projects' other overrides and constraints are security floors,
# some contradicting each other (cryptography below 47 for Bridge, at least
# 50 for AutoModel), and their locks already hold what each chose.
# Bridge requires Megatron-Core's inference extra, whose torch-memory-saver
# builds for one CUDA major release (TMS_CUDA_MAJOR).
# Transformer Engine is one package here; the separate core and PyTorch
# packages of a PyPI install, which an older environment may hold, shadow it
# (its import checks for a consistent PyPI install), so they go, and
# Transformer Engine with them, to be installed whole.
split_engine="$("${python_executable}" -c 'import importlib.metadata as m; print(" ".join(sorted({d.metadata["Name"] for d in m.distributions() if d.metadata["Name"].replace("-", "_").lower() in ("transformer_engine_torch", "transformer_engine_cu12", "transformer_engine_cu13")})))')"
if [[ -n "${split_engine}" ]]; then
  # shellcheck disable=SC2086
  "${uv_executable}" pip uninstall --python "${python_executable}" \
    ${split_engine} transformer-engine
fi
# The resolution prefers the locks' versions and departs from one only where
# a requirement forces it, the same on every machine: a package it moves from
# Bridge's version is preferred at AutoModel's in a second resolution, and
# what still departs is recorded. The result is installed exactly. The
# toolkit the environment compiles with is among the requirements, so no
# resolution moves a package cuda-home links to: one over an environment
# already built prefers otherwise, and moved CCCL past the compiler's.
printf '%s\n' "-e ${megatron_lm}[mlm]" "-e ${sources}/Megatron-Bridge" \
  "-e ${sources}/Automodel" "${toolkit_requirement}" > "${requirements}"
# Where 9.0 or 10.0 is a target Transformer Engine builds its NCCL
# expert-parallel library (Megatron's ncclep dispatcher) for its targets from
# 9.0, against the NCCL in nccl-home. The library is built inside Transformer
# Engine's build, its resolution's included, so a build that fails with it
# is done again without it (NVTE_WITH_NCCL_EP=0); Transformer Engine's record
# then says so, and later runs do not try it again until PyTorch, the
# toolkit or the list changes.
engine_without="${build_identity} without-nccl-ep"
engine_identity="${build_identity}"
nccl_ep_state="not built: no target of 9.0 or 10.0"
if ((nccl_ep)); then
  nccl_ep_state="built"
  if grep -qxF "transformer-engine ${engine_without}" "${builds_record}" 2>/dev/null; then
    nccl_ep=0
    engine_identity="${engine_without}"
    nccl_ep_state="not built: it failed before with this PyTorch, toolkit and list"
    echo "Transformer Engine's NCCL expert-parallel library ${nccl_ep_state}"
  fi
fi
# Transformer Engine and fast-hadamard-transform build inside uv's checkouts
# of them (Transformer Engine's NCCL EP library too), which the cache keeps;
# a build there reuses the objects it finds whenever their command lines are
# unchanged, as across toolkits at one path. A rebuild starts with none.
clean_engine_objects() {
  find "$("${uv_executable}" cache dir)" -maxdepth 7 -type d \
    \( -path '*/checkouts/*/*/build' -o -path '*/checkouts/*/3rdparty/nccl-extensions/build' \) \
    -prune -exec rm -rf {} + 2>/dev/null || true
}
engine_environment_for() {
  engine_environment=(
    NVTE_FRAMEWORK=pytorch TMS_CUDA_MAJOR="${torch_cuda%%.*}" NVTE_WITH_NCCL_EP="$1"
  )
  if (($1)); then
    # Transformer Engine's CMake looks for libnccl with no hint, which finds
    # the machine's own NCCL, or none; it is told the environment's (its
    # setup.py passes NVTE_CMAKE_EXTRA_ARGS to CMake as they are).
    engine_environment+=(NVTE_CMAKE_EXTRA_ARGS="-DNCCL_LIB=${nccl}/lib/libnccl.so.2")
  fi
}
resolve() {
  cp "${preferences}" "${resolved}"
  env "${build_environment[@]}" "${engine_environment[@]}" \
    "${uv_executable}" pip compile --quiet --python "${python_executable}" \
    --extra-index-url "${torch_index}" --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com --no-build-isolation \
    --constraint "${pins}" "${installs[@]}" \
    --output-file "${resolved}" "${requirements}"
}
departures="${environment_prefix}/environment.departures"
# The resolutions, their departures and the install, exactly as resolved.
build_engine() {
  resolve
  mapfile -t moved < <("${python_executable}" "${script_root}/nvidia_versions.py" \
    departures "${preferences}" "${resolved}" | awk '{print "--automodel"; print $1}')
  if ((${#moved[@]})); then
    "${python_executable}" "${script_root}/nvidia_versions.py" prefer \
      "${sources}/Megatron-Bridge/uv.lock" "${sources}/Automodel/uv.lock" \
      "${own_pins[@]}" "${moved[@]}" > "${preferences}"
    resolve
  fi
  "${python_executable}" "${script_root}/nvidia_versions.py" departures \
    "${preferences}" "${resolved}" > "${departures}"
  echo "$(wc -l < "${departures}") versions depart from NVIDIA's locks:" \
    "$(awk '{printf "%s %s->%s; ", $1, $2, $3}' "${departures}")"
  env "${build_environment[@]}" "${engine_environment[@]}" \
    "${uv_executable}" pip install --python "${python_executable}" \
    --extra-index-url "${torch_index}" --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com "${engine_rebuild[@]}" \
    --no-build-isolation --constraint "${pins}" "${installs[@]}" \
    --requirement "${resolved}"
}
mapfile -t engine_rebuild < <(rebuild_flags transformer-engine "${engine_identity}"
  rebuild_flags fast-hadamard-transform)
if ((${#engine_rebuild[@]})); then
  clean_engine_objects
fi
engine_environment_for "${nccl_ep}"
set +e
( set -e; build_engine )
engine_status=$?
set -e
if ((engine_status != 0)); then
  ((nccl_ep)) || exit "${engine_status}"
  nccl_ep=0
  engine_identity="${engine_without}"
  nccl_ep_state="failed (status ${engine_status}); Transformer Engine is built without it"
  echo "Transformer Engine's NCCL expert-parallel library ${nccl_ep_state}"
  clean_engine_objects
  mapfile -t engine_rebuild < <(rebuild_flags transformer-engine "${engine_identity}"
    rebuild_flags fast-hadamard-transform)
  engine_environment_for 0
  build_engine
fi
record_build transformer-engine "${engine_identity}"
record_build fast-hadamard-transform
if find "${cuda_home}" -xtype l | grep -q .; then
  echo "an install moved the toolkit under ${cuda_home}: it links to files that are gone" >&2
  exit 1
fi
# Every later install holds what the resolution chose.
grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' "${resolved}" >> "${pins}"
# AutoModel's optional kernels, holding PyTorch where it is.
"${uv_executable}" pip install --python "${python_executable}" \
  --constraint "${pins}" "${installs[@]}" \
  "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@${cce_revision}" \
  "liger-kernel @ git+https://github.com/linkedin/Liger-Kernel.git@${liger_revision}"
# The FlashAttention wheels this platform (its processor and Python) has,
# whatever its GPU: each is checked against its digest before it is
# installed, and the attention dispatchers take one only on a GPU it runs
# on (FlashAttention 3 at compute capability 9.0). A platform without one
# keeps PyTorch's and cuDNN's attention.
# The wheel's path, once its digest matches; a command substitution does
# not stop on a failure under errexit, so each failure exits it.
fetch_wheel() {
  local name
  name="$(basename "$1" | sed 's/%2B/+/g')"
  curl -fsSL --retry 3 -o "${wheels}/${name}" "$1" || exit 1
  echo "$2  ${wheels}/${name}" | sha256sum --check --quiet || exit 1
  echo "${wheels}/${name}"
}
machine="$(uname -m)"
python_tag="$("${python_executable}" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
flash_attention=()
if [[ "${torch_cuda%%.*}" == 13 && "${machine}" == x86_64 ]]; then
  wheel="$(fetch_wheel "${flash_attn_3_x86_64}" "${flash_attn_3_x86_64_sha256}")"
  flash_attention+=("${wheel}")
  if [[ "${python_tag}" == cp312 ]]; then
    wheel="$(fetch_wheel "${flash_attn_2_x86_64}" "${flash_attn_2_x86_64_sha256}")"
    flash_attention+=("${wheel}")
  else
    echo "no pinned FlashAttention 2 wheel for ${python_tag}; Hugging Face's attention keeps sdpa"
  fi
elif [[ "${torch_cuda%%.*}" == 13 && "${machine}" == aarch64 ]]; then
  wheel="$(fetch_wheel "${flash_attn_3_aarch64}" "${flash_attn_3_aarch64_sha256}")"
  flash_attention+=("${wheel}")
  echo "no pinned FlashAttention 2 wheel for ${machine}; Hugging Face's attention keeps sdpa"
else
  echo "no pinned FlashAttention wheels for CUDA ${torch_cuda} on ${machine}"
fi
if ((${#flash_attention[@]})); then
  "${uv_executable}" pip install --python "${python_executable}" \
    --constraint "${pins}" "${installs[@]}" "${flash_attention[@]}"
fi
# causal-conv1d's wheel where this platform has one; elsewhere its source,
# built here.
if [[ "${torch_cuda%%.*}" == 13 && "${python_tag}" == cp312 && "${machine}" == x86_64 ]]; then
  causal_conv1d="$(fetch_wheel "${causal_conv1d_x86_64}" "${causal_conv1d_x86_64_sha256}")"
elif [[ "${torch_cuda%%.*}" == 13 && "${python_tag}" == cp312 && "${machine}" == aarch64 ]]; then
  causal_conv1d="$(fetch_wheel "${causal_conv1d_aarch64}" "${causal_conv1d_aarch64_sha256}")"
else
  causal_conv1d="causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@${causal_conv1d_revision}"
fi
mapfile -t conv_rebuild < <(rebuild_flags causal-conv1d)
env "${build_environment[@]}" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  "${uv_executable}" pip install --python "${python_executable}" \
  --no-build-isolation "${conv_rebuild[@]}" --constraint "${pins}" "${installs[@]}" \
  "${causal_conv1d}"
record_build causal-conv1d

echo "[5/8] APEX's C++ and CUDA extensions from its checkout"
mapfile -t apex_rebuild < <(rebuild_flags apex)
if ((${#apex_rebuild[@]})); then
  rm -rf "${sources}/apex/build"
fi
env "${build_environment[@]}" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
  "${uv_executable}" pip install --python "${python_executable}" \
  "${apex_rebuild[@]}" --constraint "${pins}" "${installs[@]}" \
  --no-build-isolation "${sources}/apex"
record_build apex

echo "[6/8] DeepEP for Megatron's deepep and hybridep dispatchers"
# At the commit Bridge's CI image builds, with that image's patch (the CCCL
# include it names, here the environment's) and its settings: HybridEP able
# to dispatch across nodes, through DOCA GPUNetIO from DeepEP's NCCL
# submodule (only the SendRecv kernels generated) against the machine's
# rdma-core, whose headers and libraries are linked into the environment;
# within a node only where the machine has no rdma-core headers. The step
# never stops the setup: what it built, or why not, is reported and recorded.
deepep_state="not built: no target of 9.0 or 10.0"
if [[ -n "${deepep_archs}" ]]; then
  rdma_core="${environment_prefix}/rdma-core"
  multinode=1
  rm -rf "${rdma_core}"
  mkdir -p "${rdma_core}/lib"
  ln -s /usr/include "${rdma_core}/include"
  for library in libibverbs libmlx5; do
    found=""
    for directory in /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu; do
      if [[ -e "${directory}/${library}.so.1" ]]; then
        found="${directory}/${library}.so.1"
        break
      fi
    done
    if [[ -z "${found}" ]]; then
      multinode=0
    else
      ln -s "${found}" "${rdma_core}/lib/${library}.so"
      ln -s "${found}" "${rdma_core}/lib/${library}.so.1"
    fi
  done
  if [[ ! -f /usr/include/infiniband/verbs.h || ! -f /usr/include/infiniband/mlx5dv.h ]]; then
    multinode=0
  fi
  if ((multinode)); then
    reach="across nodes"
  else
    reach="within a node (no rdma-core headers and libraries here)"
  fi
  echo "DeepEP ${deepep_commit} for ${deepep_archs}, HybridEP ${reach}"
  deepep="${sources}/DeepEP"
  mapfile -t deepep_rebuild < <(rebuild_flags deep-ep)
  set +e
  (
    set -euo pipefail
    [[ -d "${deepep}/.git" ]] || git clone --quiet https://github.com/deepseek-ai/DeepEP.git "${deepep}"
    git -C "${deepep}" fetch --quiet origin "${deepep_commit}"
    if ((${#deepep_rebuild[@]})); then
      # A build for another identity, or none: from the clean commit, since
      # the NCCL submodule's make would keep objects of other targets.
      git -C "${deepep}" checkout --quiet --force "${deepep_commit}"
      git -C "${deepep}" submodule update --quiet --init --recursive --force
      git -C "${deepep}" clean -qfdx
      git -C "${deepep}" submodule foreach --quiet --recursive git clean -qfdx
      sed "s|/usr/local/cuda/include/cccl/|${cuda_home}/include/cccl/|" \
        "${sources}/Megatron-Bridge/docker/patches/deepep.patch" | git -C "${deepep}" apply
    fi
    env "${build_environment[@]}" TORCH_CUDA_ARCH_LIST="${deepep_archs}" \
      HYBRID_EP_MULTINODE="${multinode}" RDMA_CORE_HOME="${rdma_core}" ONLY_FUNCS=SendRecv \
      "${uv_executable}" pip install --python "${python_executable}" \
      --no-build-isolation "${deepep_rebuild[@]}" --constraint "${pins}" "${installs[@]}" \
      "${deepep}"
  )
  deepep_status=$?
  set -e
  if ((deepep_status == 0)); then
    record_build deep-ep
    deepep_state="built for ${deepep_archs}, HybridEP ${reach}"
  else
    deepep_state="failed (status ${deepep_status}); Megatron's deepep and hybridep dispatchers are unavailable"
    echo "DeepEP ${deepep_state}; the rest of the environment is built as without it"
  fi
else
  echo "no target of 9.0 or 10.0 (${cuda_archs:-none named}); not built"
fi

echo "[7/8] Verifying the recipe systems import beside PyTorch ${torch_version}"
"${python_executable}" - <<'PY'
import ctypes
from importlib import import_module
from importlib.metadata import version

import torch

release = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if release != (2, 13):
    raise RuntimeError(f"expected PyTorch 2.13, found {torch.__version__}")
for module_name, package_name in (
    ("transformer_engine.pytorch", "transformer_engine"),
    ("megatron.core", "megatron-core"),
    ("megatron.bridge.training.pretrain", "megatron-bridge"),
    ("nemo_automodel.recipes.llm.train_ft", "nemo-automodel"),
    ("transformers", "transformers"),
    ("fused_weight_gradient_mlp_cuda", "apex"),
    ("cut_cross_entropy", "cut-cross-entropy"),
    ("liger_kernel", "liger-kernel"),
    ("causal_conv1d", "causal-conv1d"),
):
    import_module(module_name)
    print(f"{module_name}: {version(package_name)}")

# Importing FlashAttention loads its CUDA extension without running a kernel,
# so on any GPU it shows whether this install is usable; FlashAttention 3
# runs on compute capability 9.0 only, where Transformer Engine and Hugging
# Face's attention take it.
for module_name, package_name in (
    ("flash_attn", "flash-attn"),
    ("flash_attn_interface", "flash-attn-3"),
):
    try:
        import_module(module_name)
    except ImportError as error:
        print(f"{package_name}: not loadable beside this torch, so unused ({error})")
    else:
        print(f"{package_name}: {version(package_name)}")

# What runs on this machine's GPU is reported, not required: the machine
# that builds is not always one the environment trains on.
def report(name, check):
    try:
        check()
    except Exception as error:  # noqa: BLE001 - reported, not raised
        print(f"{name}: does not run on this GPU ({type(error).__name__}: {error})")
    else:
        print(f"{name}: runs on this GPU")


def engine_linear():
    import transformer_engine.pytorch as te

    layer = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
    layer(torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)).sum().backward()
    torch.cuda.synchronize()


def attention(module_name):
    def check():
        function = import_module(module_name).flash_attn_func
        q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        function(q, q, q, causal=True)
        torch.cuda.synchronize()

    return check


def convolution():
    from causal_conv1d import causal_conv1d_fn

    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    causal_conv1d_fn(x, weight, activation="silu").sum().backward()
    torch.cuda.synchronize()


if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda}),"
          f" {torch.cuda.get_device_name(0)} at compute capability {major}.{minor}")
    report("Transformer Engine linear forward and backward", engine_linear)
    report("causal-conv1d forward and backward", convolution)
    if (major, minor) >= (8, 0):
        report("FlashAttention 2", attention("flash_attn"))
    if (major, minor) == (9, 0):
        report("FlashAttention 3", attention("flash_attn_interface"))
else:
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda}): no GPU it can use is"
          " visible here (none, or a driver older than CUDA's), so nothing ran on one")
# The NCCL PyTorch's libraries loaded (the environment's wheel, through
# their rpath), beside the release PyTorch was built against, which is what
# torch.cuda.nccl.version() reports.
code = ctypes.c_int()
ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(code))
built = ".".join(map(str, torch.cuda.nccl.version()))
print(f"NCCL: {code.value // 10000}.{code.value // 100 % 100}.{code.value % 100} loaded"
      f" (PyTorch built against {built})")
# Megatron's and AutoModel's DeepEP dispatchers, as each finds DeepEP.
for system, module_name in (
    ("Megatron", "megatron.core.transformer.moe.fused_a2a"),
    ("AutoModel", "nemo_automodel.components.moe.megatron.fused_a2a"),
):
    try:
        fused_a2a = import_module(module_name)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        print(f"{system}'s DeepEP dispatchers: not importable ({type(error).__name__}: {error})")
        continue
    for backend, present in (("deepep", fused_a2a.HAVE_DEEP_EP), ("hybridep", fused_a2a.HAVE_HYBRIDEP)):
        print(f"{system}'s {backend} dispatcher: {'available' if present else 'unavailable'}")
try:
    import_module("transformer_engine").require_nccl_ep()
except RuntimeError as error:
    print(f"Transformer Engine's NCCL expert-parallel library: unavailable ({error})")
else:
    print("Transformer Engine's NCCL expert-parallel library: available")
PY

echo "[8/8] Recording the environment in ${environment_prefix}/environment.lock.json"
"${python_executable}" - "${environment_prefix}/environment.lock.json" "${cuda_home}" \
  "${toolkit_requirement}" "${engine_archs:-its default}" "${extension_archs}" \
  "${departures}" "${nccl_ep_state}" "${deepep_state}" "${deepep_commit}" "${deepep_archs}" \
  "${bridge_revision}" "${megatron_core_revision}" \
  "${automodel_revision}" "${resiliency_revision}" "${apex_revision}" \
  "${cce_revision}" "${liger_revision}" "${causal_conv1d_revision}" \
  "${flash_attn_2_x86_64}#sha256=${flash_attn_2_x86_64_sha256}" \
  "${flash_attn_3_x86_64}#sha256=${flash_attn_3_x86_64_sha256}" <<'PY'
import json
import platform
import subprocess
import sys
from importlib.metadata import distributions

import torch

(path, cuda_home, toolkit, engine_archs, extension_archs, departures, nccl_ep_state,
 deepep_state, deepep_commit, deepep_archs, bridge, core, automodel, resiliency,
 apex, cce, liger, causal_conv1d, flash_attn_2, flash_attn_3) = sys.argv[1:]
nvcc = subprocess.run(
    [f"{cuda_home}/bin/nvcc", "--version"], capture_output=True, text=True
).stdout.strip().splitlines()[-1]
packages = sorted(
    (dist.metadata["Name"], dist.version) for dist in distributions()
)
lock = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    # The machine that built, which is not always one it trains on.
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "toolkit": {"cuda_home": cuda_home, "nvcc": nvcc, "wheels": toolkit},
    "architectures": {
        "transformer_engine": engine_archs,
        "pytorch_extensions": extension_archs,
    },
    "nccl_expert_parallel": nccl_ep_state,
    "deep_ep": {"state": deepep_state, "commit": deepep_commit, "architectures": deepep_archs},
    "departures_from_nvidia_locks": [line.split() for line in open(departures) if line.strip()],
    "revisions": {
        "Megatron-Bridge": bridge,
        "Megatron-LM": core,
        "Automodel": automodel,
        "nvidia-resiliency-ext": resiliency,
        "apex": apex,
        "cut-cross-entropy": cce,
        "liger-kernel": liger,
        "causal-conv1d": causal_conv1d,
        "flash-attn": flash_attn_2,
        "flash-attn-3": flash_attn_3,
    },
    "packages": {name: version for name, version in packages},
}
with open(path, "w") as handle:
    json.dump(lock, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"{len(packages)} packages recorded")
PY
echo "Environment ready: ${python_executable}"
