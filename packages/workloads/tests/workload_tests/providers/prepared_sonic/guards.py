"""Cold replay refuses external compilation, cache fallback and tuning."""

import os


def forbid_provider_tuning(capability):
    # Quack creates normalization tuner inventories at import. Its ordinary
    # architecture query initializes CUDA before the executor can install the
    # allocator. Select the saved Program's target using Quack's supported
    # import/cross-compilation setting. Executor still validates actual hardware
    # through build_setup after allocator installation.
    if (
        len(capability) != 2
        or any(type(value) is not int or value < 0 for value in capability)
        or capability[0] == 0
    ):
        raise ValueError("saved Program has an invalid CUDA capability")
    architecture = "".join(map(str, capability))
    if os.environ.get("QUACK_ARCH") not in (
        None,
        architecture,
        "sm" + architecture,
        "sm_" + architecture,
    ):
        raise ValueError("QUACK_ARCH differs from the saved Program target")
    os.environ["QUACK_ARCH"] = architecture
    # Registration is part of the external provider contract, not model build.
    # Install its checked patches before refusing further selection work.
    from mlops.quack_runtime._benchmark_scope import install

    install()
    from mlops.quack_runtime import initialize
    from mlops.quack_runtime.autotune import sonicmoe_functional
    from quack.autotuner import Autotuner as QuackAutotuner
    from triton.runtime.autotuner import Autotuner as TritonAutotuner

    initialize()
    sonicmoe_functional()

    def forbidden(*args, **kwargs):
        raise AssertionError("saved phase attempted provider compilation or autotuning")

    QuackAutotuner._bench = forbidden
    QuackAutotuner.check_disk_cache = forbidden
    QuackAutotuner.prune_configs = forbidden
    TritonAutotuner._bench = forbidden
    import importlib

    import triton
    from cutlass import cute
    from cutlass.base_dsl.compiler import CompileCallable, Compiler
    from triton.runtime import build

    triton.compile = triton.compiler.compile = forbidden
    importlib.import_module("triton.compiler.compiler").compile = forbidden
    triton.JITFunction._do_compile = forbidden
    build._build = build.compile_module_from_src = forbidden
    cute.compile = CompileCallable._compile = Compiler.compile = forbidden
