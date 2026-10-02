"""Isolated compiler experiments; never writes installed MoonEP source."""

import inspect
import linecache
import textwrap


def _replace(module, name, source):
    filename = f"<probe_moonep_compat_{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(module))
    exec(compile(source, filename, "exec"), namespace)
    setattr(module, name, namespace[name])


def compiler_options(options):
    from moonep import planning

    source = textwrap.dedent(inspect.getsource(planning._get_compiled))
    marker = "Int32(0), cuda.CUstream(0))"
    assert source.count(marker) == 1
    source = source.replace(marker, f"Int32(0), cuda.CUstream(0), options={options!r})")
    _replace(planning, "_get_compiled", source)


def memory_clobber():
    from moonep import _common

    for name in ("ld_acquire_sys_s32", "ld_acquire_gpu_s32", "atom_add_release_gpu", "red_add_release_sys"):
        source = textwrap.dedent(inspect.getsource(getattr(_common, name)))
        import re
        source, count = re.subn(r'("[=rl,]+)(",)', r'\1,~{memory}\2', source)
        assert count == 1, (name, count)
        _replace(_common, name, source)
