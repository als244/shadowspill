"""Diagnostic backport of three Quack vector helpers; installed files untouched.

This is a feasibility experiment, not a supported runtime compatibility patch.
It imports the current Quack package with the older DSL and replaces only the
three Vector-dependent helpers with their implementations from Quack 0.4.1.
"""

import ast
import importlib.abc
import importlib.machinery
import importlib.util
import linecache
from pathlib import Path
import sys

ROOT = Path('/home/as1669/storage/shadowspill/generic_training_0930/della_1001/moonep-declared-dsl442')
STOCK = Path('/home/as1669/.conda/envs/shadowspill/lib/python3.12/site-packages')


def _function_spans(source):
    return {
        node.name: (min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1, node.end_lineno)
        for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    }


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = (STOCK / 'quack/utils.py').read_text()
        old = (ROOT / 'quack/utils.py').read_text()
        old_lines, lines = old.splitlines(True), source.splitlines(True)
        old_spans, spans = _function_spans(old), _function_spans(source)
        names = ('make_vector', 'f32x2_to_i64', 'i64_to_f32x2')
        for name in sorted(names, key=lambda name: spans[name][0], reverse=True):
            start, end = spans[name]
            old_start, old_end = old_spans[name]
            lines[start:end] = old_lines[old_start:old_end]
        source = ''.join(lines).replace(
            'from cutlass.base_dsl.typing import Vector',
            'from cutlass._mlir.dialects import arith as _arith\n'
            'from cutlass._mlir.dialects import llvm, vector\n'
            'from cutlass.cutlass_dsl import T')
        filename = '<probe_quack_dsl442_utils>'
        module.__file__ = filename
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        exec(compile(source, filename, 'exec'), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'quack':
            return importlib.machinery.PathFinder.find_spec(fullname, [str(STOCK)])
        if fullname == 'quack.utils':
            return importlib.util.spec_from_loader(fullname, _Loader())
        return None


def apply():
    assert 'quack' not in sys.modules
    import cutlass.base_dsl.arch
    sys.modules['cutlass.base_dsl.enums'] = cutlass.base_dsl.arch
    sys.meta_path.insert(0, _Finder())


if __name__ == '__main__':
    apply()
    import cutlass
    import quack
    print('CuTe path:', cutlass.__file__, flush=True)
    print('Quack:', quack.__version__, quack.__file__, flush=True)
    from mlops.expert_parallel import QuackMoE
    print('QuackMoE import passed', flush=True)
