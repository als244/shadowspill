"""Shared process infrastructure cannot depend on the clients that consume it."""

import ast
import subprocess
import sys
from pathlib import Path


def test_repository_clients_have_one_way_dependencies():
    root = Path(__file__).resolve().parents[2]
    rules = {
        "workloads": ("benchmarking", "qualification"),
        "orchestration": ("workloads", "benchmarking", "qualification", "training"),
        "src/shadowspill": (
            "orchestration",
            "workloads",
            "benchmarking",
            "qualification",
        ),
        "benchmarking/plan_runner/api.py": ("workloads", "benchmarking.quickstart"),
    }
    violations = []
    for name, forbidden in rules.items():
        path = root / name
        paths = [path] if path.is_file() else path.rglob("*.py")
        for source in paths:
            for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
                modules = ()
                if isinstance(node, ast.Import):
                    modules = tuple(item.name for item in node.names)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    modules = (node.module or "",)
                for module in modules:
                    if any(
                        module == f or module.startswith(f + ".") for f in forbidden
                    ):
                        violations.append((str(source.relative_to(root)), module))
    assert not violations


def test_shared_imports_are_cpu_only_and_recipes_need_no_benchmark_package():
    script = """
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    blocked = {'torch', 'workloads', 'benchmarking', 'qualification'}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.blocked:
            raise AssertionError('unexpected dependency: ' + fullname)
block = Block()
sys.meta_path.insert(0, block)
for module in ('processes', 'device_worker', 'rank_device',
               'workers', 'lifecycle', 'candidates'):
    importlib.import_module('orchestration.' + module)
block.blocked = {'benchmarking', 'qualification'}
for module in ('workloads.recipes', 'workloads.recipes.phases',
               'workloads.recipes.training.build', 'workloads.recipes.training.profile',
               'workloads.recipes.training.communicate',
               'workloads.recipes.training.execute'):
    importlib.import_module(module)
"""
    result = subprocess.run(
        [sys.executable, *(["-S"] if sys.flags.no_site else []), "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
