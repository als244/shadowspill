"""Shared process infrastructure cannot depend on the clients that consume it."""

import ast
import subprocess
import sys
from pathlib import Path


def test_repository_clients_have_one_way_dependencies():
    root = Path(__file__).resolve().parents[2]
    rules = {
        "packages/workloads/src/shadowspill_workloads": (
            "benchmarks",
            "qualification",
            "examples",
            "reference",
            "tools",
        ),
        "packages/engine/src/shadowspill": (
            "orchestration",
            "shadowspill_workloads",
            "benchmarks",
            "qualification",
            "examples",
            "reference",
            "tools",
        ),
        "benchmarks/plan_runner/api.py": (
            "shadowspill_workloads",
            "benchmarks.quickstart",
        ),
    }
    violations = []
    for name, forbidden in rules.items():
        path = root / name
        assert path.exists(), f"missing dependency boundary: {path}"
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
    blocked = {'torch', 'shadowspill_workloads', 'benchmarks', 'qualification'}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.blocked:
            raise AssertionError('unexpected dependency: ' + fullname)
block = Block()
sys.meta_path.insert(0, block)
for module in ('launch.processes', 'launch.rank_device', 'launch.workers',
               'launch.staging', 'launch.supervision', 'launch.connected',
               'launch.admission', 'launch.host_memory', 'launch.device_inventory',
               'launch.device_probe', 'launch.file_grants', 'monitoring.lifecycle',
               'candidates', 'communication',
               'capture.nsys', 'capture.nsys_worker'):
    importlib.import_module('shadowspill.orchestration.' + module)
for module in ('api', 'configuration', 'phases', 'candidates', 'reuse',
               'plan', 'search', 'execute'):
    importlib.import_module('shadowspill.training.preparation.' + module)
block.blocked = {'benchmarks', 'qualification'}
for module in ('shadowspill_workloads.recipes',
               'shadowspill.training.preparation.phases',
               'shadowspill.training.preparation.build',
               'shadowspill.training.preparation.profile',
               'shadowspill.training.preparation.communicate',
               'shadowspill.training.preparation.execute'):
    importlib.import_module(module)
"""
    result = subprocess.run(
        [sys.executable, *(["-S"] if sys.flags.no_site else []), "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
