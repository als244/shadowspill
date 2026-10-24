"""A saved search loads before any algorithm has run in the worker."""

import json
import subprocess
import sys

from shadowspill.planner.search import SearchOptions


def test_builtin_options_load_in_a_fresh_cpu_process():
    payload = json.dumps(SearchOptions().to_dict())
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            "import json, sys; "
            "from shadowspill.planner.search import SearchOptions; "
            "name = 'shadowspill.planner.search.algorithms.pressurefit'; "
            "assert name not in sys.modules; "
            "value = json.loads(sys.argv[1]); "
            "assert SearchOptions.from_dict(value, 'search').to_dict() == value; "
            "assert 'torch' not in sys.modules",
            payload,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
