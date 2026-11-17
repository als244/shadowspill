"""Tensor descriptors are available without loading framework or execution code."""

import subprocess
import sys


def test_tensor_chunk_descriptor_does_not_import_execution_or_torch():
    subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            """
import sys
from shadowspill import TensorChunk
assert 'torch' not in sys.modules
assert 'shadowspill.execution' not in sys.modules
assert TensorChunk('weight', (4,), 'float32', 0, 4).storage_elements == 4
""",
        ],
        check=True,
    )
