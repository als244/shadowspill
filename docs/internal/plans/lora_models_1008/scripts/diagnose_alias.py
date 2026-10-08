import dataclasses
import json
import runpy
from pathlib import Path
from unittest.mock import patch
import shadowspill.pytorch.planning.training.programs as module

real = module.build_admission_facts

def save(program, **kwargs):
    try:
        return real(program, **kwargs)
    except Exception:
        path = Path(__file__).resolve().parents[1] / "evidence/qwen_alias_failure.json"
        path.write_text(json.dumps({"program": dataclasses.asdict(program),
                                   "inputs": kwargs}, default=lambda x: dataclasses.asdict(x) if dataclasses.is_dataclass(x) else str(x), indent=2))
        raise

with patch.object(module, "build_admission_facts", save):
    runpy.run_path(str(Path(__file__).with_name("full_model_lora.py")), run_name="__main__")
