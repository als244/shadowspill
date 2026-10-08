"""Audit optimizer inputs from saved neutral programs, without loading tensors."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for directory in ("full-model-steady-benchmark", "llama-1b", "llama-1b-timing"):
    for result_file in sorted((ROOT/"evidence"/directory).glob("*/result.json")):
        case = result_file.parent
        result = json.loads(result_file.read_text())
        inventory = json.loads((case/"parameters.json").read_text())
        programs = list((case/"artifacts/v1/planning").glob("programs/*/*/program.json"))
        assert len(programs) == 1, programs
        program = json.loads(programs[0].read_text())
        objects = {obj["object_id"]: obj for obj in program["objects"]}
        input_ids = {oid for task in program["tasks"] if task["phase"] == "optimizer" for oid in task["inputs"]}
        def role_bytes(role):
            return sum(objects[oid]["size_bytes"] for oid in input_ids if objects[oid]["role"] == role)
        observed = {role: role_bytes(role) for role in ("parameter", "gradient", "optimizer_state")}
        expected = dict(parameter=sum(p["bytes"] for p in inventory["trainable"]),
                        gradient=4*inventory["trainable_parameters"],
                        optimizer_state=8*inventory["trainable_parameters"]+8*len(inventory["trainable"]))
        assert observed == expected, (case.name, observed, expected)
        report = dict(status="passed", case=case.name, source_program=str(programs[0]),
                      trainable_parameters=inventory["trainable_parameters"],
                      trainable_tensors=len(inventory["trainable"]),
                      optimizer_input_bytes=observed,
                      expected_bytes=expected,
                      explanation="BF16 full weights or FP32 LoRA factors, FP32 gradients, two FP32 AdamW moments and one int64 step per trainable tensor. No frozen-weight optimizer state or optimizer gradient inputs.")
        (case/"optimizer-audit.json").write_text(json.dumps(report, indent=2)+"\n")
        rows.append(report)
(ROOT/"optimizer-audit.json").write_text(json.dumps(rows,indent=2)+"\n")
print(f"Optimizer input/state accounting passed for {len(rows)} cases", flush=True)
