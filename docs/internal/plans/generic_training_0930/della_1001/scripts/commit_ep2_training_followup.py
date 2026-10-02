"""Review two commit groups; --apply requires green Tübingen qualification."""

import argparse
import subprocess
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
PHASE = "docs/internal/plans/generic_training_0930/della_1001/"
GROUPS = (
    (
        "experiments: train admitted EP2 plans with preflight and online W&B",
        tuple(
            PHASE + p
            for p in (
                "CAPACITY_SWEEP.md",
                "EVALUATION_CALLBACK.md",
                "evidence/capacity_retry_all.json",
                "evidence/copy_chicago_2b.log",
                "evidence/callback_qualification.json",
                "scripts/chicago_ep2/README.md",
                "scripts/chicago_ep2/launch.sh",
                "scripts/chicago_ep2/train.py",
                "scripts/copy_chicago_prefix.py",
                "scripts/train_best_capacity.py",
                "scripts/replay_eval_admission.py",
                "scripts/commit_ep2_training_followup.py",
            )
        ),
    ),
    (
        "Preserve registered layer paths when capturing forward callbacks",
        (
            "src/shadowspill/pytorch/planning/forward/capture.py",
            "tests/shadowspill/pytorch/partition/test_partition.py",
            "docs/examples/forward-only.md",
        ),
    ),
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    assert git("branch", "--show-current") == "master"
    allowed = {p for _, files in GROUPS for p in files}
    staged = set(git("diff", "--cached", "--name-only").splitlines())
    assert staged <= allowed, staged - allowed
    for message, files in GROUPS:
        print(message, *files, sep="\n", flush=True)
        for path in files:
            assert (ROOT / path).is_file(), path
    if not args.apply:
        return
    command = """python3 - <<'PY'
from pathlib import Path
import json
r=Path('/home/shein/Documents/shadowspill/qualification/results')
s=(r/'gates_tubingen_callback_partition_1002/suite.log').read_text()
assert '100% tests passed out of 48' in s and 'passed, 1 skipped' in s
assert ' FAILED' not in s and 'FAILED ' not in s
path=r/'numerical_tubingen_callback_partition_1002/summary.json'
assert json.loads(path.read_text())['passed']
print('suite and numerical passed')
PY"""
    subprocess.run(["ssh", "-o", "BatchMode=yes", "tubingen", command], check=True)
    if staged:
        git("restore", "--staged", "--", *sorted(staged))
    for message, files in GROUPS:
        git("add", "--force", "--", *files)
        assert set(git("diff", "--cached", "--name-only").splitlines()) == set(files)
        subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
