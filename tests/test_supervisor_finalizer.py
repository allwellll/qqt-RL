import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from qqt_rl.training.io import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_supervisor_completes_without_spawning_finished_workers(tmp_path):
    repo = tmp_path / "repo"
    run = repo / "runs/demo"
    frozen = run / "frozen_runtime"
    finalizer = frozen / "scripts/finalize_bun_safe_aggression_v7.py"
    finalizer.parent.mkdir(parents=True)
    finalizer.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "m=json.loads(Path(sys.argv[2]).read_text())\n"
        "Path(m['run_root'],'COMPLETE.json').write_text('{}')\n")
    config = run / "control/config.v1.json"
    _write_json(config, {"name": "control"})
    _write_json(run / "control/status.json", {
        "candidate": "control", "status": "complete", "cycle": 1,
    })
    manifest = {
        "repo_root": str(repo), "run_root": str(run),
        "checkpoint_root": str(repo / "checkpoints/demo"),
        "data_root": str(repo / "data/demo"), "frozen_root": str(frozen),
        "python": sys.executable, "target_cycles": 1, "max_attempts": 1,
        "bootstrap": [],
        "frozen_files": [{"relative_path": "scripts/finalize_bun_safe_aggression_v7.py",
                          "sha256": sha256_file(finalizer), "bytes": finalizer.stat().st_size}],
        "candidates": [{"name": "control", "gpu": 0, "seed": 1,
                        "config": str(config), "config_sha256": sha256_file(config)}],
    }
    manifest_path = run / "manifest.json"
    _write_json(manifest_path, manifest)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/launch_bun_safe_aggression_v7.py"),
        "--manifest", str(manifest_path),
    ], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (run / "COMPLETE.json").is_file()
    state = json.loads((run / "supervisor_state.json").read_text())
    assert state["status"] == "complete"
    assert not list(run.glob("control/worker_attempt_*.log"))


def test_finalizer_verifies_and_publishes_checkpoint_bundle(tmp_path):
    run = tmp_path / "runs/demo"
    checkpoints = tmp_path / "checkpoints/demo/control"
    checkpoints.mkdir(parents=True)
    payload = {"params": {"w": np.ones((2, 2), np.float32)}}
    for cycle in (0, 1):
        cycle_dir = run / "control" / f"cycle_{cycle:03d}"
        _write_json(cycle_dir / "summary.json", {"strata": {"full": {"metrics": {}}}})
        for suffix in ("actor.pt", "critic.pkl", "target_critic.pkl"):
            with (checkpoints / f"cycle_{cycle:03d}_{suffix}").open("wb") as handle:
                pickle.dump(payload, handle)
    _write_json(run / "control/status.json", {
        "candidate": "control", "status": "complete", "cycle": 1,
        "episodes_completed": 2,
    })
    manifest = {"run_root": str(run), "checkpoint_root": str(tmp_path / "checkpoints/demo"),
                "target_cycles": 1, "milestones": [0, 1],
                "candidates": [{"name": "control"}]}
    manifest_path = run / "manifest.json"
    _write_json(manifest_path, manifest)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/finalize_bun_safe_aggression_v7.py"),
        "--manifest", str(manifest_path),
    ], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    complete = json.loads((run / "COMPLETE.json").read_text())
    assert complete["status"] == "complete"
    assert (checkpoints / "latest_actor.pt").is_file()
