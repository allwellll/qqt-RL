"""CPU-only consumer for the frozen web-vs-pilot JavaScript comparison."""
from __future__ import annotations
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/combat_mix_v1/js_web_compare_v1"
PROTOCOL = BASE / "PROTOCOL.json"
sys.path.insert(0, str(ROOT))
from scripts.population_run_state import atomic_json, lock, sha

os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def main() -> None:
    guard = lock(BASE / "OWNER.lock")
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    from scripts import je08_bun_authority as authority
    authority.OUT = BASE / "shared"
    authority.OUT.mkdir(parents=True, exist_ok=True)
    spawns = json.loads((ROOT / protocol["spawn_manifest"]).read_text())
    spawns.update(seed=protocol["seed"], source_spawn_seed=spawns["seed"])
    atomic_json(authority.OUT / "spawns.json", spawns)
    completed, failures = {}, {}
    atomic_json(BASE / "OWNER.json", dict(unique=True, cpu_only=True, gpu_control=False,
        protocol_sha256=sha(PROTOCOL), models=len(protocol["models"]), started_unix=time.time()))
    for name, spec in protocol["models"].items():
        out = BASE / name
        try:
            policy = ROOT / spec["policy"]
            assert sha(policy) == spec["policy_sha256"]
            model = authority.export(policy, out / "model")
            meta = json.loads(model.read_text())["meta"]
            assert meta["checkpoint_sha256"] == spec["policy_sha256"]
            modes = {}
            for tier in protocol["hunter_difficulties"]:
                rows = []
                for start in range(0, protocol["games_per_hunter_tier"], 8):
                    target = out / f"hunter_{tier}_{start:03d}.json"
                    if not target.exists():
                        subprocess.run([
                            "node", str(ROOT / "scripts/je05_web_hunter.js"),
                            "--model", str(model.relative_to(ROOT)), "--difficulty", tier,
                            "--spawns", str((authority.OUT / "spawns.json").relative_to(ROOT)),
                            "--start", str(start), "--end", str(start + 8),
                            "--seed", str(protocol["seed"]), "--checkpoint-sha", spec["policy_sha256"],
                            "--out", str(target.relative_to(ROOT)),
                        ], cwd=ROOT, check=True)
                    shard = json.loads(target.read_text())
                    authority.validate_hunter(shard, spec["policy_sha256"], tier, start)
                    rows.extend(shard["episodes"])
                assert len(rows) == protocol["games_per_hunter_tier"]
                modes[tier] = authority.summarize_hunter(rows)
            natural = {}
            for mode in ("categorical", "greedy"):
                target = out / f"natural_{mode}.json"
                if not target.exists():
                    env = dict(os.environ, JE10_ROUTE_TASK="natural_full2400",
                        JE10_ROUTE_GAMES=str(protocol["natural_games"]), JE10_ROUTE_START="0",
                        JE10_ROUTE_END=str(protocol["natural_games"]),
                        JE10_ROUTE_SEED_START=str(protocol["seed"]), CUDA_VISIBLE_DEVICES="")
                    subprocess.run(["node", "scripts/expert_validation_js_eval.js", mode,
                        str(model.relative_to(ROOT)), str(target.relative_to(ROOT)),
                        protocol["regions"]], cwd=ROOT, env=env, check=True)
                result = json.loads(target.read_text())
                episodes = result["results"]["natural_full2400"]["episodes"]
                assert result["passed"] and len(episodes) == protocol["natural_games"]
                assert result["model_json_sha256"] == sha(model)
                natural[mode] = result["results"]["natural_full2400"]
            summary = dict(model=name, policy_sha256=spec["policy_sha256"],
                model_json_sha256=sha(model), onnx_sha256=sha(ROOT / meta["onnx_path"]),
                checkpoint_identity=spec["identity"], hunter_js=modes, natural_js=natural,
                engine="production web/sim.js", teacher=False,
                source_release_assets_downloaded=spec["release_assets_downloaded"],
                no_peak_selection=True, finished_unix=time.time())
            atomic_json(out / "SUMMARY.json", summary)
            completed[name] = dict(summary="runs/combat_mix_v1/js_web_compare_v1/" + name + "/SUMMARY.json",
                                   summary_sha256=sha(out / "SUMMARY.json"))
        except Exception as exc:
            failures[name] = dict(error=type(exc).__name__, detail=str(exc).replace(str(ROOT), "<WORKTREE>"),
                                  unix=time.time())
            atomic_json(out / "FAILED.json", failures[name])
        atomic_json(BASE / "OWNER_STATUS.json", dict(status="RUNNING", completed=completed,
            failed=failures, pending=[n for n in protocol["models"] if n not in completed and n not in failures],
            cpu_only=True, gpu_control=False, protocol_sha256=sha(PROTOCOL), updated_unix=time.time()))
    status = "COMPLETE" if not failures and len(completed) == len(protocol["models"]) else "COMPLETE_WITH_FAILURES"
    atomic_json(BASE / "COMPLETE.json", dict(status=status, completed=completed, failed=failures,
        protocol_sha256=sha(PROTOCOL), finished_unix=time.time()))


if __name__ == "__main__":
    main()
