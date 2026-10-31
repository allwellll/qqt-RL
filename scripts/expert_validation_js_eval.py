"""Run the preregistered production-JS paired evaluation for the expert audit.

This process owns only the CPU namespace under ``runs/expert_validation_20261025``.
It never starts or controls a GPU worker.  The Node evaluator is the production
JS route evaluator; categorical and greedy modes are separate rows and share
the same fixed seed list.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import fcntl
from pathlib import Path

from scripts.population_run_state import atomic_json, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_validation_20261025"
GAMES = 50
SEED_START = 2026102860
MODELS = {
    "parent": "runs/expert_validation_20261025/models/parent.json",
    "candidate": "runs/expert_validation_20261025/models/candidate.json",
    "online": "runs/expert_validation_20261025/models/online.json",
}
OUT = B / "B_js" / "verified_v2"


def run_one(row: tuple[str, str]) -> dict:
    model_name, mode = row
    model = R / MODELS[model_name]
    output = OUT / f"{model_name}_{mode}" / "RESULT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        NODE_OPTIONS="",
        JE10_ROUTE_TASK="natural_full2400",
        JE10_ROUTE_GAMES=str(GAMES),
        JE10_ROUTE_START="0",
        JE10_ROUTE_END=str(GAMES),
        JE10_ROUTE_SEED_START=str(SEED_START),
    )
    if not output.exists():
        log = output.with_suffix(".log")
        with log.open("a") as stream:
            subprocess.run(
                ["node", "scripts/expert_validation_js_eval.js", mode,
                 str(model.relative_to(R)), str(output.relative_to(R)),
                 "runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json"],
                cwd=R, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT,
            )
    data = json.loads(output.read_text())
    assert data.get("passed") is True
    row_data = data["results"]["natural_full2400"]
    assert row_data["games"] == GAMES
    return {
        "model": model_name,
        "mode": mode,
        "games": GAMES,
        "seed_start": SEED_START,
        "model_sha256": sha(model),
        "onnx_sha256": sha(R / json.loads(model.read_text())["meta"]["onnx_path"]),
        "output": str(output.relative_to(R)),
        "delivery_episodes": row_data.get("delivery_episodes"),
        "death_episodes": row_data.get("death_episodes"),
        "deaths": row_data.get("death"),
        "pickup_episodes": row_data.get("pickup_episodes"),
        "window_events": row_data.get("window_events"),
        "oracle_opportunities": row_data.get("oracle_opportunities"),
        "oracle_matches": row_data.get("oracle_matches"),
        "first_oracle_match_episodes": row_data.get("first_oracle_match_episodes"),
        "stage_funnel": row_data.get("stage_funnel", {}),
        "item_metrics": row_data.get("item_metrics", {}),
    }


def main() -> None:
    rows = [(model, mode) for model in MODELS for mode in ("categorical", "greedy")]
    results = []
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = B / "B_js" / "OWNER.lock"
    with lock_path.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a production-JS audit owner is already active") from exc
        atomic_json(OUT / "OWNER.json", {
            "unique": True, "controls_gpu": False,
            "games_per_row": GAMES, "rows": [list(row) for row in rows],
            "started_unix": time.time(),
        })
        # One production-JS worker prevents CPU contention and duplicate owners.
        for row in rows:
            results.append(run_one(row))
            atomic_json(OUT / "PROGRESS.json", {
                "completed_rows": len(results), "total_rows": len(rows),
                "completed_games": len(results) * GAMES,
                "results": results, "controls_gpu": False,
                "updated_unix": time.time(),
            })
    results.sort(key=lambda x: (x["model"], x["mode"]))
    atomic_json(OUT / "RESULTS.json", {
        "schema": "qqt.expert-validation-js/v1",
        "engine": "production JS natural_full2400 + production JS normal Hunter",
        "teacher": {"applied": False, "query": False},
        "action_rng": "production per-episode seeded RNG; shared seed list by mode/model",
        "paired_seed_start": SEED_START,
        "games_per_row": GAMES,
        "total_games": len(results) * GAMES,
        "models": MODELS,
        "results": results,
        "no_holdout_tuning": True,
        "does_not_control_gpu": True,
        "source_sha256": {
            "scripts/ppo_scale_natural_audit.js": sha(R / "scripts/ppo_scale_natural_audit.js"),
            "scripts/dagger_rl_route.js": sha(R / "scripts/dagger_rl_route.js"),
            "scripts/ppo_scale_item_metrics_v4.js": sha(R / "scripts/ppo_scale_item_metrics_v4.js"),
            "scripts/expert_validation_js_eval.js": sha(R / "scripts/expert_validation_js_eval.js"),
        },
    })


if __name__ == "__main__":
    main()
