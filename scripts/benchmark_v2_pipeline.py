#!/usr/bin/env python3
"""Bun v2 训练热路径的可复现实测与 legacy/batched A/B。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from qqt_rl.training.counterfactual import compare_replay_arrays
from qqt_rl.training.io import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
FLOAT_FIELDS = {
    "deterministic_return", "deterministic_objective", "q", "win_rate",
    "win_ci_low", "win_ci_high", "q_ci_low", "q_ci_high", "mc_kill_rate",
    "mc_trade_rate", "mc_objective", "policy_probability", "value_target",
    "win_target", "next_global", "next_context", "immediate_reward",
    "minimum_escape_time",
}


def semantic_equivalence(report: dict, float_atol: float = 1e-6) -> dict:
    violations = {}
    for name, field in report["fields"].items():
        if field.get("exact"):
            continue
        if name in FLOAT_FIELDS and field.get("max_abs_error", float("inf")) <= float_atol:
            continue
        violations[name] = field
    return {"equivalent": not violations, "float_atol": float_atol,
            "violations": violations}


def parse_joint_steps_per_second(output: str) -> float:
    match = re.search(r"FINAL end-to-end sps = ([0-9,]+(?:\.[0-9]+)?)", output)
    if not match:
        raise ValueError("joint smoke throughput missing")
    return float(match.group(1).replace(",", ""))


def _run_json(command: list[str], environment: dict[str, str]) -> tuple[float, str]:
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(
            f"benchmark command failed rc={result.returncode}: {command}\n{result.stdout}")
    return elapsed, result.stdout


def _write_actor(path: Path) -> None:
    os.environ.setdefault("JAXBOMB_RULE", "bun")
    import jax
    from jax_bomb import bun_env
    from jax_bomb.jax_net import init_transformer
    params = init_transformer(
        jax.random.PRNGKey(20260928), bun_env.N_OBS_CH, bun_env.H, bun_env.W,
        embed=16, depth=1, heads=4, ff_factor=2, patch=3, state_dim=24)
    with path.open("wb") as handle:
        pickle.dump({"params": jax.device_get(params)}, handle)


def _counterfactual_run(actor: Path, root: Path, name: str, engine: str,
                        state_batch_size: int, cache_dir: Path,
                        states: int) -> dict:
    output = root / f"{name}.npz"
    manifest = root / f"{name}.json"
    environment = os.environ.copy()
    environment.update({
        "JAXBOMB_RULE": "bun", "JAX_PLATFORMS": "cpu", "QQT_ALLOW_CPU": "1",
        "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT),
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    })
    command = [
        sys.executable, str(ROOT / "scripts/generate_bun_tactical_v2_critic_data.py"),
        "--actor", str(actor), "--output", str(output), "--manifest", str(manifest),
        "--seed-base", "2026092801", "--train-states", str(states),
        "--validation-states", "0", "--test-states", "0", "--mc-samples", "1",
        "--mc-horizon", "1", "--collection-ticks", "0", "--engine", engine,
        "--state-batch-size", str(state_batch_size), "--jax-cache-dir", str(cache_dir),
    ]
    elapsed, stdout = _run_json(command, environment)
    payload = json.loads(manifest.read_text())
    payload.update({"process_wall_seconds": elapsed, "stdout_tail": stdout[-500:]})
    return payload


def benchmark_critic_batches(batch_sizes=(64, 128, 256), steps: int = 5) -> list[dict]:
    import jax
    import jax.numpy as jnp
    import optax
    from jax_bomb.bun_critic import init_independent_critic, independent_critic_forward
    rows = []
    for batch_size in batch_sizes:
        key = jax.random.PRNGKey(7000 + batch_size)
        params = init_independent_critic(key, input_dim=128, hidden=64, action_count=15)
        optimizer = optax.adam(3e-4)
        opt_state = optimizer.init(params)
        inputs = jax.random.normal(jax.random.PRNGKey(8000), (batch_size, 128))
        targets = jax.random.normal(jax.random.PRNGKey(9000), (batch_size,))

        @jax.jit
        def step(current, state):
            def loss_fn(value):
                prediction = independent_critic_forward(value, inputs)[0]
                return jnp.mean(jnp.square(prediction - targets))
            loss, gradients = jax.value_and_grad(loss_fn)(current)
            updates, state = optimizer.update(gradients, state, current)
            return optax.apply_updates(current, updates), state, loss

        params, opt_state, initial_loss = step(params, opt_state)
        jax.block_until_ready(initial_loss)
        started = time.perf_counter()
        final_loss = initial_loss
        for _ in range(steps):
            params, opt_state, final_loss = step(params, opt_state)
        jax.block_until_ready(final_loss)
        elapsed = time.perf_counter() - started
        rows.append({
            "batch_size": batch_size, "steps": steps,
            "samples_per_second": batch_size * steps / max(elapsed, 1e-9),
            "initial_loss": float(initial_loss), "final_loss": float(final_loss),
            "finite": bool(np.isfinite(float(final_loss))),
            "default": batch_size == 64,
        })
    return rows


def _cache_probe(cache_dir: Path) -> dict:
    code = """
import json,time,jax,jax.numpy as jnp
f=jax.jit(lambda x: jnp.tanh(x@x.T).sum())
x=jnp.ones((512,512),jnp.float32)
t=time.perf_counter(); y=f(x); jax.block_until_ready(y)
print(json.dumps({'seconds':time.perf_counter()-t,'value':float(y)}))
"""
    environment = os.environ.copy()
    environment.update({
        "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "",
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    })
    elapsed, output = _run_json([sys.executable, "-c", code], environment)
    payload = json.loads(output.strip().splitlines()[-1])
    payload["process_wall_seconds"] = elapsed
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/benchmarks/v2_pipeline")
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--counterfactual-states", type=int, default=2)
    parser.add_argument("--skip-joint-smoke", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    actor = args.actor.resolve() if args.actor else root / "synthetic_actor.pt"
    if args.actor is None:
        _write_actor(actor)

    cold_cache = root / "cache-cold"
    warm_cache = root / "cache-warm"
    shutil.rmtree(cold_cache, ignore_errors=True)
    shutil.rmtree(warm_cache, ignore_errors=True)
    cache_cold = _cache_probe(warm_cache)
    cache_warm = _cache_probe(warm_cache)
    legacy = _counterfactual_run(
        actor, root, "legacy", "legacy", 1, cold_cache,
        args.counterfactual_states)
    batched_cold = _counterfactual_run(
        actor, root, "batched_cold", "batched", args.counterfactual_states,
        root / "cache-batched", args.counterfactual_states)
    batched_warm = _counterfactual_run(
        actor, root, "batched_warm", "batched", args.counterfactual_states,
        root / "cache-batched", args.counterfactual_states)
    with np.load(root / "legacy.npz") as left, np.load(root / "batched_cold.npz") as right:
        comparison = compare_replay_arrays(
            {key: left[key] for key in left.files},
            {key: right[key] for key in right.files})
    semantic = semantic_equivalence(comparison)
    critic = benchmark_critic_batches()
    joint = {"status": "skipped"}
    if not args.skip_joint_smoke:
        environment = os.environ.copy()
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        elapsed, output = _run_json(["bash", "scripts/smoke_train_cpu.sh"], environment)
        joint = {"status": "passed", "steps_per_second": parse_joint_steps_per_second(output),
                 "wall_seconds": elapsed}
    result = {
        "schema": "qqt.v2.pipeline_benchmark/v1",
        "created_unix": time.time(), "device": "cpu",
        "persistent_cache": {"cold": cache_cold, "warm": cache_warm,
                             "speedup": cache_cold["seconds"] / max(cache_warm["seconds"], 1e-9)},
        "counterfactual": {
            "legacy": legacy, "batched_cold": batched_cold,
            "batched_warm": batched_warm,
            "cold_speedup": legacy["states_per_second"] and (
                batched_cold["states_per_second"] / legacy["states_per_second"]),
            "warm_speedup": legacy["states_per_second"] and (
                batched_warm["states_per_second"] / legacy["states_per_second"]),
            "equivalence": semantic,
            "field_differences": {key: value for key, value in comparison["fields"].items()
                                  if not value.get("exact")},
        },
        "critic_batch_sizes": critic, "joint_update": joint,
        "gpu_utilization": {
            "status": "skipped", "samples": [],
            "reason": "CPU-safe benchmark; no isolated GPU was allocated",
        },
        "artifacts": {"actor_sha256": sha256_file(actor)},
    }
    atomic_write_json(root / "benchmark.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not semantic["equivalent"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
