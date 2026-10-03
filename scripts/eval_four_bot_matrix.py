#!/usr/bin/env python3
"""64-game unified 4-bot quick matrix: each checkpoint vs Tactical v2 and web hunter easy/normal/hard.

Two protocols share one pre-declared seed and one JAX danger_arena spawn manifest:
- bun.tactical_v2: jax_bomb.bun_tactical_eval (JAX env, bf16 policy, Python bot), unchanged.
- bun.hunter@{easy,normal,hard}: scripts/eval_web_hunter.js (web/sim.js + web JS TransformerModel
  + real web/bun_hunter_bot.js through bot_contract), sharded over Node processes.
Every checkpoint x bot cell is written separately and is resumable; the summary refuses
incomplete or mismatched cells. Local checkpoint paths live only in the (uncommitted) manifest.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "four_bot_matrix_64_v1"
CELL_SCHEMA = "four_bot_matrix_cell_v1"
TACTICAL_SCHEMA = "bun_tactical_opponent_eval_v1"
SHARD_SCHEMA = "web_hunter_eval_shard_v1"
HUNTER_DIFFICULTIES = ("hard", "normal", "easy")
BOTS = ("bun.tactical_v2",) + tuple(f"bun.hunter@{d}" for d in HUNTER_DIFFICULTIES)
NODE_SOURCES = ("web/sim.js", "web/bun_hunter_bot.js", "web/bot_contract.js",
                "web/assets/maps/levels.json", "scripts/eval_web_hunter.js")
# Episode-level booleans shared by both protocols (any event within the 300-tick episode).
EPISODE_FLAGS = ("surviving_kill", "killed_by_bot", "own_bomb", "mutual", "win", "loss")


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def node_source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in NODE_SOURCES}


def load_manifest(path: Path) -> list[dict]:
    rows = read_json(path)["checkpoints"]
    names = [row["name"] for row in rows]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate checkpoint names in manifest")
    for row in rows:
        for key in ("name", "path", "source", "group"):
            if key not in row:
                raise SystemExit(f"manifest row missing {key}: {row}")
    return rows


def wait_stable(path: Path, seconds: float = 5.0) -> None:
    """Refuse checkpoints that are still being written."""
    first = path.stat()
    time.sleep(seconds)
    second = path.stat()
    if (first.st_size, first.st_mtime_ns) != (second.st_size, second.st_mtime_ns):
        raise SystemExit(f"checkpoint still changing: {path}")
    if time.time() - second.st_mtime < 60:
        raise SystemExit(f"checkpoint modified <60s ago, refusing: {path}")


# ----------------------------------------------------------------- outcome definitions

def outcome_flags(surviving_kill: bool, killed_by_bot: bool, own_bomb: bool, mutual: bool) -> dict:
    died = killed_by_bot or own_bomb or mutual
    return {"surviving_kill": bool(surviving_kill), "killed_by_bot": bool(killed_by_bot),
            "own_bomb": bool(own_bomb), "mutual": bool(mutual),
            "win": bool(surviving_kill and not died), "loss": bool(died and not surviving_kill)}


def tactical_episodes(data: dict) -> list[dict]:
    per = data["per_episode"]
    games = data["summary"]["games"]
    episodes = []
    for i in range(games):
        flags = outcome_flags(
            per["surviving_causal_kill"][i] or per["surviving_physical_kill"][i],
            per["opponent_physical_defeat"][i] or per["opponent_causal_defeat"][i],
            per["own_bomb_defeat"][i], per["mutual_death"][i])
        flags["bombs"] = int(per["bombs"][i])
        flags["danger_to_death"] = bool(per["danger_to_death"][i])
        episodes.append(flags)
    return episodes


def hunter_episodes(records: list[dict]) -> list[dict]:
    episodes = []
    for record in sorted(records, key=lambda r: r["game"]):
        c = record["counts"]
        flags = outcome_flags(c["surviving_causal_kill"] > 0 or c["surviving_physical_kill"] > 0,
                              c["killed_by_bot"] > 0, c["own_bomb_defeat"] > 0, c["mutual_death"] > 0)
        flags["bombs"] = int(c["bombs"])
        flags["bot_own_bomb"] = c["bot_own_bomb_defeat"] > 0
        flags["bot_bombs"] = int(c["bot_bombs"])
        episodes.append(flags)
    return episodes


def aggregate(episodes: list[dict]) -> dict:
    games = len(episodes)
    out = {"games": games}
    for flag in EPISODE_FLAGS:
        count = sum(1 for e in episodes if e[flag])
        out[flag] = {"count": count, "rate": count / games}
    out["bombs_per_game"] = sum(e["bombs"] for e in episodes) / games
    kills, deaths = out["surviving_kill"]["count"], out["killed_by_bot"]["count"]
    out["kill_share"] = kills / (kills + deaths) if kills + deaths else None
    for extra in ("danger_to_death", "bot_own_bomb"):
        if episodes and extra in episodes[0]:
            count = sum(1 for e in episodes if e[extra])
            out[extra] = {"count": count, "rate": count / games}
    if episodes and "bot_bombs" in episodes[0]:
        out["bot_bombs_per_game"] = sum(e["bot_bombs"] for e in episodes) / games
    return out


# ----------------------------------------------------------------- spawn manifest

def make_spawns(args) -> None:
    os.environ.setdefault("JAXBOMB_RULE", "bun")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import numpy as np
    from jax_bomb import bun_env as env
    from jax_bomb.bun_frozen_opponents import clear_destructible_bricks
    env.prepare()
    env.configure_training("danger_arena=1", 1, reward_profile="danger_arena",
                           tactical_bomb_placement_reward=1.0, tactical_bomb_resolution_reward=1.0)
    key = jax.random.PRNGKey(np.uint32(args.seed & 0xFFFFFFFF))
    states = clear_destructible_bricks(env.init_batch(key, args.games))
    cells = np.floor(np.asarray(states.core.pos)).astype(np.int32).tolist()
    lessons = np.asarray(states.lesson).tolist()
    if set(lessons) != {env.LESSON_DANGER_ARENA}:
        raise SystemExit(f"unexpected lessons {set(lessons)}")
    digest = hashlib.sha256(json.dumps(cells).encode()).hexdigest()
    write_json(Path(args.out), {"schema": "four_bot_spawn_manifest_v1", "seed": args.seed,
                                "games": args.games, "curriculum": "danger_arena=1",
                                "source": "jax_bomb.bun_env.init_batch + clear_destructible_bricks",
                                "spawn_sha256": digest, "spawn_cells": cells})
    print(f"spawns {args.games} sha {digest[:16]}")


# ----------------------------------------------------------------- tactical cells

def tactical_cell_ok(path: Path, checkpoint_sha: str, seed: int, games: int, max_steps: int,
                     spawn_cells: list, trace: bool = False) -> bool:
    if not path.exists():
        return False
    data = read_json(path)
    return (data.get("schema") == TACTICAL_SCHEMA and data.get("checkpoint_sha256") == checkpoint_sha
            and data.get("seed") == seed and data.get("max_steps") == max_steps
            and data["summary"].get("games") == games
            and data["per_episode"]["spawn_cells"] == spawn_cells
            and (not trace or len(data.get("attack_trace", {}).get("episodes", [])) == games))


def run_tactical(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    out = Path(args.out_dir)
    spawns = read_json(out / "spawns.json")
    rows = load_manifest(Path(args.manifest))
    if args.only:
        rows = [r for r in rows if r["name"] in set(args.only.split(","))]
    from qqt_rl.training.jax_cache import configure_persistent_cache
    configure_persistent_cache(explicit=out / ".jax_cache")
    from jax_bomb.bun_tactical_eval import TacticalOpponentEvaluator
    evaluator = None
    try:
        for row in rows:
            path = Path(row["path"])
            sha = sha256_file(path)
            cell = out / "cells" / row["name"] / "tactical_v2.json"
            if tactical_cell_ok(cell, sha, args.seed, args.games, args.max_steps, spawns["spawn_cells"],
                                trace=args.attack_trace):
                print(f"[tactical] {row['name']}: complete, skip")
                continue
            wait_stable(path)
            if evaluator is None:
                evaluator = TacticalOpponentEvaluator(host_workers=args.host_workers)
            started = time.time()
            data = evaluator.run(str(path), args.games, args.seed, args.max_steps, per_episode=True,
                                 trace=args.attack_trace)
            if data["checkpoint_sha256"] != sha or sha256_file(path) != sha:
                raise SystemExit(f"{row['name']}: checkpoint changed during evaluation")
            if data["per_episode"]["spawn_cells"] != spawns["spawn_cells"]:
                raise SystemExit(f"{row['name']}: tactical spawns differ from manifest")
            data["checkpoint"] = row["source"]
            data["wall_seconds"] = time.time() - started
            write_json(cell, data)
            print(f"[tactical] {row['name']}: {data['wall_seconds']:.0f}s "
                  f"kill={data['summary']['p0_surviving_causal_kill_rate']:.3f}")
    finally:
        if evaluator is not None:
            evaluator.close()


# ----------------------------------------------------------------- hunter cells

def export_model(row: dict, out: Path, sha: str) -> tuple[Path, str]:
    target = out / "models" / f"{row['name']}-{sha[:16]}.json"
    if not target.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/export_bun_web_model.py"), row["path"],
                        "--output", str(target), "--name", row["name"]],
                       check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"}, stdout=subprocess.DEVNULL)
    return target, sha256_file(target)


def shard_ok(path: Path, *, seed, max_steps, difficulty, start, end, model_sha, checkpoint_sha,
             sources, spawn_sha, trace=False) -> bool:
    if not path.exists():
        return False
    data = read_json(path)
    if trace and not all("attack_trace" in e for e in data.get("episodes", [])):
        return False
    return (data.get("schema") == SHARD_SCHEMA and data["seed"] == seed and data["max_steps"] == max_steps
            and data["bot"]["config"] == {"difficulty": difficulty}
            and data["game_range"] == [start, end] and len(data["episodes"]) == end - start
            and data["model"]["model_json_sha256"] == model_sha
            and data["checkpoint_sha256"] == checkpoint_sha
            and data["source_hashes"] == sources and data["spawn_manifest_sha256"] == spawn_sha)


def run_hunters(args) -> None:
    out = Path(args.out_dir)
    spawns_path = out / "spawns.json"
    spawns = read_json(spawns_path)
    if spawns["seed"] != args.seed or spawns["games"] != args.games:
        raise SystemExit("spawn manifest does not match --seed/--games")
    rows = load_manifest(Path(args.manifest))
    if args.only:
        rows = [r for r in rows if r["name"] in set(args.only.split(","))]
    difficulties = args.difficulties.split(",")
    sources = node_source_hashes()
    jobs, plan = [], []
    for row in rows:
        path = Path(row["path"])
        wait_stable(path, 1.0)
        sha = sha256_file(path)
        model_path, model_sha = export_model(row, out, sha)
        for difficulty in difficulties:
            cell_dir = out / "cells" / row["name"] / f"hunter_{difficulty}"
            ranges = [(s, min(s + args.shard, args.games)) for s in range(0, args.games, args.shard)]
            plan.append((row, difficulty, sha, model_sha, cell_dir, ranges))
            for start, end in ranges:
                shard = cell_dir / f"games_{start:03d}_{end:03d}.json"
                meta = dict(seed=args.seed, max_steps=args.max_steps, difficulty=difficulty, start=start,
                            end=end, model_sha=model_sha, checkpoint_sha=sha, sources=sources,
                            spawn_sha=spawns["spawn_sha256"], trace=args.attack_trace)
                if shard_ok(shard, **meta):
                    continue
                cell_dir.mkdir(parents=True, exist_ok=True)
                jobs.append((shard, ["node", str(ROOT / "scripts/eval_web_hunter.js"),
                                     "--model", str(model_path), "--difficulty", difficulty,
                                     "--spawns", str(spawns_path), "--start", str(start), "--end", str(end),
                                     "--seed", str(args.seed), "--max-steps", str(args.max_steps),
                                     "--checkpoint-sha", sha, "--out", str(shard)]
                            + (["--attack-trace", "1"] if args.attack_trace else []), meta))
    print(f"[hunter] {len(jobs)} shards to run with {args.workers} workers")
    started = time.time()

    def launch(job):
        shard, command, meta = job
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        if not shard_ok(shard, **meta):
            raise RuntimeError(f"shard failed validation: {shard}")
        return shard

    done = 0
    with futures.ThreadPoolExecutor(args.workers) as pool:
        for future in futures.as_completed([pool.submit(launch, job) for job in jobs]):
            future.result()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"[hunter] {done}/{len(jobs)} shards, {time.time() - started:.0f}s", flush=True)
    for row, difficulty, sha, model_sha, cell_dir, ranges in plan:
        merge_hunter_cell(row, difficulty, sha, model_sha, cell_dir, ranges, spawns, sources, args)


def merge_hunter_cell(row, difficulty, sha, model_sha, cell_dir, ranges, spawns, sources, args) -> None:
    shards = [read_json(cell_dir / f"games_{s:03d}_{e:03d}.json") for s, e in ranges]
    records = [episode for shard in shards for episode in shard["episodes"]]
    games = sorted(r["game"] for r in records)
    if games != list(range(args.games)):
        raise SystemExit(f"{row['name']} {difficulty}: games {len(games)} != {args.games}")
    for record in records:
        if record["spawn_cells"] != spawns["spawn_cells"][record["game"]]:
            raise SystemExit(f"{row['name']} {difficulty}: spawn mismatch game {record['game']}")
    first = shards[0]
    write_json(cell_dir.with_suffix(".json"), {
        "schema": CELL_SCHEMA, "protocol": "web_hunter", "checkpoint": row["source"],
        "checkpoint_sha256": sha, "model_json_sha256": model_sha, "seed": args.seed,
        "max_steps": args.max_steps, "games": args.games, "spawn_manifest_sha256": spawns["spawn_sha256"],
        "bot": first["bot"], "model": first["model"], "env": first["protocol"],
        "source_hashes": sources, "node_version": first["node_version"],
        "cpu_seconds": sum(s["elapsed_seconds"] for s in shards),
        "model_ms_per_tick": sum(r["model_ms"] for r in records) / sum(r["ticks"] for r in records),
        "bot_ms_per_tick": sum(r["bot_ms"] for r in records) / sum(r["ticks"] for r in records),
        "episodes": records,
    })


# ----------------------------------------------------------------- JS <-> JAX parity

def run_parity(args) -> None:
    os.environ.setdefault("JAXBOMB_RULE", "bun")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_bomb import bun_env as env
    from jax_bomb import jax_train
    from jax_bomb.bun_frozen_opponents import clear_destructible_bricks
    from jax_bomb.bun_tactical_eval import load_params
    from jax_bomb.jax_net import transformer_forward
    out = Path(args.out_dir)
    row = next(r for r in load_manifest(Path(args.manifest)) if r["name"] == args.checkpoint)
    sha = sha256_file(Path(row["path"]))
    model_path, model_sha = export_model(row, out, sha)
    dump_path = out / "parity" / f"{row['name']}_dump.json"
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["node", str(ROOT / "scripts/eval_web_hunter.js"), "--parity-dump", str(dump_path),
                    "--model", str(model_path), "--spawns", str(out / "spawns.json"),
                    "--seed", str(args.seed), "--games", str(args.games), "--ticks", str(args.ticks)],
                   check=True)
    dump = read_json(dump_path)
    env.prepare()
    env.configure_training("danger_arena=1", 1, reward_profile="danger_arena",
                           tactical_bomb_placement_reward=1.0, tactical_bomb_resolution_reward=1.0)
    states = clear_destructible_bricks(
        env.init_batch(jax.random.PRNGKey(np.uint32(args.seed & 0xFFFFFFFF)), args.games))
    obs = np.asarray(jax_train.both_perspectives(states))[:args.games]
    gvec = np.asarray(jax_train.both_states(states))[:args.games]
    move_mask = np.asarray(jax_train.both_masks(states)[0])[:args.games]
    js_obs = np.asarray([case["obs"] for case in dump["initial"]], np.float32).reshape(obs.shape)
    js_state = np.asarray([case["state"] for case in dump["initial"]], np.float32)
    js_mask = np.asarray([case["move_mask"] for case in dump["initial"]], bool)
    params = load_params(row["path"])
    trace = dump["trace"]
    t_obs = jnp.asarray(np.asarray([r["obs"] for r in trace], np.float32).reshape((-1,) + obs.shape[1:]))
    t_state = jnp.asarray(np.asarray([r["state"] for r in trace], np.float32))
    move, ability, _, _ = jax.jit(transformer_forward)(params, t_obs, t_state)
    move, ability = np.asarray(move), np.asarray(ability)

    def masked_argmax(logits, mask):
        return int(np.argmax(np.where(np.asarray(mask, bool), logits, -1e30)))

    agree_move = sum(masked_argmax(move[i], r["move_mask"]) == masked_argmax(np.asarray(r["move"]), r["move_mask"])
                     for i, r in enumerate(trace))
    agree_ability = sum(masked_argmax(ability[i], r["ability_mask"])
                        == masked_argmax(np.asarray(r["ability"]), r["ability_mask"]) for i, r in enumerate(trace))
    js_move = np.asarray([r["move"] for r in trace])
    js_ability = np.asarray([r["ability"] for r in trace])
    report = {
        "schema": "four_bot_js_jax_parity_v1", "checkpoint": row["source"], "checkpoint_sha256": sha,
        "model_json_sha256": model_sha, "seed": args.seed,
        "initial_states": args.games,
        "initial_obs_max_abs_diff": float(np.abs(obs - js_obs).max()),
        "initial_obs_channel_max_abs_diff": [float(v) for v in np.abs(obs - js_obs).max(axis=(0, 2, 3))],
        "initial_state_max_abs_diff": float(np.abs(gvec - js_state).max()),
        "initial_move_mask_equal": int((move_mask == js_mask).all(axis=1).sum()),
        "trace_ticks": len(trace),
        "trace_move_logit_max_abs_diff_js_f64_vs_jax_bf16": float(np.abs(move - js_move).max()),
        "trace_ability_logit_max_abs_diff_js_f64_vs_jax_bf16": float(np.abs(ability - js_ability).max()),
        "trace_move_argmax_agreement": agree_move / len(trace),
        "trace_ability_argmax_agreement": agree_ability / len(trace),
    }
    write_json(out / "parity" / f"{row['name']}.json", report)
    print(json.dumps(report, indent=1))


# ----------------------------------------------------------------- summary

def load_cells(out: Path, rows: list[dict], args) -> dict:
    spawns = read_json(out / "spawns.json")
    sources = node_source_hashes()
    cells = {}
    for row in rows:
        sha = sha256_file(Path(row["path"]))
        base = out / "cells" / row["name"]
        tactical = base / "tactical_v2.json"
        if not tactical_cell_ok(tactical, sha, args.seed, args.games, args.max_steps, spawns["spawn_cells"]):
            raise SystemExit(f"{row['name']}: tactical cell incomplete or mismatched")
        data = read_json(tactical)
        cells[(row["name"], "bun.tactical_v2")] = {
            "episodes": tactical_episodes(data), "wall_seconds": data.get("wall_seconds"),
            "bot": {"id": "bun.tactical_v2", "version": data["opponent"].get("version"),
                    "module_sha256": data["opponent"].get("module_sha256")},
            "runtime": data["runtime"], "checkpoint_sha256": sha}
        for difficulty in HUNTER_DIFFICULTIES:
            path = base / f"hunter_{difficulty}.json"
            if not path.exists():
                raise SystemExit(f"{row['name']}: hunter {difficulty} cell missing")
            cell = read_json(path)
            ok = (cell["schema"] == CELL_SCHEMA and cell["checkpoint_sha256"] == sha
                  and cell["seed"] == args.seed and cell["games"] == args.games
                  and len(cell["episodes"]) == args.games and cell["max_steps"] == args.max_steps
                  and cell["bot"]["config"] == {"difficulty": difficulty}
                  and cell["spawn_manifest_sha256"] == spawns["spawn_sha256"]
                  and cell["source_hashes"] == sources)
            if not ok:
                raise SystemExit(f"{row['name']}: hunter {difficulty} cell mismatched")
            cells[(row["name"], f"bun.hunter@{difficulty}")] = {
                "episodes": hunter_episodes(cell["episodes"]), "cpu_seconds": cell["cpu_seconds"],
                "bot": cell["bot"], "model_json_sha256": cell["model_json_sha256"],
                "model_ms_per_tick": cell["model_ms_per_tick"], "bot_ms_per_tick": cell["bot_ms_per_tick"],
                "checkpoint_sha256": sha}
    return cells


def summarize(args) -> None:
    out = Path(args.out_dir)
    rows = load_manifest(Path(args.manifest))
    cells = load_cells(out, rows, args)
    spawns = read_json(out / "spawns.json")
    report_dir = Path(args.report_dir)
    checkpoints = []
    for row in rows:
        entry = {"name": row["name"], "group": row["group"], "label": row.get("label", row["name"]),
                 "source": row["source"], "checkpoint_sha256": cells[(row["name"], BOTS[0])]["checkpoint_sha256"],
                 "bots": {}}
        for bot in BOTS:
            cell = cells[(row["name"], bot)]
            item = {"metrics": aggregate(cell["episodes"]), "bot": cell["bot"]}
            for key in ("wall_seconds", "cpu_seconds", "model_ms_per_tick", "bot_ms_per_tick", "model_json_sha256"):
                if cell.get(key) is not None:
                    item[key] = cell[key]
            entry["bots"][bot] = item
        checkpoints.append(entry)
    timing = read_json(out / "timing.json") if (out / "timing.json").exists() else None
    summary = {
        "schema": SCHEMA, "seed": args.seed, "games_per_cell": args.games, "max_steps": args.max_steps,
        "spawn_manifest": {"sha256": spawns["spawn_sha256"], "source": spawns["source"]},
        "bots": list(BOTS), "node_source_hashes": node_source_hashes(),
        "outcome_definitions": {
            "surviving_kill": "episode has >=1 non-trade kill (causal or physical) with actor alive after it",
            "killed_by_bot": "episode has >=1 actor death solely sourced by the bot (physical or causal), excluding trades",
            "own_bomb": "episode has >=1 actor death solely from its own bomb",
            "mutual": "episode has >=1 same-tick double death (trade)",
            "win": "surviving_kill and none of killed_by_bot/own_bomb/mutual",
            "loss": "any of killed_by_bot/own_bomb/mutual and no surviving_kill",
            "kill_share": "surviving_kill episodes / (surviving_kill + killed_by_bot episodes)",
        },
        "timing": timing, "checkpoints": checkpoints,
    }
    parity = sorted((out / "parity").glob("*.json")) if (out / "parity").exists() else []
    summary["js_jax_parity"] = [read_json(p) for p in parity if not p.name.endswith("_dump.json")]
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "summary.json", summary)
    with open(report_dir / "summary.csv", "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["checkpoint", "group", "checkpoint_sha256", "bot", "games", "win", "loss",
                         "surviving_kill", "killed_by_bot", "own_bomb", "mutual", "kill_share", "bombs_per_game"])
        for entry in checkpoints:
            for bot, item in entry["bots"].items():
                m = item["metrics"]
                writer.writerow([entry["name"], entry["group"], entry["checkpoint_sha256"], bot, m["games"],
                                 m["win"]["count"], m["loss"]["count"], m["surviving_kill"]["count"],
                                 m["killed_by_bot"]["count"], m["own_bomb"]["count"], m["mutual"]["count"],
                                 "" if m["kill_share"] is None else f"{m['kill_share']:.4f}",
                                 f"{m['bombs_per_game']:.3f}"])
    with open(report_dir / "episodes.csv", "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["checkpoint", "bot", "game"] + list(EPISODE_FLAGS) + ["bombs"])
        for row in rows:
            for bot in BOTS:
                for game, e in enumerate(cells[(row["name"], bot)]["episodes"]):
                    writer.writerow([row["name"], bot, game] + [int(e[f]) for f in EPISODE_FLAGS] + [e["bombs"]])
    print(markdown_table(summary))


def markdown_table(summary: dict) -> str:
    def pct(m, key):
        return f"{100 * m[key]['rate']:.1f}"
    lines = ["| checkpoint | " + " | ".join(
        f"{b.replace('bun.', '')} kill/被杀/自炸/换命/W-L/泡" for b in summary["bots"]) + " |",
        "|---|" + "---|" * len(summary["bots"])]
    for entry in summary["checkpoints"]:
        parts = []
        for bot in summary["bots"]:
            m = entry["bots"][bot]["metrics"]
            parts.append(f"{pct(m, 'surviving_kill')}/{pct(m, 'killed_by_bot')}/{pct(m, 'own_bomb')}/"
                         f"{pct(m, 'mutual')}/{m['win']['count']}-{m['loss']['count']}/{m['bombs_per_game']:.1f}")
        lines.append(f"| {entry['label']} | " + " | ".join(parts) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("spawns", "tactical", "hunters", "parity", "summarize"))
    parser.add_argument("--out-dir", default="runs/eval_four_bots_64")
    parser.add_argument("--manifest", default="runs/eval_four_bots_64/checkpoints.json")
    parser.add_argument("--report-dir", default="reports/four_bot_matrix_64_20261001")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="1")
    parser.add_argument("--host-workers", type=int, default=16)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--shard", type=int, default=8)
    parser.add_argument("--difficulties", default=",".join(HUNTER_DIFFICULTIES))
    parser.add_argument("--only", default="")
    parser.add_argument("--checkpoint", default="", help="parity: manifest name")
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--attack-trace", action="store_true",
                        help="tactical/hunters: also emit attack_trace_v1 raw events (default off)")
    parser.add_argument("--out", default="", help="spawns: output path")
    args = parser.parse_args()
    if args.command == "spawns":
        args.out = args.out or str(Path(args.out_dir) / "spawns.json")
        make_spawns(args)
    elif args.command == "tactical":
        run_tactical(args)
    elif args.command == "hunters":
        run_hunters(args)
    elif args.command == "parity":
        run_parity(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
