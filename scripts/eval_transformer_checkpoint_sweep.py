#!/usr/bin/env python3
"""Serial checkpoint sweep: frozen Transformer actor vs Bun Tactical v2 in danger_arena.

All checkpoints run in one process through jax_bomb.bun_tactical_eval (same
protocol and bit-identical metrics as the historical runs/eval_v2 files): the env
step compiles once, rule-bot decisions/labels use a host worker pool, and an
optional persistent JAX cache makes re-runs skip compilation. Per-checkpoint JSON,
a summary JSON/CSV and capability curves are written to the output dir.
X axis: phase-2 iteration; the phase-1 final checkpoint is plotted at 0 and the
final phase2.pt at its configured ppo_iterations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "transformer_checkpoint_sweep_v1"
EVAL_SCHEMA = "bun_tactical_opponent_eval_v1"

# Episode-level binary rates: count = rate * games exactly, so Wilson CIs are valid.
RATE_METRICS = {
    "surviving_kill": "p0_surviving_causal_kill_rate",
    "own_bomb_death": "p0_own_bomb_defeat_rate",
    "killed_by_opponent": "p0_opponent_defeat_rate",
    "avoidable_danger_death": "p0_avoidable_danger_death_rate",
    "danger_to_death": "p0_danger_to_death_rate",
    "mutual_death": "mutual_death_rate",
}
# Event-pooled ratios / means: no episode-level CI is reported.
VALUE_METRICS = {
    "safe_detonation_ratio": "p0_safe_detonation_ratio",
    "tactical_resolution_ratio": "p0_tactical_resolution_ratio",
    "avg_bombs": "p0_avg_bombs",
    "safe_tactical_placements_per_game": "p0_safe_tactical_placements_per_game",
    "policy_entropy": "p0_avg_policy_entropy",
}
PER_EPISODE_KEYS = {
    "surviving_kill": "surviving_causal_kill",
    "own_bomb_death": "own_bomb_defeat",
    "avoidable_danger_death": "avoidable_danger_death",
    "danger_to_death": "danger_to_death",
    "mutual_death": "mutual_death",
}


def wilson(count: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = count / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def final_iteration(run_dir: Path) -> int | None:
    meta = run_dir / "phase2.json"
    if not meta.exists():
        return None
    return int(json.loads(meta.read_text())["ppo_iterations"])


def discover(run_dir: Path, final_iter: int | None = None) -> list[dict]:
    """Map checkpoint files to the phase-2 x axis (phase1 final = 0)."""
    found = []
    if (run_dir / "phase1.pt").exists():
        found.append({"name": "phase1", "iteration": 0})
    for path in run_dir.glob("phase2_it*.pt"):
        match = re.fullmatch(r"phase2_it(\d+)\.pt", path.name)
        if match:
            found.append({"name": path.stem, "iteration": int(match.group(1))})
    if (run_dir / "phase2.pt").exists():
        iteration = final_iter if final_iter is not None else final_iteration(run_dir)
        if iteration is None:
            raise SystemExit("phase2.pt found but phase2.json lacks ppo_iterations; pass --final-iter")
        if any(item["iteration"] == iteration for item in found):
            raise SystemExit(f"final phase2.pt collides with an intermediate checkpoint at iter {iteration}")
        found.append({"name": "phase2", "iteration": iteration, "final": True})
    for item in found:
        item["path"] = str(run_dir / f"{item['name']}.pt")
        item.setdefault("final", False)
    return sorted(found, key=lambda item: item["iteration"])


def select(checkpoints: list[dict], spec: str) -> list[dict]:
    """spec: 'all' or comma list of 'phase1', 'final', checkpoint names, N or A-B (phase-2 iters)."""
    if spec.strip() == "all":
        return list(checkpoints)
    chosen: dict[str, dict] = {}
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if token == "final":
            hits = [c for c in checkpoints if c["final"]]
        elif re.fullmatch(r"\d+-\d+", token):
            low, high = map(int, token.split("-"))
            hits = [c for c in checkpoints if low <= c["iteration"] <= high]
        elif token.isdigit():
            hits = [c for c in checkpoints if c["iteration"] == int(token)]
        else:
            hits = [c for c in checkpoints if c["name"] == token]
        if not hits:
            raise SystemExit(f"selector {token!r} matched no checkpoint")
        for hit in hits:
            chosen[hit["name"]] = hit
    return sorted(chosen.values(), key=lambda item: item["iteration"])


def reusable(path: Path, checkpoint_sha: str, seed: int, games: int, max_steps: int) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return (data.get("schema") == EVAL_SCHEMA
            and data.get("checkpoint_sha256") == checkpoint_sha
            and data.get("seed") == seed
            and data["summary"].get("games") == games
            and data.get("max_steps", 300) == max_steps)


def configure_device(device: str) -> None:
    """Must run before JAX is imported."""
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def run_checkpoint(item: dict, args, out_dir: Path, evaluator) -> tuple[dict, str]:
    json_path = out_dir / "checkpoints" / f"{item['name']}.json"
    checkpoint_sha = sha256(Path(item["path"]))
    if not args.force:
        candidates = [json_path] + [Path(d) / f"{item['name']}.json" for d in args.reuse_dir]
        for candidate in candidates:
            if reusable(candidate, checkpoint_sha, args.seed, args.games, args.max_steps):
                data = json.loads(candidate.read_text())
                if candidate != json_path:
                    write_json(json_path, data)
                return data, f"reused:{candidate}"
    started = time.time()
    data = evaluator.run(item["path"], args.games, args.seed, args.max_steps, per_episode=True)
    if data["checkpoint_sha256"] != checkpoint_sha:
        raise SystemExit(f"checkpoint {item['path']} changed during evaluation")
    write_json(json_path, data)
    return data, f"evaluated:{time.time() - started:.0f}s"


def run_flee_probe(item: dict, args, out_dir: Path) -> dict:
    from jax_bomb.bun_tactical_eval import run_flee_bot_probe
    path = out_dir / "flee_probe" / f"{item['name']}.json"
    checkpoint_sha = sha256(Path(item["path"]))
    if path.exists() and not args.force:
        data = json.loads(path.read_text())
        if (data.get("checkpoint_sha256") == checkpoint_sha and data.get("seed") == args.seed
                and data.get("games") == args.flee_games and data.get("max_steps") == args.max_steps):
            return data
    data = run_flee_bot_probe(item["path"], args.flee_games, args.seed, args.max_steps)
    write_json(path, data)
    return data


def opponent_defeat_count(data: dict) -> int | None:
    per = data.get("per_episode")
    if per:
        return sum(a or b for a, b in zip(per["opponent_physical_defeat"], per["opponent_causal_defeat"]))
    rate = data["summary"].get("p0_opponent_defeat_rate")
    return None if rate is None else round(rate * data["summary"]["games"])


def spawn_digest(data: dict) -> str | None:
    per = data.get("per_episode")
    if not per:
        return None
    return hashlib.sha256(json.dumps(per["spawn_cells"]).encode()).hexdigest()[:16]


def row_for(item: dict, data: dict) -> dict:
    summary = data["summary"]
    games = summary["games"]
    row = {"name": item["name"], "iteration": item["iteration"], "final": item["final"],
           "checkpoint_sha256": data["checkpoint_sha256"], "seed": data["seed"],
           "games": games, "spawn_digest": spawn_digest(data), "metrics": {}}
    for metric, key in RATE_METRICS.items():
        if metric == "killed_by_opponent":
            count = opponent_defeat_count(data)
        else:
            count = round(summary[key] * games) if key in summary else None
        row["metrics"][metric] = (None if count is None else
                                  {"rate": count / games, "count": count, "wilson_95": wilson(count, games)})
    for metric, key in VALUE_METRICS.items():
        row["metrics"][metric] = {"value": summary.get(key)}
    return row


def paired_delta(data: dict, reference: dict) -> dict | None:
    """Per-episode paired difference (same seeds/spawns) with a normal-approx 95% CI."""
    a, b = data.get("per_episode"), reference.get("per_episode")
    if not a or not b or a["spawn_cells"] != b["spawn_cells"]:
        return None
    out = {}
    for metric, key in PER_EPISODE_KEYS.items():
        diffs = [int(x) - int(y) for x, y in zip(a[key], b[key])]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
        half = 1.96 * math.sqrt(var / n)
        out[metric] = {"delta": mean, "ci95": [mean - half, mean + half],
                       "significant": (mean - half > 0) or (mean + half < 0)}
    return out


def summarize(items: list[dict], results: dict[str, dict], args) -> dict:
    rows = [row_for(item, results[item["name"]]) for item in items]
    digests = {row["spawn_digest"] for row in rows if row["spawn_digest"]}
    reference = next((i for i in items if i["name"] == args.reference), None)
    if reference is not None:
        ref_data = results[reference["name"]]
        for row in rows:
            row["paired_vs_reference"] = paired_delta(results[row["name"]], ref_data)
    return {
        "schema": SCHEMA,
        "protocol": {"curriculum": "danger_arena=1", "destructible_bricks_cleared": True,
                     "opponent": "bun.tactical_v2 (player 1)", "actor_player": 0,
                     "action_selection": "argmax", "seed": args.seed, "games": args.games,
                     "max_steps": args.max_steps,
                     "x_axis": "phase-2 iteration; phase1 final = 0; phase2.pt = ppo_iterations"},
        "run_dir": str(args.run_dir),
        "reference": args.reference if reference is not None else None,
        "spawns_identical_across_checkpoints": len(digests) <= 1,
        "checkpoints": rows,
    }


def write_csv(summary: dict, path: Path) -> None:
    fields = ["name", "iteration", "final", "games", "seed"]
    for metric in RATE_METRICS:
        fields += [metric, f"{metric}_lo", f"{metric}_hi"]
    fields += list(VALUE_METRICS)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in summary["checkpoints"]:
            record = {key: row[key] for key in fields[:5]}
            for metric in RATE_METRICS:
                value = row["metrics"][metric]
                if value:
                    record[metric] = f"{value['rate']:.4f}"
                    record[f"{metric}_lo"] = f"{value['wilson_95'][0]:.4f}"
                    record[f"{metric}_hi"] = f"{value['wilson_95'][1]:.4f}"
            for metric in VALUE_METRICS:
                value = row["metrics"][metric]["value"]
                record[metric] = "" if value is None else f"{value:.4f}"
            writer.writerow(record)


PANELS = [
    ("surviving_kill", "Surviving safe kill rate", "surviving_kill_player_rate"),
    ("safe_detonation_ratio", "Safe detonation ratio (event pooled)", "safe_detonation_ratio"),
    ("own_bomb_death", "Own-bomb death rate", "self_bomb_defeat_player_rate"),
    ("killed_by_opponent", "Killed by opponent (physical or causal)", None),
    ("avoidable_danger_death", "Avoidable danger death rate", "avoidable_danger_death_player_rate"),
    ("danger_to_death", "Danger -> death rate", None),
    ("avg_bombs", "Avg bombs / game", "avg_bombs_per_player"),
    ("tactical_resolution_ratio", "Tactical resolution ratio (event pooled)", None),
    ("policy_entropy", "Policy entropy", None),
]


def plot(summaries: list[tuple[str, dict]], teacher: dict | None, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = [("#2b6cb0", "o", "-"), ("#dd6b20", "s", "none"), ("#38a169", "D", "none"),
              ("#805ad5", "^", "none")]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))
    for ax, (metric, panel_title, teacher_key) in zip(axes.ravel(), PANELS):
        for index, (label, summary) in enumerate(summaries):
            color, marker, line = styles[index % len(styles)]
            rows = summary["checkpoints"]
            xs, ys, lo, hi = [], [], [], []
            for row in rows:
                value = row["metrics"].get(metric)
                if not value:
                    continue
                y = value.get("rate", value.get("value"))
                if y is None:
                    continue
                xs.append(row["iteration"])
                ys.append(y)
                ci = value.get("wilson_95")
                lo.append(y - ci[0] if ci else 0.0)
                hi.append(ci[1] - y if ci else 0.0)
            offset = 60 * index
            xs_plot = [x + offset for x in xs]
            if metric in RATE_METRICS:
                ax.errorbar(xs_plot, ys, yerr=[lo, hi], marker=marker, capsize=3, lw=1.6,
                            ls=line, color=color, ecolor=color, alpha=0.9, label=label)
            else:
                ax.plot(xs_plot, ys, marker=marker, lw=1.6, ls=line, color=color, label=label)
        if teacher and teacher_key in teacher:
            ax.axhline(teacher[teacher_key], ls="--", color="#e53e3e", lw=1.2,
                       label=f"rule-bot self-play = {teacher[teacher_key]:.3f}")
        ax.axvline(0, ls=":", color="#718096", lw=1)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("phase-2 iter (phase1 final = 0)")
        ax.grid(alpha=0.3)
        if metric in RATE_METRICS or metric.endswith("ratio"):
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_teacher(path: str | None) -> dict | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return data.get("pooled_teacher")


def flee_summary(items: list[dict], results: dict[str, dict]) -> dict:
    rows = []
    for item in items:
        data = results[item["name"]]
        s = data["summary"]
        rows.append({"name": item["name"], "iteration": item["iteration"], "final": item["final"],
                     "games": s["games"], "metrics": {
                         **{m: {"rate": s[f"{m}_rate"], "count": s["counts"][m],
                                "wilson_95": s["wilson_95"][m]} for m in s["counts"]},
                         "avg_bombs": {"value": s["avg_bombs"]}}})
    return {"schema": "transformer_flee_probe_sweep_v1", "checkpoints": rows}


def plot_flee(summary: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("surviving_kill", "Surviving kill rate"), ("own_bomb_death", "Own-bomb death rate"),
              ("avoidable_danger_death", "Avoidable danger death rate"),
              ("avg_bombs", "Avg bombs / game")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    rows = summary["checkpoints"]
    xs = [row["iteration"] for row in rows]
    for ax, (metric, name) in zip(axes, panels):
        values = [row["metrics"][metric] for row in rows]
        ys = [v.get("rate", v.get("value")) for v in values]
        if metric == "avg_bombs":
            ax.plot(xs, ys, marker="o", color="#2f855a")
        else:
            lo = [y - v["wilson_95"][0] for y, v in zip(ys, values)]
            hi = [v["wilson_95"][1] - y for y, v in zip(ys, values)]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, color="#2f855a")
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("phase-2 iter (phase1 final = 0)")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="directory with phase1.pt / phase2_itN.pt / phase2.pt (+ phase2.json)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", default="all",
                        help="'all' or comma list: phase1, final, names, iter N, range A-B")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'")
    parser.add_argument("--host-workers", type=int, default=16,
                        help="processes for Tactical v2 decisions + funnel labels (0 = serial)")
    parser.add_argument("--jax-cache-dir", default=None,
                        help="persistent JAX compile cache (default: <out-dir>/../.jax_cache; 'none' disables)")
    parser.add_argument("--final-iter", type=int, default=None,
                        help="x position of phase2.pt (default: phase2.json ppo_iterations)")
    parser.add_argument("--reuse-dir", action="append", default=[],
                        help="dir of existing <name>.json results; reused only if checkpoint sha,"
                             " seed, games and max_steps all match")
    parser.add_argument("--force", action="store_true", help="re-evaluate even if a matching JSON exists")
    parser.add_argument("--reference", default="phase1",
                        help="checkpoint name used for paired per-episode deltas")
    parser.add_argument("--teacher-json", default=None,
                        help="rule-bot self-play baseline JSON (pooled_teacher) drawn as dashed lines")
    parser.add_argument("--overlay", action="append", default=[],
                        help="LABEL=path/to/summary.json of another sweep to overlay on the plot")
    parser.add_argument("--flee-probe", action="store_true",
                        help="also run the supplementary on-device probe vs the JAX flee bot")
    parser.add_argument("--flee-games", type=int, default=1024)
    parser.add_argument("--plot-only", action="store_true",
                        help="rebuild summary/plot from existing per-checkpoint JSON, no evaluation")
    parser.add_argument("--dry-run", action="store_true", help="list selected checkpoints and exit")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    items = select(discover(run_dir, args.final_iter), args.checkpoints)
    if args.dry_run:
        for item in items:
            print(f"{item['iteration']:>6}  {item['name']:<16} {item['path']}")
        return
    out_dir = args.out_dir
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if args.flee_probe:
        (out_dir / "flee_probe").mkdir(parents=True, exist_ok=True)

    evaluator = None
    if not args.plot_only:
        configure_device(args.device)
        cache = args.jax_cache_dir or str(out_dir.resolve().parent / ".jax_cache")
        if cache != "none":
            from qqt_rl.training.jax_cache import configure_persistent_cache
            configure_persistent_cache(explicit=cache)
        from jax_bomb.bun_tactical_eval import TacticalOpponentEvaluator
        evaluator = TacticalOpponentEvaluator(host_workers=args.host_workers)

    results: dict[str, dict] = {}
    flee_results: dict[str, dict] = {}
    try:
        for index, item in enumerate(items, 1):
            if args.plot_only:
                path = out_dir / "checkpoints" / f"{item['name']}.json"
                if not path.exists():
                    print(f"[{index}/{len(items)}] {item['name']}: missing, skipped", flush=True)
                    continue
                results[item["name"]] = json.loads(path.read_text())
                flee_path = out_dir / "flee_probe" / f"{item['name']}.json"
                if args.flee_probe and flee_path.exists():
                    flee_results[item["name"]] = json.loads(flee_path.read_text())
                continue
            data, status = run_checkpoint(item, args, out_dir, evaluator)
            results[item["name"]] = data
            s = data["summary"]
            line = (f"[{index}/{len(items)}] it{item['iteration']:>5} {item['name']:<14} {status:<14} "
                    f"kill={s['p0_surviving_causal_kill_rate']:.3f} "
                    f"own_bomb={s['p0_own_bomb_defeat_rate']:.3f} "
                    f"avoidable={s['p0_avoidable_danger_death_rate']:.3f}")
            if args.flee_probe:
                flee = run_flee_probe(item, args, out_dir)
                flee_results[item["name"]] = flee
                line += f" | flee kill={flee['summary']['surviving_kill_rate']:.3f}"
            print(line, flush=True)
    finally:
        if evaluator is not None:
            evaluator.close()

    items = [item for item in items if item["name"] in results]
    summary = summarize(items, results, args)
    write_json(out_dir / "summary.json", summary)
    write_csv(summary, out_dir / "summary.csv")

    label = f"{args.games} games, seed {args.seed}"
    series = [(label, summary)]
    for spec in args.overlay:
        name, _, path = spec.partition("=")
        series.append((name, json.loads(Path(path).read_text())))
    plot(series, load_teacher(args.teacher_json), out_dir / "capability_curves.png",
         "Transformer checkpoints vs Bun Tactical v2 (danger_arena combat probe, actor = player 0, "
         "argmax)\nerror bars = Wilson 95% on episode-level rates; x = phase-2 iter, phase1 final = 0")
    outputs = ["summary.json", "summary.csv", "capability_curves.png"]
    flee_items = [item for item in items if item["name"] in flee_results]
    if flee_items:
        flee = flee_summary(flee_items, flee_results)
        write_json(out_dir / "flee_probe_summary.json", flee)
        plot_flee(flee, out_dir / "flee_probe_curves.png",
                  f"Supplementary probe vs JAX flee bot (phase-1 training opponent, mostly non-attacking), "
                  f"{args.flee_games} games - NOT comparable to the Tactical v2 curves")
        outputs += ["flee_probe_summary.json", "flee_probe_curves.png"]
    print(f"wrote {out_dir}: {', '.join(outputs)}", flush=True)


if __name__ == "__main__":
    main()
