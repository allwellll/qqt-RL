#!/usr/bin/env python
"""Pre-registered screen/confirmation scorer (reports/eval_guided_optimization_20261002.md §4).

Usage:
  score_eval_screen.py screen  --base <cells/base> <cells/cand> [...]   # §4.2 quick screen
  score_eval_screen.py confirm --base <cells/base> <cells/cand>        # §4.3 paired bootstrap CI
Each cells dir holds tactical_v2.json + hunter_{hard,normal,easy}.json written with --attack-trace.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jax_bomb import attack_trace as at  # noqa: E402

CELLS = ("tactical_v2", "hunter_hard", "hunter_normal", "hunter_easy")
SURVIVING = {"clean_kill", "kill_then_died"}
DIED = {"kill_then_died", "trade", "self_kill", "killed_by_bot"}


def load_traces(path: Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    if "attack_trace" in d:
        return d["attack_trace"]["episodes"]
    return [e["attack_trace"] for e in d["episodes"]]


def per_game(traces: list[dict]) -> dict[str, np.ndarray]:
    rows = [at.classify(t) for t in traces]
    cls = [r["class"] for r in rows]
    return {
        "S": np.array([c in SURVIVING for c in cls], np.float64),
        "T": np.array([c == "trade" for c in cls], np.float64),
        "D": np.array([c == "self_kill" for c in cls], np.float64),
        "alive": np.array([c not in DIED for c in cls], np.float64),
        "B": np.array([r["bombs"] for r in rows], np.float64),
        "P": np.array([bool(r["funnel"]["pressure_bomb"]) for r in rows], np.float64),
    }


def load_cells(cells_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    return {c: per_game(load_traces(Path(cells_dir) / f"{c}.json")) for c in CELLS}


def cell_stats(g: dict[str, np.ndarray]) -> dict:
    return {"games": len(g["S"]), "S": int(g["S"].sum()), "T": int(g["T"].sum()), "D": int(g["D"].sum()),
            "survival": float(g["alive"].mean()), "B": float(g["B"].mean()), "P": float(g["P"].mean())}


def composite(stats: dict) -> float:
    return sum(s["S"] - s["T"] - s["D"] for s in stats.values()) / stats[CELLS[0]]["games"]


def screen(base: dict, cand: dict) -> dict:
    """§4.2: per-cell no-collapse + clear gain not coming only from fewer self-kills."""
    bs = {c: cell_stats(base[c]) for c in CELLS}
    cs = {c: cell_stats(cand[c]) for c in CELLS}
    fails = []
    for c in CELLS:
        b, x = bs[c], cs[c]
        if x["S"] < b["S"] - max(3, math.ceil(0.25 * b["S"])):
            fails.append(f"{c}:S")
        if x["D"] > b["D"] + 6:
            fails.append(f"{c}:D")
        if x["B"] < 0.75 * b["B"]:
            fails.append(f"{c}:B")
        if x["P"] < b["P"] - 0.10:
            fails.append(f"{c}:P")
    c_base, c_cand = composite(bs), composite(cs)
    games = bs[CELLS[0]]["games"]
    sum_s = sum(cs[c]["S"] for c in CELLS)
    web_hn = cs["hunter_hard"]["S"] + cs["hunter_normal"]["S"]
    gain = c_cand - c_base >= 6 / games - 1e-12
    attack_gain = sum_s >= sum(bs[c]["S"] for c in CELLS) + 3 or web_hn >= 4
    if not gain:
        fails.append("gain:C")
    if not attack_gain:
        fails.append("gain:S")
    return {"pass": not fails, "fails": fails, "C": c_cand, "C_base": c_base, "dC": c_cand - c_base,
            "rank_key": [c_cand, cs["tactical_v2"]["S"], web_hn], "cells": cs, "base_cells": bs}


def _score(g: dict[str, np.ndarray]) -> np.ndarray:
    return g["S"] - g["T"] - g["D"]


def confirm(base: dict, cand: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """§4.3: per-game paired bootstrap (same spawn index), joint over four cells."""
    n = len(base[CELLS[0]]["S"])
    for c in CELLS:
        if len(base[c]["S"]) != n or len(cand[c]["S"]) != n:
            raise ValueError(f"unpaired game counts in {c}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d_c = sum(_score(cand[c]) - _score(base[c]) for c in CELLS)
    boot_c = d_c[idx].mean(1)
    out = {"games": n, "dC": float(d_c.mean()), "dC_ci": [float(np.quantile(boot_c, q)) for q in (.025, .975)],
           "cells": {}}
    ok = out["dC_ci"][0] > 0
    for c in CELLS:
        d_s = cand[c]["S"] - base[c]["S"]
        boot = d_s[idx].mean(1)
        ci = [float(np.quantile(boot, q)) for q in (.025, .975)]
        b_ok = cand[c]["B"].mean() >= 0.75 * base[c]["B"].mean()
        ok = ok and ci[0] > -0.05 and b_ok
        out["cells"][c] = {"dS": float(d_s.mean()), "dS_ci": ci, "B": float(cand[c]["B"].mean()),
                           "B_base": float(base[c]["B"].mean()), "B_ok": bool(b_ok)}
    out["pass"] = bool(ok)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "confirm"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out")
    ap.add_argument("cands", nargs="+")
    a = ap.parse_args()
    base = load_cells(Path(a.base))
    res = {}
    for cand in a.cands:
        res[cand] = (screen if a.mode == "screen" else confirm)(base, load_cells(Path(cand)))
    if a.mode == "screen":
        order = sorted(res, key=lambda k: res[k]["rank_key"], reverse=True)
        for k in order:
            r = res[k]
            s = " ".join(f"{c[:6]}:S{r['cells'][c]['S']}/D{r['cells'][c]['D']}/T{r['cells'][c]['T']}"
                         f"/B{r['cells'][c]['B']:.1f}/P{r['cells'][c]['P']:.2f}" for c in CELLS)
            print(f"{'PASS' if r['pass'] else 'fail'} C={r['C']:+.3f} dC={r['dC']:+.3f} {Path(k).name:24s} {s} "
                  f"{','.join(r['fails'])}")
    else:
        for k, r in res.items():
            print(f"{'PASS' if r['pass'] else 'fail'} {k} dC={r['dC']:+.3f} CI={r['dC_ci']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
