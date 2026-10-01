#!/usr/bin/env python3
"""Plot transformer training capability curves from eval_v2/*.json.

X-axis = phase-2 self-play iteration (phase1 end = 0, phase2_itN = N).
Teacher (rule-bot self-play) baselines drawn as dashed reference lines.
"""
import json
import os
import glob
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(Path(__file__).resolve().parents[1] / "runs/eval_v2")
OUT = str(Path(ROOT) / "training_curves.png")
N_GAMES = 64


def gstep(name):
    if name == "phase1":
        return 0
    if name.startswith("phase2_it"):
        return int(name.split("it")[1])
    return None


def wilson(p, n=N_GAMES, z=1.96):
    if p is None:
        return 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return abs(p - (centre - half))  # symmetric-ish half-width for errorbar


# ---- load actor evals ----
points = {}
for path in glob.glob(os.path.join(ROOT, "phase*.json")):
    name = os.path.basename(path)[:-5]
    g = gstep(name)
    if g is None:
        continue
    points[g] = (name, json.load(open(path))["summary"])
xs = sorted(points)
labels = [points[x][0] for x in xs]
S = [points[x][1] for x in xs]


def series(key, combine=None):
    out = []
    for s in S:
        if combine:
            out.append(combine(s))
        else:
            out.append(s.get(key, np.nan))
    return np.array(out, float)


# ---- teacher baseline ----
tb = {}
tp = os.path.join(ROOT, "rulebot_baseline.json")
if os.path.exists(tp):
    tb = json.load(open(tp))["pooled_teacher"]

# metric -> (title, actor_key or combiner, teacher_key, higher_is_better)
PANELS = [
    ("Surviving safe kill rate (offense)", "p0_surviving_causal_kill_rate",
     "surviving_kill_player_rate", True),
    ("Safe detonation ratio", "p0_safe_detonation_ratio",
     "safe_detonation_ratio", True),
    ("Own-bomb death rate", "p0_own_bomb_defeat_rate",
     "self_bomb_defeat_player_rate", False),
    ("Killed by opponent (phys+causal)", None, None, False),
    ("Avoidable danger death rate", "p0_avoidable_danger_death_rate",
     "avoidable_danger_death_player_rate", False),
    ("Danger->death rate", "p0_danger_to_death_rate", None, False),
    ("Avg bombs / game (activity)", "p0_avg_bombs", "avg_bombs_per_player", None),
    ("Tactical resolution ratio", "p0_tactical_resolution_ratio", None, True),
    ("Policy entropy (exploration)", "p0_avg_policy_entropy", None, None),
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()
RATE = {"p0_surviving_causal_kill_rate", "p0_safe_detonation_ratio",
        "p0_own_bomb_defeat_rate", "p0_avoidable_danger_death_rate",
        "p0_danger_to_death_rate", "p0_tactical_resolution_ratio"}

for ax, (title, key, tkey, hib) in zip(axes, PANELS):
    if key is None:
        y = series(None, lambda s: s["p0_opponent_physical_defeat_rate"] +
                   s["p0_opponent_causal_defeat_rate"])
        is_rate = True
    else:
        y = series(key)
        is_rate = key in RATE
    if is_rate:
        err = np.array([wilson(v) for v in y])
        ax.errorbar(xs, y, yerr=err, marker="o", capsize=3, lw=1.8,
                    color="#2b6cb0", ecolor="#a0aec0")
    else:
        ax.plot(xs, y, marker="o", lw=1.8, color="#2b6cb0")
    if tkey and tkey in tb:
        ax.axhline(tb[tkey], ls="--", color="#e53e3e", lw=1.3,
                   label=f"rule-bot teacher = {tb[tkey]:.3f}")
        ax.legend(fontsize=8, loc="best")
    ax.axvline(0, ls=":", color="#718096", lw=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("phase-2 self-play iter")
    ax.grid(alpha=0.3)
    if is_rate:
        ax.set_ylim(-0.02, 1.02)

fig.suptitle(
    "Transformer phase-2 self-play capability curves (danger_arena vs tactical rule bot, "
    "64 games/seed 20260930)\n"
    "x=0 is the phase-1 final checkpoint; dashed red = rule-bot teacher baseline",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=110)
print("wrote", OUT)
print("checkpoints plotted:", list(zip(xs, labels)))
