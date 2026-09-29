#!/usr/bin/env python3
"""Generate the Python↔JS numeric/argmax parity fixture for the mlp4 web model.

Builds a small, deterministically-seeded mlp4 parameter set, runs it through the
SAME exporter contract (extract_mlp4 -> pack_tensors) the deployed model uses,
then computes a float64 reference forward that mirrors web/sim.js::MLP4Model
(LN population-variance eps 1e-5, LN-before-ReLU, single-layer heads, HL-Gauss
value expectation). The emitted fixture carries the model document plus a set of
fixed-input cases with expected move/bomb/value + argmax, so the Node test can
assert the browser forward matches this reference bit-for-(tol)-bit.

The synthetic model exercises the identical browser code path (weight layout,
LN math, heads, value head) as the real 6M-param checkpoint at a tiny size, so
the fixture stays small enough to commit and run in CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from jax_bomb.jax_net import init_mlp4
from scripts.export_bun_web_model import extract_mlp4, pack_tensors

# Small but structurally faithful: 5 layers (init_mlp4 hardcodes 5), tiny hidden
# and obs so the committed fixture is a few KB. obs_shape is [C, H, W].
CHANNELS, HEIGHT, WIDTH = 6, 4, 5
HIDDEN = 16
N_CASES = 5
VALUE_MIN, VALUE_MAX = -20.0, 20.0


def reference_forward(tensors: dict[str, np.ndarray], obs: np.ndarray,
                      hidden: int, depth: int, ability: int) -> dict:
    """float64 reference mirroring web/sim.js::MLP4Model.forward."""
    x = obs.astype(np.float64)
    for layer in range(1, depth + 1):
        weight = tensors[f"w{layer}"].astype(np.float64)     # [out, in]
        bias = tensors[f"b{layer}"].astype(np.float64)
        gamma = tensors[f"ln{layer}_g"].astype(np.float64)
        beta = tensors[f"ln{layer}_b"].astype(np.float64)
        y = weight @ x + bias
        mean = y.mean()
        var = ((y - mean) ** 2).mean()                       # population var
        y = (y - mean) / np.sqrt(var + 1e-5) * gamma + beta
        x = np.maximum(0.0, y)                               # LN then ReLU

    def head(name_w, name_b, out_dim):
        return (tensors[name_w].astype(np.float64) @ x
                + tensors[name_b].astype(np.float64))[:out_dim]

    move = head("wm", "bm", 5)
    bomb = head("wb", "bb", ability)
    v_logits = tensors["wv"].astype(np.float64) @ x + tensors["bv"].astype(np.float64)
    v_shift = v_logits - v_logits.max()
    probs = np.exp(v_shift) / np.exp(v_shift).sum()
    centers = VALUE_MIN + (VALUE_MAX - VALUE_MIN) * np.arange(128) / 127.0
    value = float((probs * centers).sum())
    return {
        "move": [float(v) for v in move],
        "bomb": [float(v) for v in bomb],
        "value": value,
        "move_argmax": int(np.argmax(move)),
        "bomb_argmax": int(np.argmax(bomb)),
    }


def main() -> None:
    params = init_mlp4(jax.random.PRNGKey(20260929), CHANNELS, HEIGHT, WIDTH,
                       hidden=HIDDEN)
    tensors, hidden, depth = extract_mlp4(params)
    flat, index = pack_tensors(tensors)
    ability = int(np.asarray(params["wb"]).shape[1])
    doc = {
        "meta": {
            "name": "mlp4_parity_synthetic", "display_name": "mlp4 parity",
            "arch": "mlp4", "rule": "bun", "qqt_map_id": 806,
            "move_actions": 5, "ability_actions": ability,
            "ability_encoding": ["none", "bomb", "use_item"],
            "value_min": VALUE_MIN, "value_max": VALUE_MAX,
            "obs_shape": [CHANNELS, HEIGHT, WIDTH],
            "hidden": hidden, "depth": depth, "n_players": 2,
        },
        "flat": flat, "tensors": index,
    }
    in_dim = CHANNELS * HEIGHT * WIDTH
    rng = np.random.default_rng(20260929)
    cases = []
    for _ in range(N_CASES):
        obs = rng.standard_normal(in_dim).astype(np.float32)
        expected = reference_forward(tensors, obs, hidden, depth, ability)
        cases.append({"obs": [float(v) for v in obs], **expected})

    fixture = {
        "schema": "mlp4_web_parity_fixture_v1",
        "note": "Python float64 reference vs web/sim.js MLP4Model forward",
        "doc": doc,
        "cases": cases,
    }
    out = ROOT / "web" / "test_fixtures" / "mlp4_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
