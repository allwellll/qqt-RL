#!/usr/bin/env python3
"""Export one trusted JAX Bun checkpoint (transformer | mlp4) to browser JSON."""

from __future__ import annotations

import argparse
import base64
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def pack_tensors(tensors: dict[str, np.ndarray]) -> tuple[str, dict[str, tuple[int, int]]]:
    buffer = bytearray()
    index = {}
    for name, array in tensors.items():
        value = np.asarray(array, np.float32)
        index[name] = (len(buffer) // 4, value.size)
        buffer += value.tobytes()
    return base64.b64encode(bytes(buffer)).decode(), index


def extract_transformer(params: dict) -> dict[str, np.ndarray]:
    result = {}

    def linear(name, pair):
        result[name + "_w"] = np.asarray(pair[0], np.float32)
        result[name + "_b"] = np.asarray(pair[1], np.float32)

    linear("tok", params["tok"])
    result["pos"] = np.asarray(params["pos"][0], np.float32)
    result["state_w"] = np.asarray(params["state_w"], np.float32)
    result["state_b"] = np.asarray(params["state_b"], np.float32)
    for index, block in enumerate(params["blocks"]):
        prefix = f"b{index}"
        for key in ("ln1_g", "ln1_b", "ln2_g", "ln2_b"):
            result[f"{prefix}_{key}"] = np.asarray(block[key], np.float32)
        for key in ("q", "k", "v", "proj", "ff1", "ff2"):
            linear(f"{prefix}_{key}", block[key])
    for key in ("wm", "wb", "wv"):
        linear(f"head_{key}", params["heads"][key])
    return result


def extract_mlp4(params: dict) -> tuple[dict[str, np.ndarray], int, int]:
    """Return (tensors, hidden, n_layers) for the mlp4 architecture.

    JAX linear weight is [fan_in, fan_out] (forward does x @ w); the browser
    forward indexes W[out * in_dim + in], so every linear weight is transposed
    to row-major [out, in]. Names: w{i}/b{i}/ln{i}_g/ln{i}_b for i in 1..5,
    plus wm/bm (5 moves), wb/bb (N_BOMB abilities), wv/bv (value bins).
    """
    result: dict[str, np.ndarray] = {}
    n_layers = 0
    while f"w{n_layers + 1}" in params:
        n_layers += 1
    hidden = int(np.asarray(params["w1"]).shape[1])
    for i in range(1, n_layers + 1):
        result[f"w{i}"] = np.asarray(params[f"w{i}"], np.float32).T.copy()
        result[f"b{i}"] = np.asarray(params[f"b{i}"], np.float32)
        result[f"ln{i}_g"] = np.asarray(params[f"ln{i}_g"], np.float32)
        result[f"ln{i}_b"] = np.asarray(params[f"ln{i}_b"], np.float32)
    for name in ("wm", "wb", "wv"):
        result[name] = np.asarray(params[name], np.float32).T.copy()
    for name in ("bm", "bb", "bv"):
        result[name] = np.asarray(params[name], np.float32)
    return result, hidden, n_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--arch", default=None,
                        choices=(None, "transformer", "mlp4"),
                        help="override auto-detected architecture")
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--width", type=int, default=15)
    args = parser.parse_args()

    # pickle 可执行任意 Python 代码，因此只允许导出自己训练并已校验哈希的产物。
    with args.checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    params = payload.get("params", payload)
    arch = args.arch or ("transformer" if "tok" in params else "mlp4")
    name = args.name or args.checkpoint.stem
    common_meta = {
        "name": name, "display_name": name, "rule": "bun", "qqt_map_id": 806,
        "move_actions": 5, "ability_actions": 3,
        "ability_encoding": ["none", "bomb", "use_item"],
        "value_min": -20.0, "value_max": 20.0,
        "obs_shape": [args.channels, args.height, args.width], "n_players": 2,
        "source": args.checkpoint.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if arch == "mlp4":
        tensors, hidden, n_layers = extract_mlp4(params)
        flat, index = pack_tensors(tensors)
        document = {
            "meta": {**common_meta, "arch": "mlp4", "hidden": hidden,
                     "depth": n_layers},
            "flat": flat, "tensors": index,
        }
        _write(args.output, document)
        return

    tensors = extract_transformer(params)
    flat, index = pack_tensors(tensors)
    patch_area = params["tok"][0].shape[0] // args.channels
    patch = int(round(patch_area ** 0.5))
    if patch * patch != patch_area:
        raise ValueError("cannot infer square transformer patch size")
    document = {
        "meta": {**common_meta, "arch": "transformer",
                 "embed": int(params["tok"][0].shape[1]), "patch": patch,
                 "depth": len(params["blocks"])},
        "flat": flat,
        "tensors": index,
    }
    _write(args.output, document)


def _write(output: Path, document: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
