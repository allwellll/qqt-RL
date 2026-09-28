#!/usr/bin/env python3
"""Export one trusted JAX Bun transformer checkpoint to browser JSON."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--width", type=int, default=15)
    args = parser.parse_args()

    # pickle 可执行任意 Python 代码，因此只允许导出自己训练并已校验哈希的产物。
    with args.checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    params = payload.get("params", payload)
    tensors = extract_transformer(params)
    flat, index = pack_tensors(tensors)
    patch_area = params["tok"][0].shape[0] // args.channels
    patch = int(round(patch_area ** 0.5))
    if patch * patch != patch_area:
        raise ValueError("cannot infer square transformer patch size")
    name = args.name or args.checkpoint.stem
    document = {
        "meta": {
            "name": name, "display_name": name, "arch": "transformer",
            "rule": "bun", "qqt_map_id": 806,
            "move_actions": 5, "ability_actions": 3,
            "ability_encoding": ["none", "bomb", "use_item"],
            "value_min": -20.0, "value_max": 20.0,
            "obs_shape": [args.channels, args.height, args.width],
            "embed": int(params["tok"][0].shape[1]), "patch": patch,
            "depth": len(params["blocks"]), "n_players": 2,
            "source": args.checkpoint.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "flat": flat,
        "tensors": index,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
