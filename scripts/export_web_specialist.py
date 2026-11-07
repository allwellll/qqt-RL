"""CPU-only, hash-gated export for native-items web specialist checkpoints.

The source must be a trusted local training checkpoint, independently hashed by
its producer. A fixed mode=0 projection removes only the final mode input row.
This tool does not choose a candidate or interpret training scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                  JAX_PLATFORM_NAME="cpu", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1")

import numpy as np
import torch
from scripts.export_bun_web_model import extract_transformer, pack_tensors
from scripts.je03_native_vit_export_20261011 import NativeViT

RULESET = "bun-native-v3-items"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_checkpoint(checkpoint, expected_sha256, output, name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError("a neutral lowercase model name is required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("producer checkpoint SHA256 is required")
    checkpoint, output = Path(checkpoint), Path(output)
    if digest(checkpoint) != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch; refusing deserialization")
    with checkpoint.open("rb") as handle:
        raw = pickle.load(handle)  # Only the user's trusted, hash-verified local artifact.
    params = raw.get("params", raw)
    source_dim = int(params["state_w"].shape[0])
    channels = int(params["tok"][0].shape[0]) // 9
    abilities = int(params["heads"]["wb"][1].shape[0])
    visible_dim = {3: 37, 7: 52}.get(abilities)
    if (channels != 25 or visible_dim is None or source_dim not in (visible_dim, visible_dim + 1)
            or int(params["heads"]["wm"][1].shape[0]) != 5):
        raise ValueError("unsupported observation/action ABI")
    derived = dict(params)
    if source_dim == visible_dim + 1:
        derived["state_w"] = np.asarray(params["state_w"], np.float32)[:-1].copy()
    torch.set_num_threads(1)
    full, model = NativeViT(params).eval(), NativeViT(derived).eval()
    rng = np.random.default_rng(20261107)
    projection_error = 0.0
    with torch.no_grad():
        for _ in range(8):
            obs = torch.from_numpy(rng.normal(0, .2, (1, 25, 13, 15)).astype(np.float32))
            state = torch.from_numpy(rng.uniform(-1, 1, (1, visible_dim)).astype(np.float32))
            full_state = torch.cat([state, torch.zeros((1, 1))], dim=1) if source_dim != visible_dim else state
            for expected, actual in zip(full(obs, full_state), model(obs, state)):
                projection_error = max(projection_error, float(torch.max(torch.abs(expected - actual))))
    if projection_error >= .002:
        raise ValueError("fixed-mode projection parity failed")
    output.mkdir(parents=True, exist_ok=True)
    graph = output / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros((1, 25, 13, 15)), torch.zeros((1, visible_dim))),
                          str(graph) + ".tmp", input_names=["obs", "state"],
                          output_names=["move", "bomb", "value"], opset_version=17, dynamo=False)
    Path(str(graph) + ".tmp").replace(graph)
    flat, tensors = pack_tensors(extract_transformer(derived))
    meta = dict(name=name, display_name=name, arch="transformer", rule="bun",
                rule_version=RULESET, qqt_map_id=806, move_actions=5,
                ability_actions=abilities, ability_encoding=["none", "bomb", "use_item"],
                obs_shape=[25, 13, 15], state_dim=visible_dim, n_players=2,
                embed=int(derived["tok"][0].shape[1]), patch=3, depth=len(derived["blocks"]),
                value_min=-20, value_max=20, checkpoint_sha256=expected_sha256,
                source_global_dim=source_dim, fixed_mode=0,
                projection="drop final mode input row only" if source_dim != visible_dim else "identity",
                r3_input_encoding="r3-items-route-chain-visible-v1",
                onnx_sha256=digest(graph), onnx_file="model.onnx")
    if abilities == 7:
        meta.update(schema_version="five-item-slots-v1", action_schema="move-v1+five-item-slots-v1",
                    ability_encoding=["none", "bomb", "use_slot1", "use_slot2", "use_slot3", "use_slot4", "use_slot5"],
                    global_dim=52, observation_schema="r3-visible37+slot15-v1",
                    max_engine_slots=7, max_selectable_slots=5)
    document = dict(meta=meta, flat=flat, tensors=tensors)
    target = output / "model.json"
    target.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    receipt = dict(status="PASS", ruleset=RULESET, checkpoint_sha256=expected_sha256,
                   source_global_dim=source_dim, state_dim=visible_dim, ability_actions=abilities,
                   fixed_mode=0, projection_max_abs=projection_error,
                   files={p.name: dict(sha256=digest(p), bytes=p.stat().st_size) for p in (target, graph)})
    (output / "EXPORT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = export_checkpoint(args.checkpoint, args.sha256, args.output, args.name)
    print(json.dumps(result))
