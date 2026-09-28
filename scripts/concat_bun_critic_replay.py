#!/usr/bin/env python3
"""Concatenate compatible Bun counterfactual replay files with lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    sources = []
    schema = None
    for raw_path in args.input:
        path = Path(raw_path)
        loaded = np.load(path)
        if len(loaded["obs"]) == 0:
            continue
        keys = tuple(sorted(loaded.files))
        if schema is None:
            schema = keys
        elif keys != schema:
            raise ValueError(f"dataset schema mismatch: {path}")
        sources.append((path, {key: loaded[key] for key in loaded.files}))
    if not sources:
        raise ValueError("replay contains no states")

    merged = {
        key: np.concatenate([data[key] for _, data in sources], axis=0)
        for key in schema
    }
    split_seeds = {
        split: set(map(int, merged["seed"][merged["split_name"] == split]))
        for split in ("train", "validation", "test")
    }
    effective_split_seeds = {
        split: {seed & 0xFFFFFFFF for seed in seeds}
        for split, seeds in split_seeds.items()
    }
    raw_leakage = any(
        split_seeds[left] & split_seeds[right]
        for index, left in enumerate(split_seeds)
        for right in tuple(split_seeds)[index + 1:]
    )
    effective_leakage = any(
        effective_split_seeds[left] & effective_split_seeds[right]
        for index, left in enumerate(effective_split_seeds)
        for right in tuple(effective_split_seeds)[index + 1:]
    )
    if raw_leakage or effective_leakage:
        raise ValueError("train/validation/test seed leakage in replay")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **merged)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    manifest = {
        "schema": "bun_critic_replay_concat_v1",
        "inputs": [
            {"path": str(path), "sha256": digest(path), "states": len(data["obs"])}
            for path, data in sources
        ],
        "output": str(output),
        "output_sha256": digest(output),
        "states": len(merged["obs"]),
        "split_counts": {
            split: int((merged["split_name"] == split).sum()) for split in split_seeds
        },
        "seed_leakage": raw_leakage,
        "effective_uint32_seed_leakage": effective_leakage,
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
