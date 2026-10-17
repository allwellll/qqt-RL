"""Export only the critic parameter subtree from a JE09 frozen state."""
import pickle
import sys
from pathlib import Path

import numpy as np


def flatten(tree, prefix=""):
    rows = {}
    if isinstance(tree, dict):
        items = [(str(key), tree[key]) for key in sorted(tree)]
    elif isinstance(tree, (tuple, list)):
        items = [(str(index), value) for index, value in enumerate(tree)]
    else:
        return {prefix.replace("/", "__"): np.asarray(tree)}
    for key, value in items:
        name = f"{prefix}/{key}" if prefix else key
        rows.update(flatten(value, name))
    return rows


if __name__ == "__main__":
    state_path, out_path = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    params = state["live"]["params"]["critic"]
    arrays = flatten(params)
    np.savez_compressed(out_path, **arrays)
    with np.load(out_path) as exported:
        if set(exported.files) != set(arrays):
            raise ValueError("critic parameter names changed during export")
        for name, value in arrays.items():
            if not np.array_equal(exported[name], value):
                raise ValueError(f"critic parameter mismatch: {name}")
    print(f"verified={len(arrays)} critic arrays")
