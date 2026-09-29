"""反事实数据的固定 shape 分批与逐字段等价性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class StateRequest:
    split: str
    seed: int
    ticks: int


@dataclass(frozen=True)
class StateBatch:
    requests: tuple[StateRequest, ...]
    valid_count: int


def build_state_requests(seed_base: int, split_stride: int,
                         split_counts: Mapping[str, int]) -> list[StateRequest]:
    requests = []
    for split_index, (split, count) in enumerate(split_counts.items()):
        for item in range(count):
            requests.append(StateRequest(
                split=split,
                seed=seed_base + split_index * split_stride + item,
                ticks=(8, 24, 48, 72)[item % 4],
            ))
    return requests


def padded_batches(requests: Iterable[StateRequest], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = list(requests)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        valid_count = len(chunk)
        if chunk and valid_count < batch_size:
            chunk.extend([chunk[-1]] * (batch_size - valid_count))
        yield StateBatch(tuple(chunk), valid_count)


def compare_replay_arrays(reference: Mapping[str, np.ndarray],
                          candidate: Mapping[str, np.ndarray]) -> dict:
    fields = {}
    compatible = set(reference) == set(candidate)
    for key in sorted(set(reference) | set(candidate)):
        if key not in reference or key not in candidate:
            fields[key] = {"exact": False, "reason": "missing"}
            compatible = False
            continue
        left, right = np.asarray(reference[key]), np.asarray(candidate[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            fields[key] = {
                "exact": False, "shape": [list(left.shape), list(right.shape)],
                "dtype": [str(left.dtype), str(right.dtype)],
            }
            compatible = False
            continue
        exact = bool(np.array_equal(
            left, right, equal_nan=True)) if np.issubdtype(
                left.dtype, np.number) else bool(np.array_equal(left, right))
        field = {"exact": exact}
        if np.issubdtype(left.dtype, np.number) and left.size:
            field["max_abs_error"] = float(np.nanmax(
                np.abs(left.astype(np.float64) - right.astype(np.float64))))
        fields[key] = field
        compatible &= exact
    return {"compatible": bool(compatible), "fields": fields}
