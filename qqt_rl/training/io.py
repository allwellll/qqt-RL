"""训练产物哈希、有限值检查与原子提交。"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: object) -> None:
    """先落同目录临时文件再 replace，读者永远看不到半份 manifest。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    """checkpoint 先完整复制到临时文件，再以原子 rename 发布。"""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    shutil.copy2(source_path, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination_path)
    directory_fd = os.open(destination_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def checkpoint_is_finite(path: str | Path) -> bool:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    value = value.get("params", value) if isinstance(value, dict) else value
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif not np.isfinite(np.asarray(item)).all():
            return False
    return True
