import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, seed: int, split: str) -> None:
    np.savez(path, obs=np.zeros((1, 1), np.uint8),
             seed=np.asarray([seed], np.int64),
             split_name=np.asarray([split]))


def test_concat_rejects_cross_split_seed_leakage(tmp_path):
    first = tmp_path / "train.npz"
    second = tmp_path / "validation.npz"
    output = tmp_path / "merged.npz"
    manifest = tmp_path / "manifest.json"
    _write(first, 1234, "train")
    _write(second, 1234, "validation")
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/concat_bun_critic_replay.py"),
        "--input", str(first), "--input", str(second),
        "--output", str(output), "--manifest", str(manifest),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "train/validation/test seed leakage" in result.stderr


def test_concat_accepts_disjoint_split_seeds(tmp_path):
    first = tmp_path / "train.npz"
    second = tmp_path / "validation.npz"
    output = tmp_path / "merged.npz"
    manifest = tmp_path / "manifest.json"
    _write(first, 1234, "train")
    _write(second, 5678, "validation")
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/concat_bun_critic_replay.py"),
        "--input", str(first), "--input", str(second),
        "--output", str(output), "--manifest", str(manifest),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert output.is_file() and manifest.is_file()


def test_concat_rejects_uint32_alias_across_splits(tmp_path):
    first = tmp_path / "train.npz"
    second = tmp_path / "test.npz"
    output = tmp_path / "merged.npz"
    manifest = tmp_path / "manifest.json"
    _write(first, 1, "train")
    _write(second, 2**32 + 1, "test")
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/concat_bun_critic_replay.py"),
        "--input", str(first), "--input", str(second),
        "--output", str(output), "--manifest", str(manifest),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not output.exists()
