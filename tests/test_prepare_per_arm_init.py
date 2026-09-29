import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_bun_safe_aggression_v7 import (
    ARM_NAMES,
    resolve_arm_init,
)


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_resolve_arm_init_maps_latest_triple(tmp_path):
    init_run = tmp_path / "production_360"
    for arm in ARM_NAMES:
        _touch(init_run / arm / "latest_actor.pt", arm.encode() + b"-actor")
        _touch(init_run / arm / "latest_critic.pkl", arm.encode() + b"-critic")
        _touch(init_run / arm / "latest_target_critic.pkl", arm.encode() + b"-target")
    resolved = resolve_arm_init(init_run, "safe_medium")
    assert resolved["actor"] == init_run / "safe_medium" / "latest_actor.pt"
    assert resolved["critic"] == init_run / "safe_medium" / "latest_critic.pkl"
    assert resolved["target_critic"] == init_run / "safe_medium" / "latest_target_critic.pkl"
    for path in resolved.values():
        assert path.is_file()


def test_resolve_arm_init_rejects_missing(tmp_path):
    init_run = tmp_path / "production_360"
    _touch(init_run / "control" / "latest_actor.pt")
    # critic/target missing -> must raise
    with pytest.raises(SystemExit):
        resolve_arm_init(init_run, "control")
