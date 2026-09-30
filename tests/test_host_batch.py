import json
from pathlib import Path

import numpy as np

from jax_bomb.bun_tactical_family import TacticalFamilyBot, TacticalFamilyConfig
from jax_bomb.bun_tactical_labels import label_batch
from qqt_rl.bots.host_batch import HostBatcher

FIXTURE = Path(__file__).parent / "fixtures" / "bun_rule_bot_cases.json"


def _states():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    return [case["state"] for case in cases] * 3


def test_serial_batcher_matches_fresh_bot_per_state_and_label_batch():
    states = _states()
    batcher = HostBatcher(0, bot_player=1, label_player=0)
    actions, labels = batcher(states)
    expected = np.asarray([TacticalFamilyBot(TacticalFamilyConfig()).decide(s, 1) for s in states])
    assert np.array_equal(actions, expected)
    ref = label_batch(states, np.zeros(len(states), np.int32))
    assert labels[:, 0].tolist() == [int(l.safe_attack_available) for l in ref]
    assert labels[:, 1].tolist() == [l.enemy_escape_count for l in ref]
    assert labels[:, 2].tolist() == [int(l.safe_escape_exists) for l in ref]


def test_worker_pool_is_order_preserving_and_identical_to_serial():
    states = _states()
    serial = HostBatcher(0)(states)
    pool = HostBatcher(3)
    try:
        parallel = pool(states)
    finally:
        pool.close()
    assert np.array_equal(serial[0], parallel[0])
    assert np.array_equal(serial[1], parallel[1])
