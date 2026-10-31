"""Frozen local evaluation input consistency; no capability threshold."""
import json
from pathlib import Path
import pytest


def test_declared_hunter_seed_matches_real_spawn_manifest():
 b=Path('runs/dagger_rl_20261024/r1v2')
 if not (b/'CPU_EVAL_PROTOCOL.json').exists():pytest.skip('local frozen experiment inputs unavailable')
 protocol=json.loads((b/'CPU_EVAL_PROTOCOL.json').read_text());spawns=json.loads((b/'cpu_eval_inputs/spawns.json').read_text())
 assert spawns['seed']==protocol['hunter_seed']
 assert spawns['games']==protocol['hunter_games_per_tier']==len(spawns['spawn_cells'])==64
