"""Every natural midpoint and holdout uses its declared seeds explicitly."""
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('seed_start', [None, 2026110200])
def test_r2_natural_seed_does_not_depend_on_inherited_environment(tmp_path, monkeypatch, seed_start):
    from scripts import dagger_rl_r2_cpu as evaluator
    from scripts import retention_route_eval

    monkeypatch.setattr(evaluator, 'R', tmp_path)
    monkeypatch.setattr(evaluator, 'B', tmp_path / 'experiment')
    evaluator.B.mkdir()
    (evaluator.B / 'CPU_EVAL_PROTOCOL.json').write_text(json.dumps({'natural_seed_start': 2026102480}))
    monkeypatch.setenv('JE10_ROUTE_SEED_START', '2026101800')
    monkeypatch.setattr(retention_route_eval, 'merge_episodes', lambda task, parts: {'parts': parts})
    observed = []

    def run(command, **kwargs):
        seed = int(kwargs['env']['JE10_ROUTE_SEED_START'])
        observed.append(seed)
        (tmp_path / command[3]).write_text(json.dumps({'seed': seed}))

    monkeypatch.setattr(evaluator.subprocess, 'run', run)
    if seed_start is None:
        evaluator.natural(tmp_path / 'checkpoint')
    else:
        evaluator.natural(tmp_path / 'checkpoint', seed_start=seed_start)
    assert observed == [2026102480 if seed_start is None else seed_start] * 8
