from tests.external_artifacts import requires_artifacts


def test_cpu_evaluation_uses_rebound_continuation_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import ppo_scale_longrun_cpu as cpu

    monkeypatch.setattr(cpu, 'B', tmp_path)
    monkeypatch.setattr(cpu, 'R', tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['stdout'].name == str(tmp_path / 'cpu_evaluator.log')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cpu.subprocess, 'run', run)
    checkpoint = tmp_path / 'checkpoint'
    assert cpu.consume(checkpoint) == (checkpoint, 0)
    assert calls[0][1:] == [
        'scripts/ppo_scale_longrun_cpu.py', 'evaluate', 'checkpoint']
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@requires_artifacts('runs/ppo_scale_20261025/r1_longrun/PROTOCOL.json')
def test_protocol_is_ppo_only_and_equal_environment_budget():
    p = json.loads((ROOT / "runs/ppo_scale_20261025/r1_longrun/PROTOCOL.json").read_text())
    assert p["arms"] == ["ppo_base_batch", "ppo_scaled_batch"]
    assert p["common"]["dagger_coefficient"] == 0.0
    assert p["common"]["combat_kl_coefficient"] == 0.0
    b = p["batch"]
    assert b["ppo_base_batch"]["global_rollout_batch"] == 8192
    assert b["ppo_scaled_batch"]["global_rollout_batch"] == 16384
    assert b["ppo_base_batch"]["global_minibatch"] == 256
    assert b["ppo_scaled_batch"]["global_minibatch"] == 512
    assert p["budget"]["environment_steps_per_run"]["ppo_base_batch"] == p["budget"]["environment_steps_per_run"]["ppo_scaled_batch"]


def test_training_source_has_no_supervised_or_dagger_objective_path():
    source = (ROOT / "scripts/ppo_scale_longrun_train.py").read_text()
    assert "def pure_ppo_objective" in source
    assert "teacher_not_applied" in source
    assert "C.objective(q" not in source
    assert "dagger_coefficient=0." in source


@requires_artifacts('runs/dagger_rl_20261025/r18_item_chain/PPO_SCALE_HANDOFF.json')
def test_r18_formal_queue_is_stopped_and_smoke_only():
    p = json.loads((ROOT / "runs/dagger_rl_20261025/r18_item_chain/PPO_SCALE_HANDOFF.json").read_text())
    assert p["formal_environment_steps"] == 0
    assert p["technical_smoke_preserved"] is True


def test_real_ppo_objective_ignores_teacher_rows_and_has_ability_and_critic_gradients(monkeypatch):
    import jax
    import jax.numpy as j
    import numpy as np
    from scripts import ppo_scale_longrun_train as train

    def forward(p, arch, obs, state):
        n = len(obs)
        return (j.broadcast_to(p['move'], (n, 5)),
                j.broadcast_to(p['ability'], (n, 3)),
                j.zeros(n), j.broadcast_to(p['critic'], (n, 1)))

    monkeypatch.setattr(train.C.E.train, 'net_forward', forward)
    monkeypatch.setattr(train.C.E.train, 'hl_gauss_value_loss',
                        lambda logits, target: j.mean((logits[:, 0]-target)**2))
    params = dict(move=j.zeros(5), ability=j.zeros(3), critic=j.zeros(1))
    actions = j.array([[0, 2], [1, 0]])
    batch = (j.zeros((2, 1)), j.zeros((2, 1)), actions,
             j.full(2, -j.log(15.)), j.array([1., -1.]), j.ones(2),
             j.ones((2, 5), bool), j.ones((2, 3), bool),
             j.zeros((2, 5, 3), bool), j.zeros(2, bool))
    loss, metrics = train.pure_ppo_objective(params, batch)
    altered = batch[:8] + (j.ones((2, 5, 3), bool), j.ones(2, bool))
    np.testing.assert_array_equal(loss, train.pure_ppo_objective(params, altered)[0])
    grad = jax.grad(lambda p: train.pure_ppo_objective(p, batch)[0])(params)
    assert np.linalg.norm(grad['ability']) > 0
    assert np.linalg.norm(grad['critic']) > 0
    assert np.isfinite(metrics).all()
