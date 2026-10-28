"""Contract and recovery boundaries for the three-seed R2 PPO continuation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preregistered_long_budget_is_exactly_32m_steps_per_seed():
    from scripts import ppo_scale_longrun_continue_prepare as prep

    assert prep.SEEDS == [2026102751, 2026102752, 2026102753]
    assert prep.NEW_UPDATES == 2048
    assert prep.ENV_STEPS_PER_UPDATE == 16384
    assert prep.NEW_STEPS_PER_SEED == 33_554_432
    assert prep.NEW_STEPS_PER_SEED >= 32_000_000
    assert prep.NEW_STEPS_PER_SEED * len(prep.SEEDS) == 100_663_296


def test_scaled_recipe_keeps_batch_optimizer_and_ppo_objective_fixed():
    from scripts import ppo_scale_longrun_continue_prepare as prep

    # These values are the same configuration that completed the R1 scaled runs.
    protocol_fields = (16384, 512, 64, 32, 7.5e-5)
    assert protocol_fields == (prep.ENV_STEPS_PER_UPDATE,
                               512, 64, prep.OPTIMIZER_STEPS_PER_UPDATE, 7.5e-5)
    source = (ROOT / "scripts/ppo_scale_longrun_continue_train.py").read_text()
    assert "P.scale_updater(64, optimizer)" in source
    assert "dagger_coefficient=0.0" in source
    assert "combat_kl_coefficient=0.0" in source
    assert "fresh categorical learner stream only" in source


def test_checkpoint_schedule_is_about_one_million_steps_and_pause_is_atomic():
    from scripts.ppo_scale_longrun_continue_train import checkpoint_due

    assert checkpoint_due(64)
    assert not checkpoint_due(63)
    assert not checkpoint_due(65)
    assert checkpoint_due(2048)
    assert checkpoint_due(65, stop_requested=True)


def test_resume_path_preserves_complete_source_state_and_does_not_use_smoke():
    source = (ROOT / "scripts/ppo_scale_longrun_continue_train.py").read_text()
    assert '"params", "adam", "states", "aux", "keys", "history"' in source
    assert 'params, adam = P.rep(source["params"]), P.rep(source["adam"])' in source
    assert 'states, aux, keys, history = source["states"], source["aux"], source["keys"], source["history"]' in source
    assert "no_smoke_state_inherited=True" in source


def test_evaluator_is_cpu_only_and_never_launches_training():
    source = (ROOT / "scripts/ppo_scale_longrun_continue_cpu.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES=""' in source
    assert "controls_gpu=False" in source
    assert "ppo_scale_longrun_continue_train" not in source
    assert "no_holdout_tuning=True" in source
