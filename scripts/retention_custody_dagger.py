"""Scale only set-valued DAgger rows from real short-window custody returns."""
import json
from pathlib import Path

import jax.numpy as jnp

BASELINE_MULTIPLIER = 2.0
CANDIDATE_MULTIPLIER = 4.0
WINDOW_TICKS = 120


def return_window_mask(valid, kind, preparation, actual_custody, custody_age):
    valid, kind, preparation, actual_custody, custody_age = map(
        jnp.asarray, (valid, kind, preparation, actual_custody, custody_age))
    return (valid & (kind == 3) & ~preparation & actual_custody
            & (custody_age >= 0) & (custody_age < WINDOW_TICKS))


def make_replay_loss(multiplier):
    if multiplier not in (BASELINE_MULTIPLIER, CANDIDATE_MULTIPLIER):
        raise ValueError("multiplier must be one of the preregistered arms")

    def replay_loss(params, replay):
        from jax_bomb import jax_train as train
        from scripts import je09_stage_h5c_dagger as legacy
        from scripts.return_dagger_weight import weighted_joint_loss

        required = {"actual_custody", "custody_age", "return_window"}
        if not required.issubset(replay):
            raise ValueError("custody provenance is required for return weighting")
        move, ability, _, _ = train.net_forward(params, "transformer", replay["obs"], replay["gv"])
        selected = return_window_mask(
            replay["valid"], replay["kind"], replay["preparation"],
            replay["actual_custody"], replay["custody_age"])
        selected &= replay["return_window"]
        return weighted_joint_loss(
            move / legacy.TEMPERATURE, ability / legacy.TEMPERATURE,
            replay["legal"], replay["ability_legal"], replay["joint_target"],
            replay["valid"], selected, multiplier, legacy.COEFFICIENT)[0]

    replay_loss.return_dagger_multiplier = multiplier
    return replay_loss


def install(run):
    run = Path(run)
    plan = json.loads((run / "plan.json").read_text())
    setting = plan["return_dagger_weight"]
    multiplier = float(setting["multiplier"])
    assert setting["eligible_source"] == "actual custody and age < 120 from je09_stage_h5c_dagger.labels"
    assert plan["return_expert"]["window"] == WINDOW_TICKS
    assert not plan.get("phase_conditioning", False)
    from scripts import je09_stage_h5c_dagger as dagger

    replay_loss = make_replay_loss(multiplier)
    dagger.replay_loss = replay_loss
    return dict(
        passed=True,
        multiplier=multiplier,
        baseline_multiplier=BASELINE_MULTIPLIER,
        eligibility_source=setting["eligible_source"],
        window_ticks=WINDOW_TICKS,
        loss="set-valued legal joint-action probability; existing all-label denominator",
        reward_changed=False,
        replay_selection_changed=False,
        query_budget_changed=False,
        in_memory_only=True,
    )
