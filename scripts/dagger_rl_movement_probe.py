"""Inactive candidate: learner movement-mask probe at actual physics speed.

Importing this module changes no live environment, mask, collector or policy.
Ability masks and the opponent's movement mask are always preserved.
"""
import jax.numpy as j


def install_learner_mask(train, env, *, enabled):
    """Explicit collector-only opt-in; simulator/reward bindings stay intact."""
    if enabled:
        train.legal_mask=actual_speed_mask(train.legal_mask,env,enabled=True)
    return train.legal_mask


def actual_speed_mask(legacy, env, *, enabled):
    def encode(state):
        movement,ability=legacy(state)
        if not enabled:
            return movement,ability
        scale=env._movement_scale(state)
        core=state.core._replace(spd_g=state.core.spd_g*scale)
        physics,_=env.base.legal_mask(core)
        eligible=(state.alive[0]&(state.trapped[0]==0)
                  &(state.lesson==env.LESSON_FULL)&(scale[0]!=1)
                  &(state.bun_carried[0]>=0))
        movement=movement.at[0].set(j.where(eligible,physics[0],movement[0]))
        return movement,ability
    return encode
