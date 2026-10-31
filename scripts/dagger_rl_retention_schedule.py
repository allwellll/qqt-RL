"""Inactive preparation for a matched loss schedule with combat-parent KL.

Importing this file changes no live collector, objective, optimizer or owner.
The switch is fixed at update16/17; the independent Critic and its Adam moments
are preserved in DAgger-only updates by the existing tested minibatch loop.
"""
import inspect
from numbers import Integral
from scripts import dagger_rl_update_mechanism as mechanism


def branch_for_update(arm, update):
    if not isinstance(update, Integral) or isinstance(update, bool) or not 1 <= update <= 32:
        raise ValueError('fixed update must be an integer in 1..32')
    if arm == 'sequential_anchor':
        return 'dagger_anchor' if update <= 16 else 'ppo_anchor'
    if arm in ['joint_anchor', 'ppo_anchor']:
        return arm
    raise ValueError('unknown retention schedule: ' + str(arm))


def loss_for_branch(base, metrics, branch):
    if branch == 'joint_anchor':
        return base
    if branch == 'ppo_anchor':
        # Select the fresh PPO/value/entropy and common combat KL components
        # directly, rather than numerically subtracting a replay loss from base.
        return metrics[0]+.5*metrics[1]-.005*metrics[2]+.05*metrics[6]
    if branch == 'dagger_anchor':
        return .2*metrics[3]+.05*metrics[6]
    raise ValueError('unknown retention loss: ' + str(branch))


def make_update(objective, optimizer, branch, preserve_critic):
    """Reuse frozen minibatch math, replacing only the scalar loss selector.

    PPO always receives the collector's fresh on-policy batch. Historical
    labels can affect only supervised metrics, which are unselected in PPO.
    Shared Adam time advances while inactive Critic parameters/moments freeze.
    """
    namespace = dict(vars(mechanism), branch_loss=loss_for_branch)
    source = inspect.getsource(mechanism.make_update)
    exec(compile(source, '<retention-schedule-minibatch-loop>', 'exec'), namespace)
    return namespace['make_update'](objective, optimizer, branch, preserve_critic)
