"""Current-student auxiliary DAgger behavior; never a PPO action replacement."""
import jax
import jax.numpy as j


def auxiliary_action(move, ability, move_mask, ability_mask, sampled, logp, *, greedy):
    if not greedy:
        return sampled, logp
    ml=jax.nn.log_softmax(j.where(move_mask,move,-1e9))
    al=jax.nn.log_softmax(j.where(ability_mask,ability,-1e9))
    own=j.stack([j.argmax(ml,-1),j.argmax(al,-1)],-1)
    rows=j.arange(len(own))
    return own,ml[rows,own[:,0]]+al[rows,own[:,1]]
