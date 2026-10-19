"""Successor candidate: one multiplier for verified return replay labels.

The denominator remains all valid joint labels. Acquisition/navigation/opening
contributions, legal joint sets and native label authority stay unchanged.
"""
import jax.numpy as jnp
from scripts.prestock_joint_oracle import joint_set_loss


def weighted_joint_loss(move,ability,legal_move,legal_ability,target,active,
                        returning,factor=2.,coefficient=1.):
    base,stats=joint_set_loss(move,ability,legal_move,legal_ability,target,active,coefficient)
    valid=active&(target&legal_move[..., :,None]&legal_ability[...,None,:]).any((-2,-1))
    selected=valid&returning
    extra,_=joint_set_loss(move,ability,legal_move,legal_ability,target,selected,coefficient)
    fraction=selected.sum()/jnp.maximum(valid.sum(),1)
    return base+(factor-1.)*fraction*extra,stats
