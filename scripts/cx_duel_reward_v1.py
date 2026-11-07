"""One-life actual native death/unique kill rewards; no geometric/control payment."""
import jax.numpy as j

VERSION='duel-native-one-life-kill-v1'
KILL=16.
DEATH=16.
SELF_EXTRA=8.
HORIZON=300


def settle(*,death,credited_kill,causal_kill,own_defeat,alive,already_done):
    unique=(credited_kill|causal_kill)&death[:,::-1]&alive&~death.all(-1)[:,None]
    unique &= ~already_done[:,None]
    lost=death&~already_done[:,None]
    own=lost&(own_defeat|death.all(-1)[:,None])
    parts=j.stack([KILL*unique,-DEATH*lost,-SELF_EXTRA*own],-1)
    return parts.sum(-1),dict(parts=parts,kill=unique,death=lost,self_death=own)


def select_rewards(full,duel,mode):
    """Mode0 is identity; no specialist reward ever enters a package episode."""
    return j.where(mode[:,None],duel,full)
