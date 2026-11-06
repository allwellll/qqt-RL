"""K4 native arena transition/reward ABI and mixed-mode PPO utilities.

This is an explicit building block, not a registration or live-worker install.
Factual forced/source-batch evidence must be supplied by the real collector;
requests, logits and geometric closeness are never accepted as that evidence.
"""
from typing import NamedTuple
import jax
import jax.numpy as j

VERSION='r4-k4-sugar-respawn-collector-v2-complete-counterfactual'
BUN_MODE=0
ARENA_MODE=1
GAMMA=.995
LAMBDA=.95


class ArenaAux(NamedTuple):
    pending_batch:object
    pending_age:object
    used_life:object


def aux():
    return ArenaAux(j.full(2,-1,j.int32),j.full(2,-1,j.int32),j.zeros(2,bool))


def settle(ledger,*,forced,source_batch,kill_batch,causal_kill,death,self_death,
           eligible_victim,terminal,winner):
    """Two-player factual cashflow, protected victims excluded symmetrically.

    Forced credit is pending (not a positive reward) until the same batch
    kills within20ticks. It expires on timeout/other death/terminal; one use
    per victim life. Terminal is1200tick only, never a sugar death/respawn.
    """
    eligible_attacker=eligible_victim[::-1]
    start=forced&eligible_attacker&~ledger.used_life&(source_batch>=0)&~terminal
    pending=j.where(start,source_batch,ledger.pending_batch)
    age=j.where(start,0,j.where(ledger.pending_age>=0,ledger.pending_age+1,-1))
    matched=(pending>=0)&(kill_batch==pending)&(age<=20)&causal_kill&eligible_attacker
    kill=causal_kill&eligible_attacker
    loss=death&eligible_victim&~self_death
    own=self_death&eligible_victim
    terminal_pay=j.where(terminal&(winner>=0),j.where(j.arange(2)==winner,1.,-1.),0.)
    parts=j.stack([kill.astype(j.float32),-loss.astype(j.float32),-1.5*own,
                   .2*matched,terminal_pay],-1)
    close=death[::-1]|terminal|(age>=20)
    next_ledger=ArenaAux(j.where(close,-1,pending),j.where(close,-1,age),
                        j.where(death[::-1],False,ledger.used_life|start))
    return next_ledger,parts.sum(-1),dict(parts=parts,forced_started=start,
        forced_realized=matched,pending_expired=close&(pending>=0)&~matched,
        protected_death=death&~eligible_victim)


def arena_transition(arena,ledger,actions,key,*,evidence,items_disabled=False):
    """Real independent K.step; all reward inputs are terminal-before-reset."""
    from scripts import r4_kill_arena as native
    after,done,info=native.step(arena,actions,key,items_disabled)
    forced,source_batch,kill_batch=evidence(arena,after,info,ledger)
    # protected is indexed by victim. Factual arena_kill is by attacker.
    ledger,reward,metrics=settle(ledger,forced=forced,source_batch=source_batch,
        kill_batch=kill_batch,causal_kill=info['arena_kill'],death=info['death'],
        self_death=info['arena_self'],eligible_victim=~info['arena_protected'],
        terminal=done,winner=info['winner'])
    return after,ledger,reward,done,dict(info,k4_metrics=metrics,k4_mode=j.int8(ARENA_MODE))


def arena_inputs(arenas,seats,batches=None):
    """Visible real arena state; cage planes are zero because none exist.

    This uses the parent's legal action and item/status encodings. No arena
    map_id, causal reward ledger, future state or oracle enters the Actor.
    """
    from jax_bomb import je07_bun_env_r1_items as env
    from scripts import r3_items_observation as parent
    from scripts.r4_k4_observation import append_mode
    states=arenas.state
    danger=jax.vmap(lambda st:env.base._danger_map(st.fuse,st.wall,st.core.bomb_blast,st.brick))(states)
    # make_obs includes a Python tuple indexed by physical seat. Trace each
    # fixed seat separately, then select; a vmapped traced seat is invalid.
    observations=[jax.vmap(lambda st,dm:env.make_obs(st,p,dm))(states,danger)
                  for p in (0,1)]
    obs=j.where(seats[:,None,None,None]==0,observations[0],observations[1])
    obs=obs.at[:,14:16].set(0.)
    # A real arena ends at1200, independently of the bun lesson's horizon.
    # Keep both visible clock encodings consistent with the actual terminal.
    clock=states.t.astype(j.float32)/1200.
    obs=obs.at[:,6].set(j.broadcast_to(clock[:,None,None],obs[:,6].shape))
    globals_=[jax.vmap(lambda st:env.global_vec(st,p))(states) for p in (0,1)]
    global_state=j.where(seats[:,None]==0,globals_[0],globals_[1])
    global_state=global_state.at[:,0].set(clock)
    movement,ability=jax.vmap(env.legal_mask)(states)
    rows=j.arange(len(seats))
    obs,global_state=parent.expand_inputs(obs,global_state,states,seats,batches)
    return obs,append_mode(global_state,ARENA_MODE),movement[rows,seats],ability[rows,seats]


def gae(rewards,values,next_values,episode_done):
    """No life/death mask is accepted: only actual1200tick episode_done."""
    def step(carry,row):
        reward,value,next_value,done=row
        delta=reward+GAMMA*j.where(done,0.,next_value)-value
        advantage=delta+GAMMA*LAMBDA*(~done)*carry
        return advantage,advantage
    _,result=jax.lax.scan(step,j.zeros_like(values[0]),
        (rewards[::-1],values[::-1],next_values[::-1],episode_done[::-1]))
    return result[::-1]


def normalize_by_mode(advantages,modes,live):
    """Frozen per-mode on-policy normalization; report both actual exposures.

    No offline records are accepted. The caller must pass true on-policy live
    rows (including seat masks), and keep raw return targets for the Critic.
    """
    normalized=j.zeros_like(advantages);counts=[];mean_abs=[]
    for mode in (BUN_MODE,ARENA_MODE):
        use=live&(modes==mode);count=use.sum();den=j.maximum(count,1)
        mean=j.where(use,advantages,0.).sum()/den
        variance=j.where(use,(advantages-mean)**2,0.).sum()/den
        scaled=(advantages-mean)/j.sqrt(variance+1e-8)
        normalized=j.where(use,scaled,normalized)
        counts.append(count);mean_abs.append(j.where(use,j.abs(scaled),0.).sum()/den)
    return normalized,dict(mode_samples=j.stack(counts),mode_mean_abs_adv=j.stack(mean_abs))


def mix_counts(total):
    """Static batch split with nearest feasible40% arena; expose rounding."""
    if total<2:raise ValueError('both modes require at least two environments')
    if total%2:raise ValueError('physical-seat half allocation requires even environments')
    arena=2*round(total*.4/2)
    return total-arena,arena


def ppo_objective(params,batch):
    """Frozen PPO math using explicit K4 forward, with no shared monkeypatch.

    Advantages are normalized by mode in each actual on-policy train
    minibatch by the caller; return targets remain raw GAE+value.
    """
    from scripts.r4_k4_observation import actor_critic_forward
    from jax_bomb.jax_train import hl_gauss_value_loss
    obs,global_state,actions,old_logp,advantage,returns,move_mask,ability_mask=batch[:8]
    move,ability,_,value_logits=actor_critic_forward(params,obs,global_state)
    lm=jax.nn.log_softmax(j.where(move_mask,move,-1e9))
    la=jax.nn.log_softmax(j.where(ability_mask,ability,-1e9))
    rows=j.arange(len(old_logp))
    selected=lm[rows,actions[:,0]]+la[rows,actions[:,1]]
    ratio=j.exp(selected-old_logp)
    policy=j.maximum(-advantage*ratio,-advantage*j.clip(ratio,.9,1.1)).mean()
    value=hl_gauss_value_loss(value_logits,returns)
    entropy=-(j.sum(j.exp(lm)*lm,-1)+j.sum(j.exp(la)*la,-1)).mean()
    loss=policy+.5*value-.005*entropy
    return loss,j.array([policy,value,entropy,(old_logp-selected).mean(),(j.abs(ratio-1.)>.1).mean()])
