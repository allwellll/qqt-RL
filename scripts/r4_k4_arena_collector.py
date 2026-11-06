"""Real on-policy native K4 arena collector, with factual batch settlement."""
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
from scripts import r4_k4_collector as C,r4_k4_runtime as runtime
from scripts import r4_k4_observation as mode,r4_kill_arena as native
from scripts.combat_mix_opponents import fixed_allocation

# Match parental conditional category ratios. The first two metadata labels
# share native tier3 in the existing physical-seat collector; disclose this
# rather than claiming a production JS Hunter implementation.
WEIGHTS=(.5*5/70,.5*35/70,.5*30/70,.25,1/6,1/12)


class Aux(NamedTuple):
    reward:object
    birth:object
    trap_batch:object
    batches:object
    counters:object
    last_placed:object


def auxiliary():
    return Aux(C.aux(),j.full((13,15),-1,j.int32),j.full(2,-1,j.int32),
        j.full((13,15),-1,j.int32),j.zeros(2,j.int32),j.full(2,-1000,j.int32))


def transition(arena,aux,actions,key):
    # Native respawn is the first operation of the actual step. Use that same
    # state for evidence and life protection, never the pre-respawn dead shell.
    before=native.respawn(arena)
    after,done,info=native.step(before,actions,key,False)
    old=before.state;new=after.state
    rows=j.arange(2);cell=j.clip(j.floor(old.pos).astype(j.int32),j.array([0,0]),j.array([12,14]))
    added=(new.fuse>0)&~((old.fuse>0)&(new.core.owner==old.core.owner))
    birth=j.where(added,old.t,j.where(new.fuse>0,aux.birth,-1))
    # Explosion ancestry needs new immediate-chain bombs as well as old ones.
    resolved_birth=j.where((info['k4_resolve_owner']>=0)&(aux.birth>=0),aux.birth,-1)
    for p in (0,1):
        at=tuple(cell[p]);installed=info['bomb_placed'][p]
        resolved_birth=resolved_birth.at[at].set(j.where(installed,old.t,resolved_birth[at]))
    source=runtime.unique_explosion_batches(info,new.pos,resolved_birth)
    newly_trapped=(old.trapped==0)&(new.trapped>0)
    trap=j.where(newly_trapped[::-1],source,aux.trap_batch)
    # A contact pop is a factual kill, but does not establish a bomb batch.
    pop=info['native_trap_popper'][::-1]>=0
    kill_batch=j.where(pop,-1,j.where(old.trapped[::-1]>0,trap,source))
    installed=info['bomb_placed']&(new.fuse[cell[:,0],cell[:,1]]>0)
    def forced_for(attacker):
        def measure(_):
            own=j.array([attacker],j.int32)
            def batch(st):return jax.tree.map(lambda a:a[None],st)
            without=remove_bomb_for_counterfactual(new,cell[attacker])
            previous,_=runtime.safe_exits(batch(without),own)
            current,_=runtime.safe_exits(batch(new),own)
            safe=runtime.native_safety().analyze_selected_actions(old,actions).selected_survivable[attacker]
            return previous[0].any()&~current[0].any()&safe
        return jax.lax.cond(installed[attacker],measure,lambda _:j.array(False),None)
    forced=j.stack([forced_for(p) for p in (0,1)])
    eligible=~info['arena_protected']
    credit,reward,metrics=C.settle(aux.reward,forced=forced,
        source_batch=j.where(forced,old.t,-1),kill_batch=kill_batch,
        causal_kill=info['arena_kill'],death=info['death'],self_death=info['arena_self'],
        eligible_victim=eligible,terminal=done,winner=info['winner'])
    counters=aux.counters;last=aux.last_placed;batches=aux.batches
    for p in (0,1):
        placed=info['bomb_placed'][p]
        counter=counters[p]+(placed&(old.t-last[p]>=8)).astype(j.int32)
        batches=j.where(added&(new.core.owner==p),counter,batches)
        counters=counters.at[p].set(counter);last=last.at[p].set(j.where(placed,old.t,last[p]))
    batches=j.where(new.fuse>0,batches,-1)
    trap=j.where(info['death'][::-1],-1,trap)
    return after,Aux(credit,birth,trap,batches,counters,last),reward,done,dict(info,k4_metrics=metrics)


def exact_allocation(count,seed):
    """Exactly half bot and half selfplay; ratios within each half rounded."""
    if count%2:raise ValueError('arena requires even physical-seat/mode allocation')
    expected=np.asarray(WEIGHTS)*count;values=np.floor(expected).astype(int)
    for group in (slice(0,3),slice(3,6)):
        deficit=count//2-values[group].sum()
        ids=np.arange(6)[group]
        for index in ids[np.argsort(-(expected[group]-values[group]),kind='stable')[:deficit]]:values[index]+=1
    kinds=np.repeat(np.arange(6),values).astype(np.int32)
    np.random.default_rng(seed^0x4A4).shuffle(kinds)
    return kinds


def collector(steps,count,seed):
    runtime.prepare();runtime.native_safety()
    from jax_bomb import bun_jax_bots as bots
    kinds=exact_allocation(count,seed)
    groups={name:np.flatnonzero(kinds==i) for i,name in ((3,'current'),(4,'ring'),(5,'fixed'))}
    groups['bot']=np.flatnonzero(kinds<3)
    seats=np.zeros(count,np.int32);seats[np.random.default_rng(seed^0x5050).permutation(count)[:count//2]]=1
    seats=j.asarray(seats);rows=j.arange(count)
    def prediction(p,arenas,aux,seat):
        obs,glob,mm,bm=C.arena_inputs(arenas,seat,dict(bomb_batch=aux.batches))
        move,ability,value,_=mode.actor_critic_forward(p,obs,glob)
        return obs,glob,mm,bm,move,ability,value
    def collect(p,arenas,aux,keys,ring,fixed,new_update):
        def body(carry,_):
            arenas,aux,keys=carry
            # Respawn before observing/sampling, matching the JS arena loop.
            arenas=jax.vmap(native.respawn)(arenas)
            nk,ka,kb,ko,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,5))(keys),0,1)
            obs,glob,mm,bm,mv,ab,value=prediction(p,arenas,aux,seats)
            lm=jax.nn.log_softmax(j.where(mm,mv,-1e9));la=jax.nn.log_softmax(j.where(bm,ab,-1e9))
            own=j.stack([jax.vmap(jax.random.categorical)(ka,lm),jax.vmap(jax.random.categorical)(kb,la)],-1)
            lp=lm[rows,own[:,0]]+la[rows,own[:,1]]
            _,_,emm,ebm,_,_,_=prediction(p,arenas,aux,1-seats)
            enemy=j.zeros_like(own);idx=j.asarray(groups['bot'])
            if len(idx):
                states=jax.tree.map(lambda a:a[idx],arenas.state)
                tiers=j.where(j.asarray(kinds)[idx]==2,4,3)
                def bot(st,seat,mm,bm,key,tier):
                    return bots.rule_bot_actions(jax.tree.map(lambda a:a[None],st),seat[None],mm[None],bm[None],key,tier[None])[0]
                enemy=enemy.at[idx].set(jax.vmap(bot)(states,1-seats[idx],emm[idx],ebm[idx],ko[idx],tiers))
            for name,actor in (('current',p['actor']),('ring',ring),('fixed',fixed)):
                ids=j.asarray(groups[name])
                if not len(ids):continue
                ar=jax.tree.map(lambda a:a[ids],arenas);ax=jax.tree.map(lambda a:a[ids],aux)
                o,g,_,_=C.arena_inputs(ar,1-seats[ids],dict(bomb_batch=ax.batches))
                move,ability,_,_=mode.forward(actor,o,g)
                l0=jax.nn.log_softmax(j.where(emm[ids],move,-1e9));l1=jax.nn.log_softmax(j.where(ebm[ids],ability,-1e9))
                ks=jax.vmap(lambda k:jax.random.split(k,2))(ko[ids])
                enemy=enemy.at[ids].set(j.stack([jax.vmap(jax.random.categorical)(ks[:,0],l0),jax.vmap(jax.random.categorical)(ks[:,1],l1)],-1))
            actions=j.zeros((count,2,2),j.int32).at[rows,seats].set(own).at[rows,1-seats].set(enemy)
            after,na,reward,done,info=jax.vmap(transition)(arenas,aux,actions,ke)
            nv=prediction(p,after,na,seats)[-1]
            fresh_keys=jax.vmap(lambda k:jax.random.fold_in(k,0xF3))(nk)
            fresh=runtime.batch_fresh(fresh_keys,new_update=new_update)
            fa=jax.vmap(lambda _:auxiliary())(keys)
            def select(mask,new,old):return jax.tree.map(lambda a,b:j.where(mask.reshape(mask.shape+(1,)*(a.ndim-mask.ndim)),a,b),new,old)
            # Match the bun live denominator: death, not immobility, defines
            # an alive actor. Frozen trapped masks only allow wait/noop.
            live=arenas.state.alive[rows,seats]
            metric=dict(reward=reward[rows,seats],done=done,live=live,
                placed=info['bomb_placed'][rows,seats],death=info['death'][rows,seats],
                kills=info['arena_kill'][rows,seats],self_death=info['arena_self'][rows,seats],
                protected_death=info['k4_metrics']['protected_death'][rows,seats],
                forced=info['k4_metrics']['forced_started'][rows,seats],
                forced_realized=info['k4_metrics']['forced_realized'][rows,seats],
                parts=info['k4_metrics']['parts'][rows,seats],
                pool_ticks=jax.nn.one_hot(j.asarray(kinds),6),seat_ticks=jax.nn.one_hot(seats,2),
                mode=j.ones(count,j.int8),map_id=arenas.map_id)
            data=(obs,glob,own,lp,value,nv,reward[rows,seats],done,mm,bm,metric)
            return (select(done,fresh,after),select(done,fa,na),nk),data
        end,data=jax.lax.scan(body,(arenas,aux,keys),None,length=steps)
        o,g,actions,lp,value,nv,reward,done,mm,bm,metrics=data
        advantage=C.gae(reward,value,nv,done)
        return end,(o,g,actions,lp,advantage,advantage+value,mm,bm),metrics
    return collect


def remove_bomb_for_counterfactual(state,cell):
    """Remove all bomb source fields atomically in evidence-only state.

    The safety ABI treats owner>=0 as an existing source even at fuse0.
    Leaving owner after clearing only fuse invents an immediate ghost source.
    This private diagnostic branch never changes the actual environment.
    """
    at=tuple(cell)
    return state._replace(core=state.core._replace(
        fuse=state.fuse.at[at].set(0),
        owner=state.core.owner.at[at].set(-1),
        bomb_blast=state.core.bomb_blast.at[at].set(0)))
