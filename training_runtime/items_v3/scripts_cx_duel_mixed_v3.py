"""Half unchanged full package episodes, half explicitly marked native duels."""
import json,os
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from scripts import combat_regions_mixed_v2 as regions
from scripts import r4_k4_observation as observation
from scripts import r4_k4_collector as common
from scripts import cx_duel_reward_v1 as reward
from scripts.cx_duel_scenarios_v1 import FAMILIES
CATALOG=Path('runs/cc_train_20261105/training/duel_valid_20261107/SCENARIOS.json')

WEIGHTS=(0.,.025,.025,.35,.20,.10,.20,.05,.05)
base=regions.base
NAMES=regions.NAMES
DUEL_SCENE=9
FRACTION=float(os.environ.get("DUEL_FRACTION", ".5"))

def full_count(count):
    assert FRACTION in (.25,.5,.75)
    return 2*round(count*(1-FRACTION)/2)


def modes(count):
    assert count>=4 and count%4==0
    return np.arange(count)>=full_count(count)


def seats(count,seed):
    value=np.tile(np.array([0,1],np.int32),count//2)
    # Exact half in each mode; permutation independent of reset RNG.
    rng=np.random.default_rng(seed^0xD0E1)
    for start,end in ((0,full_count(count)),(full_count(count),count)):
        rng.shuffle(value[start:end])
    return value


def slice_tree(tree,start,end=None):
    return jax.tree.map(lambda a:a[start:end],tree)


def concatenate(a,b):
    return jax.tree.map(lambda x,y:j.concatenate([x,y],0),a,b)


def migrate_aux(aux):
    aux=regions.migrate_aux(aux);n=len(aux['terminal'])
    meta=dict(dict(duel_id=j.zeros(n,j.int32),duel_episode=j.zeros(n,j.int32),
              duel_last_explosion=j.full(n,-1000,j.int32),duel_streak=j.zeros(n,j.int32)),**aux['a3sr'])
    return dict(aux,a3sr=meta)


def scenario_state(st,keys,physical,episode,split='train'):
    """Independent valid empty-handed initial state. No source history is reused."""
    from jax_bomb import je07_native_items as items
    rows=[r for r in json.loads(CATALOG.read_text())['rows'] if r['split']==split]
    n=len(keys);r=j.arange(n)
    choice=jax.vmap(lambda k:jax.random.randint(jax.random.fold_in(k,0xD0E1),(),0,len(rows)))(keys)
    # Keep easy variants throughout, progressively unlock far/dense variants.
    if split=='train':
        hard=j.minimum(15,3+episode//4)
        family=choice//16;variant=choice%16
        choice=family*16+j.minimum(variant,hard)
    arr=lambda name,dtype:j.asarray([x[name] for x in rows],dtype)[choice]
    stats=arr('stats',j.float32);spawn=arr('spawn',j.float32)+.5
    fuse=arr('fuse',st.fuse.dtype)
    core=st.core._replace(wall=arr('wall',bool),brick=arr('brick',bool),
        pushable=j.zeros_like(st.core.pushable),push_t=j.zeros_like(st.core.push_t),
        bush=j.zeros_like(st.core.bush),crate=j.zeros_like(st.core.crate),
        rec_crate=j.zeros_like(st.core.rec_crate),fuse=fuse,
        # Pre-existing hazards belong to the opponent, never free learner credit.
        owner=j.where(fuse>0,(1-physical)[:,None,None],-1).astype(st.core.owner.dtype),
        bomb_blast=arr('blast',st.core.bomb_blast.dtype),
        blast_linger=j.zeros_like(st.core.blast_linger),brick_linger=j.zeros_like(st.core.brick_linger),
        pos=spawn,alive=j.ones_like(st.alive),hp=j.ones_like(st.hp),
        invuln=j.zeros_like(st.core.invuln),bombs_cap=stats[:,:,0].astype(st.core.bombs_cap.dtype),
        blast_cap=stats[:,:,1].astype(st.core.blast_cap.dtype),spd_g=stats[:,:,2],
        t=j.zeros_like(st.t),is_open=~arr('brick',bool).any((-2,-1)))
    # Fields inherited from a clean designated fresh constructor, not from Bank.
    slots=arr('slots',st.item_slots.dtype);counts=arr('counts',st.item_counts.dtype)
    native=jax.vmap(lambda _:items.fresh())(keys)
    st=st._replace(core=core,bun_stored=j.zeros_like(st.bun_stored),bun_loose=j.zeros_like(st.bun_loose),
        bun_carried=j.full_like(st.bun_carried,-1),bun_respawn=j.zeros_like(st.bun_respawn),
        bun_spawn_pos=spawn,held_item=slots[:,:,0],item_slots=slots,item_counts=counts,
        tactical_crate=j.zeros_like(st.tactical_crate),field_item=j.zeros_like(st.field_item),
        field_owner=j.full_like(st.field_owner,-1),field_armed=j.zeros_like(st.field_armed),
        move_status=j.zeros_like(st.move_status),status_ticks=j.zeros_like(st.status_ticks),
        slide_dir=arr('heading',st.slide_dir.dtype),last_move_dir=arr('heading',st.last_move_dir.dtype),
        trapped=j.zeros_like(st.trapped),trapped_source=j.zeros_like(st.trapped_source),
        trapped_causal=j.zeros_like(st.trapped_causal),native_items=native,
        scene=j.full_like(st.scene,DUEL_SCENE),lesson=j.zeros_like(st.lesson),
        blast_owner_linger=j.zeros_like(st.blast_owner_linger),
        blast_causal_linger=j.zeros_like(st.blast_causal_linger),
        blast_trigger_linger=j.zeros_like(st.blast_trigger_linger),
        kill_window_ticks=j.zeros_like(st.kill_window_ticks),milestones=j.zeros_like(st.milestones),
        danger_pending_ticks=j.zeros_like(st.danger_pending_ticks),
        danger_pending_avoidable=j.zeros_like(st.danger_pending_avoidable),
        danger_chain_active=j.zeros_like(st.danger_chain_active),danger_moved=j.zeros_like(st.danger_moved),
        tactical_bomb_marks=j.zeros_like(st.tactical_bomb_marks))
    ids=j.asarray([x['id'] for x in rows],j.int32)[choice]
    from scripts.cx_items_rules_v2 import supply_batch
    st,_item_supply=supply_batch(st,keys,duel=True)
    return st,ids


def bun_backend():
    original=regions.bun_backend();native=original.E.env
    natural_fresh=original.fresh

    def fresh(keys,bs,ba,designated,spawns,physical,reset_mask=None,episode=None):
        n=len(keys);half=full_count(n)
        marker=j.asarray(modes(n))
        st,ax=natural_fresh(keys,bs,ba,marker,spawns,physical,reset_mask)
        ax=migrate_aux(ax)
        episode=j.zeros(n,j.int32) if episode is None else episode
        duel,ids=scenario_state(slice_tree(st,half),keys[half:],physical[half:],episode[half:])
        st=concatenate(slice_tree(st,0,half),duel)
        ax['a3sr']['duel_id']=ax['a3sr']['duel_id'].at[half:].set(ids)
        ax['a3sr']['duel_episode']=episode
        ax['provenance_source']=ax['provenance_source'].at[half:].set(-1)
        return st,ax

    def inputs(st,physical,ledger=None):
        obs,g,mm,bm=original.inputs(st,physical,ledger)
        mode=st.scene==DUEL_SCENE
        obs=obs.at[:,14:16].set(j.where(mode[:,None,None,None],0.,obs[:,14:16]))
        clock=st.t/float(reward.HORIZON)
        obs=obs.at[:,6].set(j.where(mode[:,None,None],clock[:,None,None],obs[:,6]))
        g=g.at[:,0].set(j.where(mode,clock,g[:,0]))
        g=g.at[:,-1].set(mode.astype(g.dtype))
        return obs,g,mm,bm

    def prediction(p,st,physical,ledger=None):
        o,g,mm,bm=inputs(st,physical,ledger)
        mv,ab,v,vl=observation.actor_critic_forward(p,o,g)
        return mv,ab,v,vl,o,g,mm,bm

    def transition(st,ax,actions,keys,coefficient):
        half=full_count(len(keys))
        fa,fn,fr,fd,fi,fp,fm=original.transition(slice_tree(st,0,half),slice_tree(ax,0,half),actions[:half],keys[:half],coefficient)
        before=slice_tree(st,half);da=slice_tree(ax,half)
        after,_,di=jax.vmap(lambda s,a,k:native.step(s,a,k,auto_reset=False,return_info=True))(before,actions[half:],keys[half:])
        dr,dm=reward.settle(death=di['death'],credited_kill=di['credited_kill'],causal_kill=di['causal_kill'],
            own_defeat=di['own_bomb_defeat'],alive=after.alive,already_done=da['terminal'])
        dd=di['death'].any(-1)|(after.t>=reward.HORIZON)
        winner=j.where(dm['kill'][:,0],0,j.where(dm['kill'][:,1],1,-1)).astype(fi['winner'].dtype)
        di=dict(di,winner=j.where(dd,winner,-2))
        dn=dict(da,terminal=dd)
        physical=da['combat_mix']['learner_seat'];r=j.arange(len(physical))
        placed=di['bomb_placed'][r,physical]
        explosion=(before.fuse>0)&(after.fuse==0)&(before.core.owner==physical[:,None,None])
        exploded=explosion.any((-2,-1))
        last=da['a3sr']['duel_last_explosion'];streak=da['a3sr']['duel_streak']
        next_streak=j.where(exploded,j.where(after.t-last<=20,streak+1,1),streak)
        dn['a3sr']=dict(da['a3sr'],duel_last_explosion=j.where(exploded,after.t,last),duel_streak=next_streak)
        # Fill inherited full-game diagnostics with zero for specialist rows.
        info={k:j.concatenate([v,di[k] if k in di else j.zeros((len(physical),)+v.shape[1:],v.dtype)],0) for k,v in fi.items()}
        parts=j.zeros((len(physical),)+fp.shape[1:],fp.dtype).at[:,:,-1].set(dr)
        metrics={k:j.concatenate([v,j.zeros((len(physical),)+v.shape[1:],v.dtype)],0) for k,v in fm.items()}
        pad=lambda v:j.concatenate([j.zeros((half,)+v.shape[1:],v.dtype),v],0)
        active=~da['terminal'];one=jax.nn.one_hot(physical,2)
        families=jax.nn.one_hot(da['a3sr']['duel_id']//16,8)
        metrics.update(duel_mode_ticks=jax.nn.one_hot((st.scene==DUEL_SCENE).astype(j.int32),2),
            duel_scenario_ticks=pad(families),duel_seat_ticks=pad(one),
            duel_finished=pad(dd),duel_timeouts=pad(dd&~di['death'].any(-1)),
            duel_learner_kills=pad(dm['kill'][r,physical]),duel_learner_deaths=pad(dm['death'][r,physical]),
            duel_self_deaths=pad(dm['self_death'][r,physical]),duel_trap_enter=pad(di['native_trap_enter'][r,1-physical]),
            duel_trap_to_kill=pad(dm['kill'][r,physical]&(before.trapped[r,1-physical]>0)),
            duel_bombs=pad(placed),duel_explosions=pad(exploded),
            duel_safe_explosions=pad(exploded&after.alive[r,physical]&(after.trapped[r,physical]==0)),
            duel_streak_sum=pad(j.where(dd,next_streak,0)),duel_terminal_ticks=pad(j.where(dd,after.t,0)),
            duel_reward_parts=pad(dm['parts'][r,physical]),
            duel_package_events=pad(di['capture'].sum(-1)+di['steal'].sum(-1)),
            duel_initial_bank_rows=pad((da['provenance_source']>=0).astype(j.int32)))
        return concatenate(fa,after),concatenate(fn,dn),j.concatenate([fr,dr]),j.concatenate([fd,dd]),info,j.concatenate([fp,parts]),metrics

    source=original._league_source
    _anchor='            diagnostics.update(policy_diag)'
    assert source.count(_anchor)==1
    source=source.replace(_anchor,_anchor+"\n            diagnostics.update(items_reset_ground=(((fst.core.crate>0)|(fst.tactical_crate>0)).sum((-2,-1))*done).sum(),items_reset_natural=(done&(fa['provenance_source']<0)).sum(),items_reset_bank=(done&(fa['provenance_source']>=0)).sum(),items_reset_duel=(done&(fst.scene==9)).sum())")
    old='kinds=fixed_allocation(count,seed,weights);seats=j.asarray(seat_allocation(count,seed,seat_fraction))'
    assert source.count(old)==1
    source=source.replace(old,'kinds=fixed_allocation(count,seed,weights);seats=j.asarray(duel_seats(count,seed))')
    old='fresh(fkeys,bs,ba,designated,spawn_pairs,seats,done)'
    assert source.count(old)==1
    source=source.replace(old,"fresh(fkeys,bs,ba,designated,spawn_pairs,seats,done,ax['a3sr']['duel_episode']+1)")
    anchor='            diagnostics.update(policy_diag)';assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n            diagnostics.update({k:v.sum(0) for k,v in sm.items() if k.startswith('duel_')})")
    scope=dict(original.collector.__globals__)
    exec(compile(source,__file__,'exec'),scope)
    scope.update(fresh=fresh,transition=transition,inputs=inputs,prediction=prediction,
                 metadata=original.metadata,E=original.E,fixed_allocation=base.allocation,
                 LEAGUE_NAMES=NAMES,ATTACK=True,duel_seats=seats)
    return SimpleNamespace(collector=scope['collector'],fresh=fresh,transition=transition,inputs=inputs,
        prediction=prediction,metadata=original.metadata,E=original.E,_league_source=source)


def collector(steps,count,seed,spawn_pairs):
    impl=bun_backend();collect_bun=impl.collector(steps,count,seed,0.,0.,spawn_pairs,WEIGHTS)
    def collect(p,st,ax,keys,bs,ba,bank,generation):
        end,batch,exposure=collect_bun(p,st,ax,keys,bs,ba,None,None,bank,generation)
        mode=batch[1][...,-1].astype(j.int8);live=batch[8]
        _,normalization=common.normalize_by_mode(batch[4],mode,live)
        return end,batch[:8]+(mode,live),dict(bun=exposure,**normalization)
    return collect

updater=base.updater
