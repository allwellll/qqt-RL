"""PPO learner trajectory with an explicit, checkpointed opponent pool."""
from __future__ import annotations
import jax
import jax.numpy as j
import numpy as np
from scripts import carrier_mechanism_engine as E, combat_item_engine as T
from scripts import combat_mix_shaping as shaping
from scripts.combat_mix_opponents import fixed_allocation,combat_scene_mask
from jax_bomb import bun_jax_bots as bots


def metadata(st,kinds,scene=None):
    n=len(st.t)
    return dict(kind=j.asarray(kinds,j.int32),scene=j.zeros(n,bool) if scene is None else scene,
        birth=j.where(st.fuse>0,st.t[:,None,None]-j.maximum(1,E.env.base.FUSE-st.fuse),-1).astype(j.int32),
        shaping=shaping.state(n))


def choose(p,ring,fixed,states,aux,keys,groups):
    # Same five-way learner RNG split and masked sampling as E.choose.
    nk,ka,kb,ko,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,5))(keys),0,1)
    mv,ab,v,vl,o,g,mm,bm=E.prediction(p,states)
    bm=bm.at[:,1:].set(bm[:,1:]&~aux['terminal'][:,None])
    lm=jax.nn.log_softmax(j.where(mm,mv,-1e9));lb=jax.nn.log_softmax(j.where(bm,ab,-1e9))
    move=jax.vmap(jax.random.categorical)(ka,lm);ability=jax.vmap(jax.random.categorical)(kb,lb)
    own=j.stack([move,ability],-1);lp=lm[j.arange(len(keys)),move]+lb[j.arange(len(keys)),ability]
    allmm,allbm=jax.vmap(E.env.legal_mask)(states)
    enemybm=allbm[:,1].at[:,1:].set(allbm[:,1,1:]&~aux['terminal'][:,None])
    enemy=j.zeros_like(own)
    idx=j.asarray(groups['bot'])
    if len(groups['bot']):
        ss=jax.tree.map(lambda x:x[idx],states)
        tiers=j.where(aux['combat_mix']['kind'][idx]==2,4,3)
        def bot(s,m,b,k,t):
            return bots.rule_bot_actions(jax.tree.map(lambda x:x[None],s),j.array([1]),
                m[None],b[None],k,t[None])[0]
        enemy=enemy.at[idx].set(jax.vmap(bot)(ss,allmm[idx,1],enemybm[idx],ko[idx],tiers))
    # Static disjoint row groups avoid evaluating all three policies on every env.
    for name,actor in [('current',p['actor']),('ring',ring),('fixed',fixed)]:
        ids=j.asarray(groups[name])
        if not len(groups[name]):continue
        ss=jax.tree.map(lambda x:x[ids],states)
        count=len(ids)
        obs=E.train.both_perspectives(ss)[count:];gv=E.train.both_states(ss)[count:]
        mv,ab,_,_=E.train.net_forward(dict(actor=actor,critic=p['critic']),'transformer',obs,gv)
        lm1=jax.nn.log_softmax(j.where(allmm[ids,1],mv,-1e9))
        lb1=jax.nn.log_softmax(j.where(enemybm[ids],ab,-1e9))
        ks=jax.vmap(lambda k:jax.random.split(k,2))(ko[ids])
        ma=jax.vmap(jax.random.categorical)(ks[:,0],lm1)
        aa=jax.vmap(jax.random.categorical)(ks[:,1],lb1)
        enemy=enemy.at[ids].set(j.stack([ma,aa],-1))
    return nk,ke,j.stack([own,enemy],1),(o,g,mm,bm,own,lp,v)


def arena_fresh(keys,bs,ba,designated,spawn_pairs):
    st,ax=T.fresh(keys,bs,ba)
    # Real native empty-hand spawns. Bank states are never teleported or relabeled.
    ar,aa=E.initial(keys,1)
    slot=jax.vmap(lambda k:jax.random.randint(jax.random.fold_in(k,0x806),(),0,len(spawn_pairs)))(keys)
    pos=j.asarray(spawn_pairs,j.float32)[slot]+.5
    ar=ar._replace(core=ar.core._replace(pos=pos,hp=j.ones_like(ar.hp),invuln=j.zeros_like(ar.core.invuln)),
        scene=j.zeros_like(ar.scene),bun_spawn_pos=pos)
    aa=dict(aa,tactics=T.events.state(len(keys)),provenance_source=j.full(len(keys),-1,j.int32))
    st=E.select_tree(designated,ar,st);ax=E.select_tree(designated,aa,ax)
    return st,ax


def transition(st,ax,actions,keys,coefficient):
    meta=ax['combat_mix'];base={k:v for k,v in ax.items() if k!='combat_mix'}
    after,na,r,done,info,parts=T.transition(st,base,actions,keys,'control')
    if coefficient:
        event,birth=shaping.chain_event(st,after,info,base,meta)
        ledger,extra,metrics=shaping.settle(event,meta['shaping'],coefficient)
    else:
        birth=meta['birth'];ledger=meta['shaping'];extra=j.zeros_like(r[:,0])
        metrics=dict(candidate=j.zeros_like(done),settled_count=j.zeros_like(st.t),pay=extra)
    r=r.at[:,0].add(extra)
    parts=parts.at[:,0,-1].set(r[:,0])
    na['combat_mix']=dict(meta,birth=birth,shaping=ledger)
    return after,na,r,done,info,parts,metrics


def collector(steps,count,seed,fraction,coefficient,spawn_pairs,weights):
    kinds=fixed_allocation(count,seed,weights)
    designated=j.asarray(combat_scene_mask(count,seed^0x806,fraction))
    groups={name:np.flatnonzero(kinds==i) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]}
    groups['bot']=np.flatnonzero(kinds<3)
    def collect(p,st,ax,keys,bs,ba,ring,fixed):
        def body(c,_):
            st,ax,key=c
            nk,ke,act,data=choose(p,ring,fixed,st,ax,key,groups)
            o,g,mm,bm,own,lp,v=data
            after,na,r,done,info,parts,sm=transition(st,ax,act,ke,coefficient)
            nv=E.prediction(p,after)[2]
            fkeys=jax.vmap(lambda k:jax.random.fold_in(k,0xF3))(nk)
            if fraction:
                fresh,fa=arena_fresh(fkeys,bs,ba,designated,spawn_pairs)
            else:fresh,fa=T.fresh(fkeys,bs,ba)
            fa=dict(fa,combat_mix=metadata(fresh,kinds,designated))
            live=~ax['terminal']
            pool=jax.nn.one_hot(ax['combat_mix']['kind'],6)
            # Counts retain explicit denominator: all ticks/live ticks/new episodes.
            diagnostics=dict(pool_ticks=pool.sum(0),pool_live_ticks=(pool*live[:,None]).sum(0),
                pool_new_episodes=(pool*done[:,None]).sum(0),
                clear_scene_ticks=ax['combat_mix']['scene'].sum(),total_ticks=j.array(count),
                resets=done.sum(),bank_resets=(done&(fa['provenance_source']>=0)).sum(),
                clear_scene_resets=(done&designated).sum(),
                chain_candidates=sm['candidate'].sum(),chain_settled=sm['settled_count'].sum(),
                chain_reward=sm['pay'].sum(),native_parts=parts[:,0].sum(0),
                learner_bomb_requests=(own[:,1]==1).sum(),enemy_bomb_requests=(act[:,1,1]==1).sum(),
                learner_deaths=info['death'][:,0].sum(),captures=info['capture'][:,0].sum())
            output=(o,g,own,lp,v,nv,r[:,0],done,mm,bm,diagnostics)
            after=E.select_tree(done,fresh,after);na=E.select_tree(done,fa,na)
            return (after,na,nk),output
        end,d=jax.lax.scan(body,(st,ax,keys),None,length=steps)
        o,g,act,lp,v,nv,r,done,mm,bm,diag=d
        adv=E.train.compute_gae(r,v,nv,done,.995,.95)
        return end,(o,g,act,lp,adv,adv+v,mm,bm),jax.tree.map(lambda x:x.sum(0),diag)
    return jax.pmap(collect,in_axes=(0,0,0,0,None,None,0,0))
