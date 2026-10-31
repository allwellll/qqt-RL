"""A3S v1: physical-seat binding; genuine Bank remains untouched P0 data.

All native two-player reward/auxiliary ledgers retain physical seat axes.
Only learner projections and the formerly learner-only threat ledger select seat.
"""
import jax
import jax.numpy as j
import numpy as np
from scripts import bomb_threat_collector_diagnostics_v1 as old
from scripts import carrier_mechanism_engine as E,combat_item_engine as T
from scripts.combat_mix_opponents import fixed_allocation,combat_scene_mask
from scripts.bomb_threat_passive_diagnostics import policy_sums
from jax_bomb import bun_jax_bots as bots

default_collector=old.collector


def seat_allocation(count,seed,fraction=.5):
    if fraction not in (0,.5):raise ValueError('A3S supports only frozen 0 or .5')
    if fraction and count%2:raise ValueError('exact half requires even env count')
    seats=np.zeros(count,np.int32)
    seats[np.random.default_rng(seed^0x5050).permutation(count)[:int(count*fraction)]]=1
    return seats


def select_player(values,seat):return values[j.arange(len(seat)),seat]
def select_perspective(values,seat):return values[j.arange(len(seat))+seat*len(seat)]
def joint_actions(own,enemy,seat):
    out=j.zeros((len(seat),2,2),own.dtype);r=j.arange(len(seat))
    return out.at[r,seat].set(own).at[r,1-seat].set(enemy)


def conditional_bank_probability(probability,seat_fraction):
    value=probability/(1-seat_fraction)
    if not 0<=value<=1:raise ValueError('Bank probability cannot fit P0 natural allocation')
    return value


def bank_selection(keys,seats,probability=.25):
    conditional=conditional_bank_probability(probability,.5)
    return (seats==0)&jax.vmap(lambda k:jax.random.bernoulli(k,conditional))(keys)


def fresh(keys,bs,ba,designated,spawn_pairs,seats):
    """Identical acquisition Bank bytes/aux; no swap, relabel or carry fabrication."""
    n=len(keys);r=j.arange(n);split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
    st,ax=T.OLD_INITIAL(split[:,0],0)
    choice=jax.vmap(lambda k:jax.random.randint(k,(),0,bs.t.size))(split[:,1])
    certified=jax.tree.map(lambda a:a[choice],bs);ca=jax.tree.map(lambda a:a[choice],ba)
    use=bank_selection(split[:,2],seats,.25)
    item=jax.vmap(lambda k:jax.random.randint(k,(),0,3))(split[:,3])
    slots=st.item_slots.at[r,seats,0].set(item.astype(st.item_slots.dtype))
    counts=st.item_counts.at[r,seats,0].set((item>0).astype(st.item_counts.dtype))
    st=st._replace(item_slots=slots,item_counts=counts,held_item=slots[:,:,0])
    st=E.select_tree(use,certified,st);ax=E.select_tree(use,ca,ax)
    ax=dict(ax,tactics=T.events.state(n),provenance_source=j.where(use,choice,-1))
    ar,aa=E.initial(keys,1)
    slot=jax.vmap(lambda k:jax.random.randint(jax.random.fold_in(k,0x806),(),0,len(spawn_pairs)))(keys)
    pos=j.asarray(spawn_pairs,j.float32)[slot]+.5
    ar=ar._replace(core=ar.core._replace(pos=pos,hp=j.ones_like(ar.hp),invuln=j.zeros_like(ar.core.invuln)),
        scene=j.zeros_like(ar.scene),bun_spawn_pos=pos)
    aa=dict(aa,tactics=T.events.state(n),provenance_source=j.full(n,-1,j.int32))
    return E.select_tree(designated,ar,st),E.select_tree(designated,aa,ax)


def inputs(states,seats):
    obs=select_perspective(E.train.both_perspectives(states),seats)
    gv=select_perspective(E.train.both_states(states),seats)
    mm,bm=jax.vmap(E.env.legal_mask)(states)
    return obs,gv,select_player(mm,seats),select_player(bm,seats)


def prediction(params,states,seats):
    o,g,mm,bm=inputs(states,seats);mv,ab,v,vl=E.train.net_forward(params,'transformer',o,g)
    return mv,ab,v,vl,o,g,mm,bm


def teacher_eligible(st,ax,seats):
    carry=select_player(st.bun_carried,seats)
    custody=select_player(ax['carry_items']['ledger']['custody'],seats)
    actual=(carry==1-seats)&custody
    native=~select_player(st.alive,1-seats)|(select_player(st.trapped,1-seats)>0)
    return select_player(st.alive,seats)&(select_player(st.trapped,seats)==0)&~ax['terminal']&j.where(carry>=0,actual,native)


def route_teacher(st,ax):
    from scripts.je10_route_oracle import labels
    from jax_bomb import bun_safety
    seats=ax['combat_mix']['learner_seat'];o,g,mm,bm=inputs(st,seats)
    target=labels(st,ax,mm,seats,o[:,5]);safe=select_player(jax.vmap(bun_safety.analyze_actions)(st).survivable,seats)
    return (target[:,:5]>0)[:,:,None]&safe&mm[:,:,None]&bm[:,None,:]&teacher_eligible(st,ax,seats)[:,None,None]


def combat_teacher(st,ax,keys):
    from scripts import je09_stage_g_combat as combat
    seats=ax['combat_mix']['learner_seat'];_,_,mm,bm=inputs(st,seats)
    target=combat.labels(st,seats,mm,bm,ax['terminal'],keys[0])
    return (target[:,:5]>0)[:,:,None]&(target[:,5:8]>0)[:,None,:]&(target[:,8]>0)[:,None,None]


def choose_with_diagnostics(p,ring,fixed,states,aux,keys,groups):
    seats=aux['combat_mix']['learner_seat'];enemyseat=1-seats
    nk,ka,kb,ko,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,5))(keys),0,1)
    mv,ab,v,vl,o,g,mm,bm=prediction(p,states,seats)
    bm=bm.at[:,1:].set(bm[:,1:]&~aux['terminal'][:,None])
    lm=jax.nn.log_softmax(j.where(mm,mv,-1e9));lb=jax.nn.log_softmax(j.where(bm,ab,-1e9))
    move=jax.vmap(jax.random.categorical)(ka,lm);ability=jax.vmap(jax.random.categorical)(kb,lb)
    own=j.stack([move,ability],-1);lp=lm[j.arange(len(keys)),move]+lb[j.arange(len(keys)),ability]
    allmm,allbm=jax.vmap(E.env.legal_mask)(states)
    enemy_mm=select_player(allmm,enemyseat);enemy_bm=select_player(allbm,enemyseat)
    enemy_bm=enemy_bm.at[:,1:].set(enemy_bm[:,1:]&~aux['terminal'][:,None]);enemy=j.zeros_like(own)
    idx=j.asarray(groups['bot'])
    if len(groups['bot']):
        ss=jax.tree.map(lambda x:x[idx],states);tiers=j.where(aux['combat_mix']['kind'][idx]==2,4,3)
        def bot(s,m,b,k,t,seat):
            return bots.rule_bot_actions(jax.tree.map(lambda x:x[None],s),seat[None],m[None],b[None],k,t[None])[0]
        enemy=enemy.at[idx].set(jax.vmap(bot)(ss,enemy_mm[idx],enemy_bm[idx],ko[idx],tiers,enemyseat[idx]))
    for name,actor in [('current',p['actor']),('ring',ring),('fixed',fixed)]:
        ids=j.asarray(groups[name])
        if not len(groups[name]):continue
        ss=jax.tree.map(lambda x:x[ids],states);obs,gv,*_=inputs(ss,enemyseat[ids])
        mv,ab,_,_=E.train.net_forward(dict(actor=actor,critic=p['critic']),'transformer',obs,gv)
        lm1=jax.nn.log_softmax(j.where(enemy_mm[ids],mv,-1e9));lb1=jax.nn.log_softmax(j.where(enemy_bm[ids],ab,-1e9))
        ks=jax.vmap(lambda k:jax.random.split(k,2))(ko[ids])
        enemy=enemy.at[ids].set(j.stack([jax.vmap(jax.random.categorical)(ks[:,0],lm1),jax.vmap(jax.random.categorical)(ks[:,1],lb1)],-1))
    return nk,ke,joint_actions(own,enemy,seats),(o,g,mm,bm,own,lp,v),policy_sums(lm,lb,~aux['terminal'])


def choose(*args):return choose_with_diagnostics(*args)[:4]


def transition(st,ax,actions,keys,coefficient):
    from scripts import bomb_threat_seat5050_reward as threat
    meta=ax['combat_mix'];seats=meta['learner_seat'];ridx=j.arange(len(seats))
    base={k:v for k,v in ax.items() if k!='combat_mix'}
    after,na,r,done,info,parts=T.transition(st,base,actions,keys,'control')
    ledger,extra,metrics,birth=threat.transition(st,after,info,base,meta,seats)
    r=r.at[ridx,seats].add(extra);parts=parts.at[ridx,seats,-1].set(select_player(r,seats))
    na['combat_mix']=dict(meta,birth=birth,bomb_threat=ledger)
    return after,na,r,done,info,parts,metrics


def metadata(st,kinds,scene,seats):return dict(old.metadata(st,kinds,scene),learner_seat=j.asarray(seats,j.int32))


def collector(steps,count,seed,fraction,coefficient,spawn_pairs,weights,seat_fraction=.5):
    if seat_fraction==0:return default_collector(steps,count,seed,fraction,coefficient,spawn_pairs,weights)
    kinds=fixed_allocation(count,seed,weights);seats=j.asarray(seat_allocation(count,seed,seat_fraction))
    designated=j.asarray(combat_scene_mask(count,seed^0x806,fraction))
    groups={name:np.flatnonzero(kinds==i) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]};groups['bot']=np.flatnonzero(kinds<3)
    def collect(p,st,ax,keys,bs,ba,ring,fixed):
        def body(c,_):
            st,ax,key=c;seat=ax['combat_mix']['learner_seat']
            nk,ke,act,data,policy_diag=choose_with_diagnostics(p,ring,fixed,st,ax,key,groups)
            o,g,mm,bm,own,lp,v=data
            after,na,r,done,info,parts,sm=transition(st,ax,act,ke,coefficient)
            nv=prediction(p,after,seat)[2]
            fkeys=jax.vmap(lambda k:jax.random.fold_in(k,0xF3))(nk)
            fst,fa=fresh(fkeys,bs,ba,designated,spawn_pairs,seats)
            fa=dict(fa,combat_mix=metadata(fst,kinds,designated,seats))
            live=~ax['terminal'];pool=jax.nn.one_hot(ax['combat_mix']['kind'],6);onehot=jax.nn.one_hot(seat,2)
            reward=select_player(r,seat);placed=select_player(info['bomb_placed'],seat)
            diagnostics=dict(pool_ticks=pool.sum(0),pool_live_ticks=(pool*live[:,None]).sum(0),pool_new_episodes=(pool*done[:,None]).sum(0),
                clear_scene_ticks=ax['combat_mix']['scene'].sum(),total_ticks=j.array(count),resets=done.sum(),
                bank_resets=(done&(fa['provenance_source']>=0)).sum(),clear_scene_resets=(done&designated).sum(),
                chain_candidates=sm['candidate'].sum(),chain_settled=sm['settled_count'].sum(),chain_reward=sm['pay'].sum(),
                native_parts=select_player(parts,seat).sum(0),learner_bomb_requests=(own[:,1]==1).sum(),
                enemy_bomb_requests=(select_player(act,1-seat)[:,1]==1).sum(),learner_deaths=select_player(info['death'],seat).sum(),
                captures=select_player(info['capture'],seat).sum(),learner_bombs_placed=placed.sum(),
                learner_bomb_requests_rejected=((own[:,1]==1)&~placed).sum(),threat_events=sm['threat_events'].sum(),
                topology_events=sm['brick_events'].sum(),topology_paid_events=sm['brick_paid_events'].sum(),
                total_reward=reward.sum(),absolute_total_reward=j.abs(reward).sum(),
                seat_ticks=onehot.sum(0),seat_live_ticks=(onehot*live[:,None]).sum(0),
                bank_seat_ticks=(onehot*(ax['provenance_source']>=0)[:,None]).sum(0),
                pool_seat_ticks=j.einsum('ni,nj->ij',pool,onehot),
                pool_seat_live_ticks=j.einsum('ni,nj->ij',pool,onehot*live[:,None]),
                bank_seat_resets=(onehot*(done&(fa['provenance_source']>=0))[:,None]).sum(0))
            diagnostics.update(policy_diag)
            output=(o,g,own,lp,v,nv,reward,done,mm,bm,diagnostics)
            return (E.select_tree(done,fst,after),E.select_tree(done,fa,na),nk),output
        end,d=jax.lax.scan(body,(st,ax,keys),None,length=steps)
        o,g,act,lp,v,nv,r,done,mm,bm,diag=d
        adv=E.train.compute_gae(r,v,nv,done,.995,.95)
        return end,(o,g,act,lp,adv,adv+v,mm,bm),jax.tree.map(lambda x:x.sum(0),diag)
    return jax.pmap(collect,in_axes=(0,0,0,0,None,None,0,0))
