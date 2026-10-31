"""Isolated R1 collector: symmetric real Bank, optional R1 event package."""
from pathlib import Path
from types import ModuleType
import pickle
import jax
import jax.numpy as j
from scripts import bomb_threat_seat5050_collector as original
from scripts import bomb_threat_a3sr_spawn as spawn
from scripts import bomb_threat_r1_reward as reward
from scripts.population_run_state import sha
E=original.E
ATTACK=False


def configure(attack=False):
    global ATTACK
    ATTACK=bool(attack)


def bank_selection(keys,seats,probability=.25):
    return jax.vmap(lambda k:jax.random.bernoulli(k,probability))(keys)


def banks(p0_states,p0_aux,manifest):
    from scripts.carrier_mechanism_analysis import merged
    path=Path(manifest['p1_file']);assert sha(path)==manifest['p1_sha256']
    items=pickle.load(path.open('rb'));p1_states=merged(items,'states');p1_aux=merged(items,'aux')
    p1_aux={**p1_aux,'provenance_source':j.arange(len(p1_states.t))}
    p0_aux={**p0_aux,'tactics':original.T.events.state(len(p0_states.t)), 'provenance_source':j.arange(len(p0_states.t))}
    assert jax.tree.structure(p0_aux)==jax.tree.structure(p1_aux)
    return jax.tree.map(j.asarray,((p0_states,p1_states),(p0_aux,p1_aux)))


def fresh(keys,bs,ba,designated,spawn_pairs,seats,reset_mask=None):
    n=len(keys);rows=j.arange(n);split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
    st,ax=original.T.OLD_INITIAL(split[:,0],0)
    candidates=[];axes=[]
    for p in (0,1):
        choice=jax.vmap(lambda k:jax.random.randint(k,(),0,bs[p].t.size))(split[:,1])
        candidates.append(jax.tree.map(lambda a:a[choice],bs[p]));axes.append(jax.tree.map(lambda a:a[choice],ba[p]))
    cert=E.select_tree(seats==1,candidates[1],candidates[0]);ca=E.select_tree(seats==1,axes[1],axes[0])
    # Match the old natural-slot inventory randomization; do not alter genuine Bank.
    item=jax.vmap(lambda k:jax.random.randint(k,(),0,3))(split[:,3])
    slots=st.item_slots.at[rows,seats,0].set(item.astype(st.item_slots.dtype));counts=st.item_counts.at[rows,seats,0].set((item>0).astype(st.item_counts.dtype))
    st=st._replace(item_slots=slots,item_counts=counts,held_item=slots[:,:,0])
    # Compensate natural candidate probability for the existing10% designated
    # clear scene exclusion. Same conditional probability for both seats.
    use=bank_selection(split[:,2],seats,.25/.9)&~designated
    st=E.select_tree(use,cert,st);ax=E.select_tree(use,ca,dict(ax,tactics=original.T.events.state(n),provenance_source=j.full(n,-1,j.int32)))
    ar,aa=E.initial(keys,1)
    slot=jax.vmap(lambda k:jax.random.randint(jax.random.fold_in(k,0x806),(),0,len(spawn_pairs)))(keys)
    pos=j.asarray(spawn_pairs,j.float32)[slot]+.5
    ar=ar._replace(core=ar.core._replace(pos=pos,hp=j.ones_like(ar.hp),invuln=j.zeros_like(ar.core.invuln)),scene=j.zeros_like(ar.scene),bun_spawn_pos=pos)
    aa=dict(aa,tactics=original.T.events.state(n),provenance_source=j.full(n,-1,j.int32))
    st=E.select_tree(designated,ar,st);ax=E.select_tree(designated,aa,ax)
    sk=jax.vmap(lambda k:jax.random.fold_in(k,0xA3))(keys)
    st,meta=spawn.apply_batch(sk,st,ax,reset_mask=reset_mask,enabled=True)
    return st,dict(ax,a3sr=meta)


def metadata(st,kinds,scene,seats):
    m=original.metadata(st,kinds,scene,seats)
    if ATTACK:m=dict(m,bomb_threat=reward.state(len(st.t)))
    return m


def transition(st,ax,actions,keys,coefficient):
    meta=ax['combat_mix'];seats=meta['learner_seat'];rows=j.arange(len(seats))
    base={k:v for k,v in ax.items() if k not in ('combat_mix','a3sr')}
    after,na,r,done,info,parts=original.T.transition(st,base,actions,keys,'control')
    if ATTACK:
        ledger,extra,metrics,birth=reward.transition(st,after,info,dict(base,r1_next_terminal=na['terminal']),meta,seats)
    else:
        from scripts import bomb_threat_seat5050_reward as oldreward
        ledger,extra,metrics,birth=oldreward.transition(st,after,info,base,meta,seats)
    r=r.at[rows,seats].add(extra);parts=parts.at[rows,seats,-1].set(original.select_player(r,seats))
    na=dict(na,combat_mix=dict(meta,bomb_threat=ledger,birth=birth),a3sr=ax['a3sr'])
    return after,na,r,done,info,parts,metrics


_source=Path('scripts/bomb_threat_seat5050_collector.py').read_text()
_source=_source.replace('fst,fa=fresh(fkeys,bs,ba,designated,spawn_pairs,seats)','fst,fa=fresh(fkeys,bs,ba,designated,spawn_pairs,seats,done)')
_scale_anchor="            fa=dict(fa,combat_mix=metadata(fst,kinds,designated,seats))"
_scale_replacement=_scale_anchor+"\n            if ATTACK:\n                credit=dict(fa['combat_mix']['bomb_threat'],r1_scale=ax['combat_mix']['bomb_threat']['r1_scale'])\n                fa=dict(fa,combat_mix=dict(fa['combat_mix'],bomb_threat=credit))"
assert _source.count(_scale_anchor)==1
_source=_source.replace(_scale_anchor,_scale_replacement)

_anchor='            diagnostics.update(policy_diag)'
_repl='''            diagnostics.update(policy_diag)
            diagnostics.update({name:(done*fa['a3sr'][name]).sum() for name in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
            # Actual terminal winner, physical-seat inventory; count once at
            # first terminal, never repeatedly during delayed settlement tail.
            terminal_first=~ax['terminal']&na['terminal']
            winner=info['winner']
            for label,mask in [('win',winner==seat),('draw',winner==-1),('loss',winner==1-seat),('unresolved',winner < -1)]:
                diagnostics['opponent_'+label]=(pool*(terminal_first&mask)[:,None]).sum(0)
            if ATTACK:
                diagnostics.update({name:sm[name].sum() for name in sm if name.startswith('r1_') or name=='approach'})'''
assert _source.count(_anchor)==1
_impl=ModuleType('r1_seat_collector');exec(compile(_source.replace(_anchor,_repl),__file__,'exec'),_impl.__dict__)
_impl.fresh=fresh;_impl.transition=transition;_impl.metadata=metadata
for _name in ('seat_allocation','inputs','choose','choose_with_diagnostics','route_teacher','combat_teacher','prediction'):
    globals()[_name]=getattr(_impl,_name)

def collector(*args,**kwargs):
    _impl.ATTACK=ATTACK
    return _impl.collector(*args,**kwargs)
