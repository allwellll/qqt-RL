"""Frozen Bank curves and fixed backward/wait Bank-P; all failures retained."""
import pickle,time
import numpy as np
import jax
import jax.numpy as j
from scripts.dagger_rl_run import B,R,read,save,notify,authorize
from scripts.population_run_state import atomic_json,sha
from scripts.carrier_mechanism_analysis import bank,merged,summarize_paths,branch_moves
from scripts import combat_item_engine as T,carrier_mechanism_engine as E


def aligned_next_values(values, tail, horizon):
    """Align before trimming: the last partial chunk has an observed next V."""
    return np.concatenate([values[1:], tail[None]])[:horizon]


def reward_consistent_transition(st,ax,actions,keys):
    # Match each tested policy's reward family, including deferred refund state.
    codes=ax['eval_arm'];old={k:v for k,v in ax.items() if k!='eval_arm'}
    after,na,r,done,info,parts=T.transition(st,old,actions,keys,'control')
    names=E.ops.COMPONENTS
    positive=parts[:,:,names.index('item_positive')]
    gp=j.minimum(.5*parts[:,:,names.index('glue_attributed_hit')],positive)
    paid={'glue':gp,'banana':positive-gp};deltas={}
    enemy=1-j.arange(2)[None]
    bad=info['death']|info['drop']|info['carrier_death']|info['mutual_death'][:,None]|(after.trapped>0)
    lost=bad|(~((after.bun_carried==enemy)&after.alive)&~info['capture'])
    for family in ['banana','glue']:
        before=old['tactics'];new=na['tactics']
        pay=new[family+'_gross']-before[family+'_gross']
        net=pay-j.where(lost,before[family+'_balance']+pay,0.)
        oldnet=paid[family]-j.where(lost,before[family+'_legacy_balance']+paid[family],0.)
        deltas[family]=net-oldnet
    extra=j.where(codes[:,None]==2,deltas['banana'],j.where(codes[:,None]==3,deltas['glue'],0.))
    r=r+extra;parts=parts.at[:,:,-2].set(extra).at[:,:,-1].set(r);na['eval_arm']=codes
    return after,na,r,done,info,parts


def run():
 authorize();T.configure();assert jax.local_device_count()==8
 E.transition=reward_consistent_transition
 items=bank();assert (B/'TRAINING_COMPLETE.json').exists()
 control=B/'formal/no_update/u000'
 files=sorted((B/'formal').glob('*_s*/u*/COMMITTED.json'))
 dev=[control]+[p.parent for p in files if read(p)['update'] in [1,8,16,17,24,32]]
 finals=[control]+[p.parent for p in files if read(p)['update']==32]
 assert len(dev)==55 and len(finals)==10
 pred=jax.pmap(lambda p,s:E.prediction(p,s)[2])
 normal=jax.pmap(lambda p,c:E.rollout(p,c[0],c[1],c[2],128,ended=c[3]))
 branch=jax.pmap(lambda p,c,f:E.rollout(p,c[0],c[1],c[2],128,ended=c[3],first_moves=f))
 rep=lambda tr:jax.tree.map(lambda a:j.stack([j.asarray(a)]*8),tr)
 for split,cps in [('dev',dev),('holdout',finals)]:
  chosen=[x for x in items if x['record']['split']==split];selected=[x for x in chosen for _ in range(8)];n=len(selected)
  st=merged(selected,'states');ax=dict(merged(selected,'aux'),tactics=T.events.state(n),eval_arm=j.zeros(n,j.int32));keys=np.array([np.asarray(jax.random.fold_in(jax.random.PRNGKey(2026102317),i)) for i in range(n)])
  for label,paths in [('native',cps),('bankP_backward',finals),('bankP_wait',finals)]:
   pending=[p for p in paths if not (p/f'{label}_{split}.json').exists()]
   for start in range(0,len(pending),8):
    authorize();group=pending[start:start+8];padded=group+[group[0]]*(8-len(group));pars=[]
    for cp in padded:
     assert sha(cp/'state.pkl')==read(cp/'COMMITTED.json')['state_sha256'];pars.append(pickle.load((cp/'state.pkl').open('rb'))['params'])
    p=jax.tree.map(lambda *a:j.stack(a),*pars)
    repax=rep(ax);codes=j.array([2 if cp.parent.name.startswith('banana_') else 3 if cp.parent.name.startswith('glue_') else 0 for cp in padded]);repax['eval_arm']=j.broadcast_to(codes[:,None],(8,n))
    carry=(rep(st),repax,rep(keys),j.zeros((8,n),bool));datas=[]
    if label!='native':
     moves,dist=branch_moves(carry[0],carry[1],E,jax);forced=np.asarray(moves)[...,1 if label=='bankP_backward' else 2]
    for ci in range(20):
     carry,data=branch(p,carry,forced) if ci==0 and label!='native' else normal(p,carry)
     jax.block_until_ready(carry[0].t);datas.append(tuple(np.asarray(a) for a in jax.device_get(data)))
     atomic_json(B/'BANK_STATUS.json',dict(split=split,label=label,models=[str(cp.relative_to(B)) for cp in group],ticks=(ci+1)*128))
     if np.asarray(carry[3]).all():break
    tail_all=np.asarray(pred(p,carry[0]))
    for slot,cp in enumerate(group):
     full=tuple(np.concatenate([a[i][slot] for a in datas]) for i in range(len(datas[0])))
     data=tuple(a[:2528] for a in full)
     summary,arrays=summarize_paths(data,chosen,8,n)
     # Reward-consistent critic diagnostics are descriptive; capability uses actual deliveries/deaths.
     rr,vv,terminal,flags,dd,act,lp,valid,components=data
     tail=tail_all[slot]
     nv=aligned_next_values(full[1],tail,2528)
     gae=np.concatenate([np.asarray(E.train.compute_gae(rr[t:t+128],vv[t:t+128],nv[t:t+128],terminal[t:t+128],.995,.95)) for t in range(0,len(rr),128)])
     backwards=np.flatnonzero((np.diff(dd[:,0])>1e-5)&valid[:-1,0]&valid[1:,0]&flags[:-1,0,3]&flags[1:,0,3])
     tick=int(backwards[0]) if len(backwards) else None
     fixed=dict(source=chosen[0]['record']['source_episode'],repeat=0,rule='first source repeat0, never selected by outcome',first_regressing_tick=tick,diagnostic_scope='distance regression; not full safe-oracle disagreement')
     if tick is not None:fixed.update(action=act[tick,0].tolist(),distance_before=float(dd[tick,0]),distance_after=float(dd[tick+1,0]),reward=float(rr[tick,0]),GAE=float(gae[tick,0]),GAE_next=float(gae[tick+1,0]),components=components[tick,0].tolist())
     summary['fixed_first_regression']=fixed
     summary['GAE_scope']='reconstructed actual policy-specific reward/Critic and128-step boundaries, not historical optimizer advantage or Q'
     if label!='native':summary['actual_intervention_rows']=int((forced[slot]>=0).sum())
     if cp in finals and label=='native':save(cp/f'fixed_{split}_trajectory.pkl',dict(rewards=rr[:,0],values=vv[:,0],GAE=gae[:,0],distance=dd[:,0],actions=act[:,0],valid=valid[:,0],components=components[:,0],terminal=terminal[:,0],source=chosen[0]['record']))
     summary.update(policy_sha256=sha(cp/'policy.pt'),split=split,label=label,map_limitation='map806 P0 allfar only',
       intervention='none' if label=='native' else 'one legal progress-opposing/wait move; future RNG matched; immediate deaths/drops included; not survivor-conditioned',holdout_used_for_tuning=False)
     save(cp/f'{label}_{split}.pkl',arrays);atomic_json(cp/f'{label}_{split}.json',summary)
    notify(f'固定Bank评估批终态：{split}/{label} {min(start+8,len(pending))}/{len(pending)}，不改模型。')
 atomic_json(B/'NATIVE_BANK_COMPLETE.json',dict(passed=True,dev_models=55,holdout_models=10,bankP_models_per_split=20,all_failures_retained=True))

if __name__=='__main__':run()
