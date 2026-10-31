"""R9 learning-rate screen with identical saved Adam and on-policy pipeline."""
import os
import sys
import subprocess
import pickle
import numpy as np
import jax
import optax
import time
import json
from scripts import dagger_rl_r6_train as T
from scripts.dagger_rl_r9_run import B,R,read,authorize,notify
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest
from scripts.dagger_rl_update_mechanism import make_update
from scripts.carrier_mechanism_train import preserve_critic
C=T.C
ORIGINAL_OBJECTIVE=C.objective


def bind():
    T.B,T.R,T.authorize,T.notify,T.read=B,R,authorize,notify,read
    T.ACTIVE_MIX=False
    T.bind()


setup,initial,collector,aggregate=T.setup,T.initial,T.collector,T.aggregate
rep,one,norms,save,commit=C.rep,C.one,C.norms,C.save,C.commit


def arm_updater(arm,opt):
    protocol=read(B/'PROTOCOL.json')
    rate=protocol['learning_rate_by_arm'][arm]
    assert protocol['dagger_coefficient_by_arm'][arm]==.2
    if rate!=protocol['lr']:
        opt=optax.chain(optax.clip_by_global_norm(protocol['gradient_clip']),
                        optax.adam(rate,eps=1e-8))
    return C.updater('anchor',opt)


def smoke():
    bind();items,source=setup();st,ax,key,bs,ba=initial(2026102498,items)
    p,a=rep(source['params']),rep(source['adam'])
    (st,ax,key),batch,diag=collector(8)(p,st,ax,key,bs,ba)
    history,rb,stats=aggregate(batch,None,2026102498,1);rb=tuple(v[:,:64] for v in rb)
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8));rows={};outputs={}
    keys=np.asarray(jax.random.split(jax.random.PRNGKey(8),8))
    for arm in read(B/'PROTOCOL.json')['arms']:
        (pn,an),loss=arm_updater(arm,opt)(p,a,batch,rb,keys);jax.block_until_ready(pn)
        pp,aa=one(pn),one(an);n=norms(source['params'],pp)
        ability=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,pp['actor']['heads']['wb'],source['params']['actor']['heads']['wb'])))
        assert all(v>0 for v in n.values()) and ability>0
        assert all(np.isfinite(v).all() for v in jax.tree.leaves((pp,aa))) and np.isfinite(np.asarray(loss)).all()
        rows[arm]=dict(finite=True,nonzero=n,ability_head_update_L2=ability,ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()))
        outputs[arm]=(pn,an)
    assert digest(one(outputs['control'][0]))!=digest(one(outputs['lr_quarter'][0])),'learning-rate treatment did not reach optimizer'
    pn,an=outputs['control'];cp=commit(B/'smoke',1,pn,an,st,ax,key,2026102498,'technical',history,formal=False)
    x=pickle.load((cp/'state.pkl').open('rb'));expected=C.next_update_probe(x,bs,ba)
    atomic_json(B/'smoke/EXPECTED.json',dict(next_update=expected,adam=digest(x['adam']),history=digest(history),nonzero=rows['control']['nonzero']))
    code=subprocess.run([sys.executable,'-m','scripts.dagger_rl_r9_train','restore'],stdout=(B/'smoke/restore.log').open('a'),stderr=subprocess.STDOUT).returncode
    assert code==0 and read(B/'smoke/RESTORE.json')['passed'],'fresh same-GPU full update mismatch'
    atomic_json(B/'TECHNICAL_SMOKE.json',dict(passed=True,finite=True,arms=rows,same_fresh_batch=True,actual_treatment_nonzero=True,
        fresh_process_restore=True,parity_environments=64,state_sha256=sha(cp/'state.pkl'),
        total_executed_environment_steps=1536,total_executed_optimizer_steps=8,formal_inherits_smoke=False,capability_gate=False))
    notify('R9新技术smoke通过：两学习率真实Actor/Critic/ability非零且参数不同；fresh64下一完整collector/replay/Adam精确，立即正式固定队列。')


def restore():
    bind();C.restore()


# The formal loop below is materialized from the frozen R6 loop. Its only
# optimization difference is arm_updater and the recorded learning rate.

def formal():
 bind()
 items,source=setup();pr=read(B/'PROTOCOL.json');opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(pr['lr'],eps=1e-8));collect=collector(128)
 control=B/'formal/no_update/u000';control.mkdir(parents=True,exist_ok=True)
 if not (control/'COMMITTED.json').exists():
  save(control/'state.pkl',source);save(control/'policy.pt',source['params']['actor']);atomic_json(control/'COMMITTED.json',dict(passed=True,update=0,no_update=True,state_sha256=sha(control/'state.pkl'),policy_sha256=sha(control/'policy.pt')))
 records=[]
 for seed in pr['paired_seeds']:
  for arm in pr['arms']:
   update=arm_updater(arm,opt)
   authorize();base=B/'formal'/f'{arm}_s{seed}';base.mkdir(parents=True,exist_ok=True)
   if (base/'COMPLETE.json').exists():records.append(read(base/'COMPLETE.json'));continue
   p=rep(source['params']);a=rep(source['adam']);st,ax,key,bs,ba=initial(seed,items);start=0;history=None
   last=sorted(base.glob('u*/COMMITTED.json'))
   if last:
    cp=last[-1].parent;assert sha(cp/'state.pkl')==read(cp/'COMMITTED.json')['state_sha256'];x=pickle.load((cp/'state.pkl').open('rb'));start=x['update'];p=rep(x['params']);a=rep(x['adam']);st=x['states'];ax=x['aux'];key=x['keys'];history=x['history']
   else:commit(base,0,p,a,st,ax,key,seed,arm)
   if start==0:
    assert digest(one(p))==digest(source['params']) and digest(one(a))==digest(source['adam'])
    atomic_json(base/'FRESH_PARENT.json',dict(passed=True,parent_input_sha256=sha(B/'inputs/parent.pkl'),params_exact=True,Adam_exact=True,smoke_not_inherited=True))
   began=time.time()
   for u in range(start+1,33):
    authorize();before=one(p);mode='anchor'
    (st,ax,key),batch,diag=collect(p,st,ax,key,bs,ba);history,rb,ag=aggregate(batch,history,seed,u)
    (p,a),loss=update(p,a,batch,rb,np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),u),8)));jax.block_until_ready(p)
    after=one(p);n=norms(before,after);assert n['actor']>0 and (n['critic']==0 if mode=='dagger' else n['critic']>0)
    assert np.isfinite(np.asarray(loss)).all() and all(np.isfinite(v).all() for v in jax.tree.leaves(after))
    fn=['carry','banana_inventory','ability2_legal','sample_release','greedy_release','release_probability_sum','learner_any_item_contact','delivery','death','window_stock','free_stock','loose','transfer','self_defeat','empty_combat','empty_window']
    flags=np.asarray(diag[1]);od=np.asarray(diag[2]);item=np.asarray(batch[9]);carry=flags[...,0]>0
    row=dict(arm=arm,seed=seed,update=u,mode=mode,finite=True,update_l2=n,environment_steps=u*8192,optimizer_steps=u*32,aggregation=ag,elapsed_seconds=time.time()-began,effective_steps_this_process=(u-start)*8192,steps_per_second=(u-start)*8192/(time.time()-began),loss=np.asarray(loss).mean((0,1)).tolist(),ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),ability_head_update_l2=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,after['actor']['heads']['wb'],before['actor']['heads']['wb']))),events=dict(zip(fn,flags.sum((0,1,2)).tolist())),item_opportunities=dict(release=int(od[...,2].sum()),contact=int(od[...,3].sum()),mean_forecast_net=float(od[...,4][item].mean()) if item.any() else None),carry_inventory_ticks=int((carry&(flags[...,1]>0)).sum()),carry_release=int((carry&(flags[...,3]>0)).sum()),reward_components=np.asarray(diag[0]).sum((0,1,2)).tolist(),opponent_exposure=dict(current=0,ring=0,fixed=8192,kind='JAX native tier3'),GAE_mean=float(np.asarray(diag[-1]).mean()),GAE_variance=float(np.asarray(diag[-1]).var()))
    row['dagger_coefficient']=pr['dagger_coefficient_by_arm'][arm]
    row['learning_rate']=pr['learning_rate_by_arm'][arm]
    exposure=np.asarray(diag[7]);selected=exposure[...,0]>0
    row['opponent_exposure']=dict(current=0,ring=0,fixed=8192,rule3=int((~selected).sum()),historical_parent=int(selected.sum()),kind='JAX rule3/frozen common-parent categorical; fixed historical, not current/ring/JS')
    row['combat_support']={name:dict(rule3=int(((exposure[...,i]>0)&~selected).sum()),historical_parent=int(((exposure[...,i]>0)&selected).sum())) for i,name in enumerate(['unused','live','combat','near_le4','mid_4to8','far_gt8']) if i>0}
    row['GAE_mean']=float(np.asarray(diag[5]).mean());row['GAE_variance']=float(np.asarray(diag[5]).var())
    row['anchor_KL']=float(np.asarray(loss)[...,7].mean());row['anchor_eligible_fraction']=float(np.asarray(loss)[...,8].mean())
    origins=np.asarray(diag[6]);row['source_exposure']={}
    for name,valid in [('natural',origins<0),('real_pickup_bank',origins>=0)]:
     row['source_exposure'][name]=dict(ticks=int(valid.sum()),events={k:float(flags[...,i][valid].sum()) for i,k in enumerate(fn)},item_release_labels=int(od[...,2][valid].sum()),item_contact_labels=int(od[...,3][valid].sum()))
    with (base/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    # Every update is atomic for a safe user stop; evaluator consumes declared points only.
    cp=commit(base,u,p,a,st,ax,key,seed,arm,history,row)
    atomic_json(B/'STATUS.json',dict(stage='FORMAL',**row))
    if u==1:notify(f'R9正式首更新 {arm}/s{seed} u1 {mode} finite：Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，8192真实学生步，{ag["new_item_rows"]}道具老师行，原子SHA {sha(cp/"state.pkl")[:16]}。')
    if u==17:notify(f'R9 {arm}/s{seed} u17：{mode}，Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，中段邻点完整保存（本轮无阶段切换）；未使用历史数据作PPO。')
   final=base/'u032';receipt=dict(arm=arm,seed=seed,update=32,finite=True,state_sha256=sha(final/'state.pkl'),policy_sha256=sha(final/'policy.pt'),environment_steps=262144,optimizer_steps=1024)
   atomic_json(base/'COMPLETE.json',receipt);records.append(receipt);notify(f'DAgger/RL固定run终态 {arm}/s{seed} u32完整原子提交；262144环境步/1024优化步，立即接续冻结队列，CPU不阻塞。')
 atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=1572864,optimizer_steps=6144,old_continuous_resume=False))


if __name__=='__main__':globals()[sys.argv[1]]()
