"""Route banana label qualification with frozen common optimizer and reward."""
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
from scripts.dagger_rl_r10_run import B,R,read,authorize,notify
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest
from scripts.dagger_rl_update_mechanism import make_update
from scripts.carrier_mechanism_train import preserve_critic
from scripts.dagger_rl_release_qualification import qualified_teacher
import jax.numpy as j
C=T.C
ORIGINAL_OBJECTIVE=C.objective


def bind():
    T.B,T.R,T.authorize,T.notify,T.read=B,R,authorize,notify,read
    T.ACTIVE_MIX=False
    T.bind()
    C.collector=collector


setup,initial,aggregate=T.setup,T.initial,T.aggregate
rep,one,norms,save,commit=C.rep,C.one,C.norms,C.save,C.commit


def collector(steps, qualified=True):
    def collect(p, st, ax, keys, bs, ba):
        def body(c, _):
            st, ax, key = c
            nk, ke, act, data = C.E.choose(p, st, ax, key)
            o, g, mm, bm, own, lp, v = data
            route = C.T.route_teacher(st, ax)
            item, od = C.O.teacher(st, ax)
            teacher, _ = qualified_teacher(route,item,st.held_item[:,0]==1,enabled=qualified)
            candidate=(st.held_item[:,0]==1)&~item.any((1,2))&route[:,:,2].any(1)
            qualification=j.stack([candidate,candidate&qualified,candidate&qualified&~teacher.any((1,2))],-1)
            after, na, reward, done, info, parts = C.T.transition(st, ax, act, ke, 'control')
            nv = C.E.prediction(p, after)[2]
            fresh, fa = C.T.fresh(jax.vmap(lambda k: jax.random.fold_in(k, 0xF3))(nk), bs, ba)
            # Assignment changes at episode reset; the treatment arm never does.
            fa = dict(fa, opponent_enabled=ax['opponent_enabled'])
            after = C.E.select_tree(done, fresh, after)
            na = C.E.select_tree(done, fa, na)
            carry = st.bun_carried[:, 0] == 1
            window = ~st.alive[:, 1] | (st.trapped[:, 1] > 0)
            _, ab, _, _ = C.E.train.net_forward(p, 'transformer', o, g)
            probability = jax.nn.softmax(j.where(bm, ab, -1e9))[:, 2]
            flags = j.stack([carry, st.held_item[:, 0] == 1, bm[:, 2], own[:, 1] == 2,
                j.argmax(j.where(bm, ab, -1e9), -1) == 2, probability,
                info['trap_hit'][:, 0], info['capture'][:, 0], info['death'][:, 0],
                info['steal'][:, 0] & window, info['steal'][:, 0] & ~window,
                info['pickup_loose'][:, 0], info['native_transfer'][:, 0],
                info['own_bomb_defeat'][:, 0], ~carry & ~window, ~carry & window], -1)
            live = ~ax['terminal']
            opponent = ax['opponent_enabled'] & ax['opponent_candidate']
            combat = C.combat_rows(o, g) & live
            distance = j.abs(st.pos[:, 0] - st.pos[:, 1]).sum(-1)
            exposure = j.stack([opponent, live, combat, combat & (distance <= 4),
                                 combat & (distance > 4) & (distance <= 8),
                                 combat & (distance > 8)], -1)
            return (after, na, nk), (o, g, own, lp, v, nv, reward[:, 0], done,
                mm, bm, teacher, item.any((1, 2)), live, parts[:, 0], flags, od,
                ax['provenance_source'], exposure, qualification)
        end, d = jax.lax.scan(body, (st, ax, keys), None, length=steps)
        o,g,act,lp,v,nv,reward,done,mm,bm,teacher,isitem,live,parts,flags,od,origins,exposure,qualification = d
        advantage = C.E.train.compute_gae(reward, v, nv, done, .995, .95)
        return end, (o,g,act,lp,advantage,advantage+v,mm,bm,teacher,isitem,live,origins), \
            (parts,flags,od,reward,v,advantage,origins,exposure,qualification)
    return jax.pmap(collect, in_axes=(0,0,0,0,None,None))


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
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    (pn,an),loss=arm_updater('qualified_route',opt)(p,a,batch,rb,np.asarray(jax.random.split(jax.random.PRNGKey(8),8)));jax.block_until_ready(pn)
    pp,aa=one(pn),one(an);n=norms(source['params'],pp)
    ability=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,pp['actor']['heads']['wb'],source['params']['actor']['heads']['wb'])))
    assert all(v>0 for v in n.values()) and ability>0
    assert all(np.isfinite(v).all() for v in jax.tree.leaves((pp,aa))) and np.isfinite(np.asarray(loss)).all()
    assert read(B/'REAL_TRAIN_LABEL_AUDIT.json')['passed']
    cp=commit(B/'smoke',1,pn,an,st,ax,key,2026102498,'technical',history,formal=False)
    x=pickle.load((cp/'state.pkl').open('rb'));expected=C.next_update_probe(x,bs,ba)
    atomic_json(B/'smoke/EXPECTED.json',dict(next_update=expected,adam=digest(x['adam']),history=digest(history),nonzero=n))
    code=subprocess.run([sys.executable,'-m','scripts.dagger_rl_r10_train','restore'],stdout=(B/'smoke/restore.log').open('a'),stderr=subprocess.STDOUT).returncode
    assert code==0 and read(B/'smoke/RESTORE.json')['passed'],'fresh full collector/replay/Adam update mismatch'
    atomic_json(B/'TECHNICAL_SMOKE.json',dict(passed=True,finite=True,nonzero=n,ability_head_update_l2=ability,
        ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),fresh_process_restore=True,parity_environments=64,
        state_sha256=sha(cp/'state.pkl'),total_executed_environment_steps=1536,total_executed_optimizer_steps=6,
        real_train_label_audit_sha256=sha(B/'REAL_TRAIN_LABEL_AUDIT.json'),formal_inherits_smoke=False,capability_gate=False))
    notify('R10技术smoke通过：Actor/Critic/ability真实非零、finite、fresh64下一完整collector/replay/Adam精确；立即正式固定队列。')


def restore():
    bind();C.restore()


# The formal loop below is materialized from the frozen R6 loop. Its only
# difference is only qualified route labels and passive intervention counters.

def formal():
 bind()
 items,source=setup();pr=read(B/'PROTOCOL.json');opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(pr['lr'],eps=1e-8));collectors={arm:collector(128,qualified=pr['release_qualification'][arm]) for arm in pr['arms']}
 control=B/'formal/no_update/u000';control.mkdir(parents=True,exist_ok=True)
 if not (control/'COMMITTED.json').exists():
  save(control/'state.pkl',source);save(control/'policy.pt',source['params']['actor']);atomic_json(control/'COMMITTED.json',dict(passed=True,update=0,no_update=True,state_sha256=sha(control/'state.pkl'),policy_sha256=sha(control/'policy.pt')))
 records=[]
 for seed in pr['paired_seeds']:
  for arm in pr['arms']:
   update=arm_updater(arm,opt);collect=collectors[arm]
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
    q=np.asarray(diag[8]);row['route_release_qualification']=dict(zip(['candidate','removed','emptied'],q.sum((0,1,2)).astype(int).tolist()))
    row['route_release_by_source_phase']={}
    for sn,sv in [('natural',np.asarray(diag[6])<0),('real_pickup_bank',np.asarray(diag[6])>=0)]:
     for ph,pv in [('carry',flags[...,0]>0),('empty_window',flags[...,15]>0),('empty_combat',flags[...,14]>0)]:
      ix=sv&pv;row['route_release_by_source_phase'][sn+'/'+ph]=dict(ticks=int(ix.sum()),**dict(zip(['candidate','removed','emptied'],q[ix].sum(0).astype(int).tolist())))
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
    if u==1:notify(f'R10正式首更新 {arm}/s{seed} u1 {mode} finite：Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，8192真实学生步，{ag["new_item_rows"]}道具老师行，原子SHA {sha(cp/"state.pkl")[:16]}。')
    if u==17:notify(f'R10 {arm}/s{seed} u17：{mode}，Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，中段邻点完整保存（本轮无阶段切换）；未使用历史数据作PPO。')
   final=base/'u032';receipt=dict(arm=arm,seed=seed,update=32,finite=True,state_sha256=sha(final/'state.pkl'),policy_sha256=sha(final/'policy.pt'),environment_steps=262144,optimizer_steps=1024)
   atomic_json(base/'COMPLETE.json',receipt);records.append(receipt);notify(f'DAgger/RL固定run终态 {arm}/s{seed} u32完整原子提交；262144环境步/1024优化步，立即接续冻结队列，CPU不阻塞。')
 atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=1572864,optimizer_steps=6144,old_continuous_resume=False))


if __name__=='__main__':globals()[sys.argv[1]]()
