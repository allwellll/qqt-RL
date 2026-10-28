"""Bounded native Bank curve evaluation; eight independent model slots on eight GPUs."""
import pickle,time
import numpy as np
from scripts.carrier_mechanism_run import B,read,save,notify,authorize
from scripts.carrier_mechanism_analysis import bank,merged,setup,summarize_paths
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest


def run():
    jax,E=setup();import jax.numpy as jnp
    items=bank();protocol=read(B/'PROTOCOL.json')['C'];assert (B/'C_TRAINING_COMPLETE.json').exists()
    files=sorted((B/'C').glob('*_s*/u*/COMMITTED.json'));control=B/'C/no_update/u000'
    # All u0 parameters/Adam are identical; evaluate once, retain explicit alias receipts.
    dev=[control]+[p.parent for p in files if read(p)['update']>0]
    finals=[control]+[p.parent for p in files if read(p)['update']==16]
    assert len(dev)==61 and len(finals)==16
    normal=jax.pmap(lambda p,c:E.rollout(p,c[0],c[1],c[2],128,ended=c[3]))
    def teacher_fit(p,st,ax):
        mv,ab,_,_,_,_,mm,bm=E.prediction(p,st);teacher=E.safe_teacher(st,ax)
        joint=jax.nn.softmax(jnp.where(mm,mv,-1e9))[:,:,None]*jax.nn.softmax(jnp.where(bm,ab,-1e9))[:,None,:]
        return teacher.any((1,2)),jnp.where(teacher,joint,0).sum((1,2))
    fit=jax.pmap(teacher_fit)
    for split,cps in [('dev',dev),('holdout',finals)]:
        chosen=[x for x in items if x['record']['split']==split];repeats=8;selected=[x for x in chosen for _ in range(repeats)];actual=len(selected)
        st=merged(selected,'states');ax=merged(selected,'aux');keys=np.array([np.asarray(jax.random.fold_in(jax.random.PRNGKey(2026102317),i)) for i in range(actual)])
        pending=[p for p in cps if not (p/f'native_{split}.json').exists()]
        for start in range(0,len(pending),8):
            authorize();group=pending[start:start+8];padded=group+[group[0]]*(8-len(group));pars=[]
            for cp in padded:
                marker=read(cp/'COMMITTED.json');assert sha(cp/'state.pkl')==marker['state_sha256'];pars.append(pickle.load((cp/'state.pkl').open('rb'))['params'])
            params=jax.tree.map(lambda *a:jnp.stack(a),*pars)
            rep=lambda tr:jax.tree.map(lambda a:jnp.stack([a]*8),tr)
            eligible,mass=jax.device_get(fit(params,rep(st),rep(ax)))
            carry=(rep(st),rep(ax),rep(keys),jnp.zeros((8,actual),bool));datas=[];began=time.time()
            for ci in range(20):
                carry,data=normal(params,carry);jax.block_until_ready(carry[0].t);datas.append(tuple(np.array(a,copy=True) for a in jax.device_get(data)))
                atomic_json(B/'C_BANK_EVAL_STATUS.json',dict(split=split,models=[str(p.relative_to(B)) for p in group],ticks=(ci+1)*128,seconds=time.time()-began,training_updates_unchanged=True,unix=time.time()))
                if np.asarray(carry[3]).all():break
            for slot,cp in enumerate(group):
                data=tuple(np.concatenate([d[i][slot] for d in datas])[:2528] for i in range(len(datas[0])))
                summary,arrays=summarize_paths(data,chosen,repeats,actual)
                summary.update(checkpoint=str(cp.relative_to(B)),split=split,policy_sha256=sha(cp/'policy.pt'),parameters_sha256=digest(pars[slot]),holdout_used_for_tuning=False)
                ii=np.asarray(eligible[slot],bool)
                summary['initial_safe_teacher_fit']=dict(eligible_source_states=int(ii.reshape(-1,repeats).any(-1).sum()),
                    source_states=len(chosen),accepted_joint_probability_mean=float(np.asarray(mass[slot])[ii].mean()) if ii.any() else None,
                    scope='fixed genuine acquisition snapshots only; all far; no near/mid or full-trajectory fit claim')
                save(cp/f'native_{split}.pkl',arrays);atomic_json(cp/f'native_{split}.json',summary)
            notify(f'C固定Bank评估完成一批：{split} {len(group)}模型，未改参数；累计{min(start+8,len(pending))}/{len(pending)}。')
    for path in files:
        if read(path)['update']==0:
            atomic_json(path.parent/'native_dev.json',dict(shared_no_update_control=True,source='C/no_update/u000/native_dev.json',independent_samples=False,**read(control/'native_dev.json')))
    summary={}
    from scripts.carrier_mechanism_stats import cluster_interval
    for split in ['dev','holdout']:
        parent=read(control/f'native_{split}.json');byarm={}
        for arm in protocol['arms']:
            rows=[read(B/'C'/f'{arm}_s{seed}'/'u016'/f'native_{split}.json') for seed in protocol['paired_seeds']]
            delta=np.array([r['delivery']-parent['delivery'] for r in rows])
            baseline=pickle.load((control/f'native_{split}.pkl').open('rb'))['delivery'].reshape(-1,8).mean(-1)
            paired=np.array([pickle.load((B/'C'/f'{arm}_s{seed}'/'u016'/f'native_{split}.pkl').open('rb'))['delivery'].reshape(-1,8).mean(-1)-baseline for seed in protocol['paired_seeds']])
            rng=np.random.default_rng(20261023);si=rng.integers(0,5,(5000,5));ei=rng.integers(0,len(baseline),(5000,len(baseline)))
            joint=paired[si[:,:,None],ei[:,None,:]].mean((1,2))
            byarm[arm]=dict(delivery=[r['delivery'] for r in rows],direct_custody_delivery=[r['direct_custody_delivery'] for r in rows],delta_to_control=delta.tolist(),paired_training_seed_CI=cluster_interval(delta,protocol['paired_seeds']),paired_seed_and_source_episode_bootstrap_CI=np.quantile(joint,[.025,.975]).tolist(),death=[r['death'] for r in rows],drop=[r['drop'] for r in rows])
        summary[split]=dict(control_delivery=parent['delivery'],arms=byarm)
    curves={arm:{str(seed):{str(u):read(B/'C'/f'{arm}_s{seed}'/f'u{u:03d}'/'native_dev.json') for u in protocol['save_updates']} for seed in protocol['paired_seeds']} for arm in protocol['arms']}
    atomic_json(B/'C_NATIVE_BANK_COMPLETE.json',dict(passed=True,dev_curves=curves,dev_unique_checkpoints=61,holdout_final_models=16,summary=summary,training_seed_CI_separate_from_source_episode_CI=True,holdout_tuning=False,unix=time.time()))
    notify('C native Bank完整曲线与固定最终holdout已完成；诊断GPU预算终态，释放GPU，旧O5与储备队列继续暂停。')
