"""Consume the fixed paired endpoints, parent and switch evidence without peaks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
import pickle,json,time
import numpy as np
import jax
from scripts.dagger_rl_run import B,R,read,notify
from scripts.population_run_state import atomic_json,sha,lock
from scripts.terminal_checkpoint_restore import digest
from scripts.combat_item_report import natural,paired_uncertainty


def run():
 unique=lock(B/'analysis.lock');p=read(B/'PROTOCOL.json')
 while not (B/'PARENT_PASSIVE_BACKFILL.json').exists():
  assert not (B/'PARENT_PASSIVE_BACKFILL_FAILURE.json').exists(),'parent passive backfill failed'
  time.sleep(5)
 assert read(B/'TRAINING_COMPLETE.json')['passed'] and read(B/'CPU_EVALUATION_COMPLETE.json')['passed'] and read(B/'NATIVE_BANK_COMPLETE.json')['passed']
 source=pickle.load((B/'inputs/parent.pkl').open('rb'));parent=B/'formal/no_update/u000'
 par_dev,par_bd=natural(parent);par_hold,par_bh=natural(parent,True)
 parret=read(parent/'retention/COMPLETE.json');results={};binaries={};initial={};accept=[]
 for arm in p['arms']:
  results[arm]=[];binaries[arm]=[]
  for seed in p['paired_seeds']:
   base=B/'formal'/f'{arm}_s{seed}';cp=base/'u032';rows=[json.loads(x) for x in (base/'updates.jsonl').read_text().splitlines()]
   assert [x['update'] for x in rows]==list(range(1,33))
   assert all(x['finite'] and x['update_l2']['actor']>0 for x in rows)
   assert all((x['update_l2']['critic']==0 if x['mode']=='dagger' else x['update_l2']['critic']>0) for x in rows)
   for u in p['checkpoints']:
    c=base/f'u{u:03d}';m=read(c/'COMMITTED.json');assert sha(c/'state.pkl')==m['state_sha256'] and sha(c/'policy.pt')==m['policy_sha256']
    x=pickle.load((c/'state.pkl').open('rb'));actor=pickle.load((c/'policy.pt').open('rb'));assert digest(actor)==digest(x['params']['actor'])
    assert int(x['adam'][1][0].count)==int(source['adam'][1][0].count)+u*32
    if u==0:
     assert digest(x['params'])==digest(source['params']) and digest(x['adam'])==digest(source['adam'])
     d=digest((x['states'],x['aux'],x['keys']))
     if seed in initial:assert initial[seed]==d
     else:initial[seed]=d
    if arm=='sequential' and u<=16:
     assert digest(x['params']['critic'])==digest(source['params']['critic'])
     for name in ['mu','nu']:assert digest(getattr(x['adam'][1][0],name)['critic'])==digest(getattr(source['adam'][1][0],name)['critic'])
    accept.append(dict(checkpoint=str(c.relative_to(B)),state_sha256=m['state_sha256'],policy_sha256=m['policy_sha256']))
   dev,bd=natural(cp);hold,bh=natural(cp,True);mid,_=natural(base/'u016');ret=read(cp/'retention/COMPLETE.json')
   binaries[arm].append(bh)
   stage={}
   for mode in ['dagger','joint','ppo']:
    rr=[x for x in rows if x['mode']==mode]
    if not rr:continue
    stage[mode]=dict(updates=len(rr),environment_steps=8192*len(rr),optimizer_steps=32*len(rr),new_item_rows=sum(x['aggregation']['new_item_rows'] for x in rr),replay_item_rows=sum(x['aggregation']['replay_item_rows'] for x in rr),carry_inventory_ticks=sum(x['carry_inventory_ticks'] for x in rr),carry_release=sum(x['carry_release'] for x in rr),events={k:sum(x['events'][k] for x in rr) for k in rr[0]['events']},mean_ability_head_gradient=float(np.mean([x['ability_head_gradient_norm'] for x in rr])))
   curve={str(u):read(base/f'u{u:03d}/native_dev.json') for u in [1,8,16,17,24,32]}
   row=dict(seed=seed,dev=dev,holdout=hold,midpoint=mid,Hunter=ret['hunter'],Tactical=ret['tactical'],Bank=read(cp/'native_holdout.json'),BankP={k:read(cp/f'bankP_{k}_holdout.json') for k in ['backward','wait']},curves=curve,stage=stage,tail_natural_delivery_mean=(mid['episode_counts']['delivery']+dev['episode_counts']['delivery'])/2,worst_saved_Bank_direct=min(x['direct_custody_delivery'] for x in curve.values()),actual_opponents=dict(current=0,ring=0,fixed=262144,identity='JAX native tier3; not JS Hunter/Python Tactical'),switch_points={str(u):read(base/f'u{u:03d}/COMMITTED.json')['metrics'] for u in [16,17]})
   results[arm].append(row)
 comparisons={}
 for arm in ['sequential','joint']:
  comparisons[arm]={}
  for metric in binaries[arm][0]:
   comparisons[arm][metric]=paired_uncertainty(np.stack([np.asarray(a[metric],float)-np.asarray(c[metric],float) for a,c in zip(binaries[arm],binaries['ppo'])]))
 parent_differences={arm:{metric:paired_uncertainty(np.stack([np.asarray(a[metric],float)-np.asarray(par_bh[metric],float) for a in binaries[arm]])) for metric in par_bh} for arm in p['arms']}
 result=dict(passed=True,id=p['id'],parent=dict(dev=par_dev,holdout=par_hold,Hunter=parret['hunter'],Tactical=parret['tactical']),arms=results,paired_vs_ppo=comparisons,vs_shared_parent=parent_differences,no_peak_selection=True,no_new_confirmation_claim=True,bank_limit=p['bank_limits'])
 atomic_json(B/'ATOMIC_ACCEPTANCE.json',dict(passed=True,declared_checkpoints=len(accept),all_saved_Adam_counts_exact=True,actor_exports_exact=True,paired_u0_fullstate_exact=True,sequential_critic_and_moments_frozen=True,rows=accept))
 atomic_json(B/'FINAL_COMPARISON.json',result)
 # Decision evidence ranks complete-chain/safety before item proxy. No automatic promotion.
 delivery_parent=par_hold['episode_counts']['delivery'];summary={}
 for arm,rs in results.items():
  summary[arm]=dict(holdout_delivery=[r['holdout']['episode_counts']['delivery'] for r in rs],holdout_death=[r['holdout']['episode_counts']['death'] for r in rs],tail=[r['tail_natural_delivery_mean'] for r in rs],worst=min(r['holdout']['episode_counts']['delivery'] for r in rs),item_exposure=[{k:v['new_item_rows'] for k,v in r['stage'].items()} for r in rs])
 atomic_json(B/'SUCCESSOR_EVIDENCE.json',dict(frozen_question=p['successor'],summary=summary,parent_delivery=delivery_parent,selection_requires_full_evidence=True,additional_training_not_dispatched=True,old_continuous_resume=False))
 lines=['\n## 固定R1终态比较\n','| 臂 | 自然holdout交付/64（三seed） | 死亡对局/64 | dev尾部均值 | 最差最终交付 |','|---|---|---|---|---|']
 for arm,r in summary.items():lines.append(f"| {arm} | {r['holdout_delivery']} | {r['holdout_death']} | {r['tail']} | {r['worst']} |")
 lines += [f'\n共享未训练父点自然holdout交付{delivery_parent}/64。配对PPO对照、父点差异与来源局×训练seed CI见FINAL_COMPARISON，不能以退化control宣称改进。', '\n三seed为筛选；既有Bank单图P0全far且已使用，不是全新跨图确认。完整阶段有效环境/优化步、道具暴露/聚合消费、切换16/17、全保存点及最差点均保留。生产JS greedy老师关闭，Python Tactical/JAX rule/JS Hunter独立。无择峰/晋升/发布，旧队列保持暂停。']
 with (R/'reports/dagger_rl_iteration_20261024.md').open('a') as f:f.write('\n'.join(lines)+'\n')
 atomic_json(B/'ANALYSIS_COMPLETE.json',dict(passed=True,comparison_sha256=sha(B/'FINAL_COMPARISON.json'),report='reports/dagger_rl_iteration_20261024.md'))
 notify('R1完整固定比较已写当日报告与FINAL_COMPARISON；多seed/未训练父/匹配PPO/尾部与安全均已消费，等待有证据的下一项预注册，不恢复旧长训。')
if __name__=='__main__':run()
