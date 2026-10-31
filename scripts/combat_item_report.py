"""Fixed final, matched-seed synthesis; no ranking by peaks or shaped reward."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
import json,pickle,time,subprocess
from pathlib import Path
import numpy as np
from scripts.combat_item_run import B,R,read,notify
from scripts.population_run_state import atomic_json,sha
from scripts.carrier_mechanism_report import natural_observation
from scripts.carrier_mechanism_stats import cluster_interval
from scripts.terminal_checkpoint_restore import digest


def paired_uncertainty(delta):
 d=np.asarray(delta,float);rng=np.random.default_rng(20261023);ns,ne=d.shape
 si=rng.integers(0,ns,(5000,ns));ei=rng.integers(0,ne,(5000,ne));boot=d[si[:,:,None],ei[:,None,:]].mean((1,2))
 return dict(per_seed=d.mean(1).tolist(),mean=float(d.mean()),seed_only_CI=cluster_interval(d.mean(1),np.arange(ns)),seed_and_source_episode_CI=np.quantile(boot,[.025,.975]).tolist(),training_seeds=ns,source_episodes=ne,warning='three training seeds; intervals descriptive, not evidence of safety equivalence')


def natural(cp,holdout=False):
 document=read(cp/'retention'/('natural_holdout64.json' if holdout else 'natural_full64.json'))
 summary,binary=natural_observation(document);episodes=document['results']['natural_full2400']['episodes']
 counts={k:sum(e.get(k,0) for e in episodes) for k in ['first_control','attributed_control','final_pop','surviving_kill','window_events','target_events','first_oracle_actions','first_oracle_matches','progress_ticks','target_ticks','window_stock_pickup','stock_pickup_dead','stock_pickup_native_sugar','stock_pickup_free_alive','window_stock_delivery','free_stock_delivery','nonstock_pickup','nonstock_delivery','pickup','delivery','death','self_death']}
 ratio=lambda a,b:counts[a]/counts[b] if counts[b] else None
 summary['funnel_counts']=counts
 summary['funnel_ratios']=dict(window_stock_per_window_event=ratio('window_stock_pickup','window_events'),direct_window_stock_delivery_per_window_stock=ratio('window_stock_delivery','window_stock_pickup'),first_direction_match_per_first_action=ratio('first_oracle_matches','first_oracle_actions'),progress_per_target_tick=ratio('progress_ticks','target_ticks'))
 items={}
 for e in episodes:
  for stage,values in e.get('item_metrics',{}).items():
   row=items.setdefault(stage,{})
   for key,value in values.items():row[key]=row.get(key,0)+value
 summary['items']=items;summary['item_stage_scope']='state phase at each event; cross-phase release->hit is not a per-placement conversion cohort. Net distance/delay are physical diagnostics, not capability or action Q.'
 summary['fixed_first_deviations']=[dict(game=e['game'],seed=e['seed'],team=e['team'],by_phase=e['first_deviations_by_phase']) for e in episodes]
 return summary,binary


def atomic_acceptance():
 p=read(B/'PROTOCOL.json');source=pickle.load((B/'inputs/parent.pkl').open('rb'));parent_adam=digest(source['adam']);parent_params=digest(source['params']);parent_count=int(source['adam'][1][0].count)
 cp=B/'formal/no_update/u000';m=read(cp/'COMMITTED.json');assert sha(cp/'state.pkl')==m['state_sha256'] and sha(cp/'policy.pt')==m['policy_sha256']
 frozen=pickle.load((cp/'state.pkl').open('rb'));assert digest(frozen['params'])==parent_params and digest(frozen['adam'])==parent_adam
 perseed={};rows=[]
 for seed in p['paired_seeds']:
  for arm in p['arms']:
   base=B/'formal'/f'{arm}_s{seed}';updates=[json.loads(x) for x in (base/'updates.jsonl').read_text().splitlines()]
   assert [x['update'] for x in updates]==list(range(1,33))
   assert all(x['finite'] and all(v>0 for v in x['update_l2'].values()) for x in updates)
   assert read(base/'COMPLETE.json')['update']==32
   for u in p['checkpoints']:
    cp=base/f'u{u:03d}';receipt=read(cp/'COMMITTED.json');assert receipt['state_sha256']==sha(cp/'state.pkl') and receipt['policy_sha256']==sha(cp/'policy.pt')
    state=pickle.load((cp/'state.pkl').open('rb'));actor=pickle.load((cp/'policy.pt').open('rb'));assert digest(actor)==digest(state['params']['actor'])
    assert int(state['adam'][1][0].count)==parent_count+u*32
    assert all(np.isfinite(x).all() for x in __import__('jax').tree.leaves(state['params']))
    if u==0:
     assert digest(state['params'])==parent_params and digest(state['adam'])==parent_adam
     aux=dict(state['aux']);aux['tactics']={k:v for k,v in aux['tactics'].items() if not k.endswith('_frontier')}
     identity=digest((state['states'],aux,state['keys']))
     if seed in perseed:assert perseed[seed]==identity
     else:perseed[seed]=identity
    evidence={}
    if u==0:evidence=dict(alias='formal/no_update/u000',independent_model=False)
    else:
     assert (cp/'native_dev.json').exists();evidence['native_dev']=sha(cp/'native_dev.json')
     if u==16:assert (cp/'retention/natural_full64.json').exists();evidence['natural_dev64']=sha(cp/'retention/natural_full64.json')
     if u==32:
      for label in ['native_holdout','bankP_backward_dev','bankP_backward_holdout','bankP_wait_dev','bankP_wait_holdout']:assert (cp/f'{label}.json').exists();evidence[label]=sha(cp/f'{label}.json')
      assert (cp/'retention/COMPLETE.json').exists() and (cp/'retention/natural_holdout64.json').exists()
      evidence['CPU_JS']=sha(cp/'retention/COMPLETE.json');evidence['natural_holdout64']=sha(cp/'retention/natural_holdout64.json')
    atomic_json(cp/'EVALUATION_CONSUMED.json',dict(passed=True,policy_sha256=receipt['policy_sha256'],update=u,coverage=evidence,no_gpu_launch=True))
    rows.append(dict(checkpoint=str(cp.relative_to(B)),state_sha256=receipt['state_sha256'],policy_sha256=receipt['policy_sha256'],update=u))
 atomic_json(B/'ATOMIC_ACCEPTANCE.json',dict(passed=True,run_checkpoints=72,completed_runs=12,paired_initial_env_aux_rng_exact=True,passive_frontier_schema_normalized=True,parent_params_digest=parent_params,parent_Adam_digest=parent_adam,all_Adam_counts_exact=True,all_saved_actor_exports_exact=True,rows=rows))


def fixed_replay_links():
 rows=[]
 cps=[B/'formal/no_update/u000']+[x.parent for x in sorted((B/'formal').glob('*_s*/u032/COMMITTED.json'))]
 for cp in cps:
  for split,folder in [('dev',cp/'retention'),('holdout',cp/'holdout/retention')]:
   paths=sorted(folder.glob('fixed-*.trace.json'));assert len(paths)==1
   trace=read(paths[0]);first=trace['first_deviation']
   row=dict(checkpoint=str(cp.relative_to(B)),split=split,seed=trace['seed'],selection='predeclared first episode, never selected by failure',first_deviation=first)
   if first:
    ticks=trace['diagnostic_ticks']
    matches=[i for i,t in enumerate(ticks) if t['state_before_sha256']==first['state_hash']]
    assert len(matches)==1
    i=matches[0];t=ticks[i]
    # The natural evaluator alternates seats by game index; this fixed episode is game0/P0.
    team=0
    action=first['action'];flat=[action[0],2 if action[2] else 1 if action[1] else 0]
    assert t['action'][2*team:2*team+2]==flat
    row.update(decision_tick=first['tick'],post_transition_tick=t['tick'],matched_by='pre-action state hash, not post-transition tick',
      approximate_reward_components=t['approximate_reward_components'],approximate_reward_GAE=t['approximate_reward_gae'],
      next_approximate_reward_GAE=ticks[i+1]['approximate_reward_gae'] if i+1<len(ticks) else None,
      approximate_reward_GAE_change=ticks[i+1]['approximate_reward_gae']-t['approximate_reward_gae'] if i+1<len(ticks) else None,
      scope='JS observable reward reconstruction only; not saved training GAE or counterfactual advantage')
   rows.append(row)
 atomic_json(B/'FIXED_DEVIATION_LINKS.json',dict(passed=True,rows=rows,absence_of_fixed_deviation_is_not_success=True))


def final_comparison():
 p=read(B/'PROTOCOL.json');results={};binary={}
 for arm in p['arms']:
  results[arm]=[];binary[arm]=dict(dev=[],holdout=[],Bank=[],BankP_backward=[],BankP_wait=[])
  for seed in p['paired_seeds']:
   base=B/'formal'/f'{arm}_s{seed}';cp=base/'u032';retention=read(cp/'retention/COMPLETE.json')
   dev,bd=natural(cp);hold,bh=natural(cp,True);native=read(cp/'native_holdout.json')
   updates=[json.loads(x) for x in (base/'updates.jsonl').read_text().splitlines()]
   sums={k:sum(x['reward_components'].get(k,0) for x in updates) for k in updates[-1]['reward_components']}
   ev={k:sum(x['events'][k] for x in updates) for k in updates[-1]['events']}
   own_item=sums['item_net']+sums['item_replacement_delta']
   row=dict(seed=seed,dev=dev,holdout=hold,native=native,Hunter=retention['hunter'],Tactical=retention['tactical'],
      BankP={k:read(cp/f'bankP_{k}_holdout.json') for k in ['backward','wait']},
      train=dict(events=ev,components=sums,actual_item_reward_net=own_item,actual_item_share_of_signed_total=own_item/sums['combined_total'] if sums['combined_total'] else None,
        route_rows=sum(x['route_teacher_rows'] for x in updates),combat_rows=sum(x['combat_teacher_rows'] for x in updates),
        actual_opponents=dict(current=0,ring=0,fixed=262144,shares=[0,0,1],identity='JAX rule tier3; not JS Hunter/Python Tactical',scope='selected opponent per environment transition; effective free-action subset smaller'),
        steady_u8_u32_steps_per_second=24*8192/(updates[-1]['elapsed_seconds']-updates[7]['elapsed_seconds']),
        first_update_L2=updates[0]['update_l2'],final_update_L2=updates[-1]['update_l2']))
   row['curves']={str(u):read(base/f'u{u:03d}'/'native_dev.json') for u in [1,8,16,24,32]}
   mid,midbinary=natural(base/'u016');row['midpoint_natural']=mid
   row['natural_tail_delivery_mean']=(mid['episode_counts']['delivery']+dev['episode_counts']['delivery'])/2
   row['native_tail_direct_mean']=float(np.mean([row['curves'][str(u)]['direct_custody_delivery'] for u in [16,24,32]]))
   row['native_worst_saved_direct']=min(x['direct_custody_delivery'] for x in row['curves'].values())
   if arm!='control':
    control=results['control'][len(results[arm])]
    assert dev['episode_ids']==control['dev']['episode_ids']
    assert hold['episode_ids']==control['holdout']['episode_ids']
   row['train']['replacement_nonzero_updates']=sum(x['reward_components']['item_replacement_delta']!=0 for x in updates)
   row['train']['policy_matches_paired_control']=sha(cp/'policy.pt')==sha(B/'formal'/f'control_s{seed}'/'u032/policy.pt')
   results[arm].append(row);binary[arm]['dev'].append(bd);binary[arm]['holdout'].append(bh)
   for key,label in [('Bank','native'),('BankP_backward','bankP_backward'),('BankP_wait','bankP_wait')]:
    a=pickle.load((cp/f'{label}_holdout.pkl').open('rb'))
    binary[arm][key].append({metric:np.asarray(a[metric]).reshape(32,8).mean(-1) for metric in ['delivery','direct_custody_delivery','death','drop']})
 comparisons={}
 for arm in p['arms'][1:]:
  report={}
  for source in ['dev','holdout','Bank','BankP_backward','BankP_wait']:
   keys=binary[arm][source][0].keys();report[source]={}
   for metric in keys:
    d=np.stack([np.asarray(a[metric],float)-np.asarray(c[metric],float) for a,c in zip(binary[arm][source],binary['control'][source])]);report[source][metric]=paired_uncertainty(d)
  for key in ['Tactical']:
   report[key]={metric:dict(paired_count_deltas=[a[key][metric]-c[key][metric] for a,c in zip(results[arm],results['control'])]) for metric in ['safe_win','death_episode','self_episodes']}
  report['Hunter']={tier:{metric:dict(paired_count_deltas=[a['Hunter'][tier][metric]-c['Hunter'][tier][metric] for a,c in zip(results[arm],results['control'])]) for metric in ['safe_win','death_episode','self_episodes']} for tier in ['hard','normal','easy']}
  comparisons[arm]=report
 parent=B/'formal/no_update/u000';parent_dev,_=natural(parent);parent_holdout,_=natural(parent,True)
 result=dict(passed=True,id=p['id'],parent=dict(dev=parent_dev,holdout=parent_holdout,Hunter=read(parent/'retention/COMPLETE.json')['hunter'],Tactical=read(parent/'retention/COMPLETE.json')['tactical']),arms=results,paired_to_same_seed_control=comparisons,all_fixed_final_points=True,no_holdout_tuning=True,no_peak_promotion=True,old_continuous_resume=False)
 atomic_json(B/'FINAL_COMPARISON.json',result)
 return result


def plot(d):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=dict(control='#555555',combat='#1976d2',banana='#d99500',glue='#14946b')
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for arm,rows in d['arms'].items():
  values=np.array([[r['midpoint_natural']['episode_counts']['delivery'],r['dev']['episode_counts']['delivery']] for r in rows])/64
  axes[0,0].plot([16,32],values.mean(0),label=arm,color=colors[arm],marker='o')
  axes[0,0].fill_between([16,32],values.min(0),values.max(0),color=colors[arm],alpha=.10)
  steps=[1,8,16,24,32];v=np.array([[r['curves'][str(u)]['direct_custody_delivery'] for u in steps] for r in rows])
  axes[1,0].plot(steps,v.mean(0),label=arm,color=colors[arm],marker='o')
  axes[1,0].fill_between(steps,v.min(0),v.max(0),color=colors[arm],alpha=.10)
 for i,(arm,rows) in enumerate(d['arms'].items()):
  for ax,key in [(axes[0,1],'delivery'),(axes[1,1],'death')]:
   vals=np.array([r['holdout']['episode_counts'][key] for r in rows])/64
   ax.scatter(i+np.array([-.12,0,.12]),vals,color=colors[arm],alpha=.8)
   ax.plot([i-.2,i+.2],[vals.mean()]*2,color=colors[arm],lw=3)
 for ax,key in [(axes[0,1],'delivery'),(axes[1,1],'death')]:
  ax.axhline(d['parent']['holdout']['episode_counts'][key]/64,color='black',ls='--',lw=1,label='shared no-update parent')
  ax.set_xticks(range(4),list(d['arms']));ax.set_ylabel('episode rate / fixed64')
 axes[0,0].axhline(d['parent']['dev']['episode_counts']['delivery']/64,color='black',ls='--',lw=1,label='shared parent')
 axes[0,0].set(title='Natural JS development: all seed tails',xlabel='fixed update',ylabel='delivery episode rate')
 axes[0,1].set_title('Disjoint natural retention seeds: final delivery')
 axes[1,0].set(title='Conditional genuine Bank dev: direct custody delivery',xlabel='fixed update',ylabel='rate; map806/P0/far only')
 axes[1,1].set_title('Disjoint natural retention seeds: final death')
 for ax in axes.flat:ax.grid(alpha=.2);ax.set_ylim(-.025,1.025)
 axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
 fig.suptitle('Fixed paired screen: 3 training seeds/arm; bands show min-max, not CI')
 fig.tight_layout();fig.savefig(R/'reports/combat_item_experiment_20261023.png',dpi=160);plt.close(fig)


def markdown(d):
 lines=['\n## 全部固定预算终态与比较\n',
 '12个run全部u32，3145728真实训练环境步、12288优化器步，另512步技术smoke；73个正式/共同父原子模型点已核验（12×6+1）。全seed/尾部/最差点均保留，无择峰晋升。本轮无发布；旧continuous/O5及储备暂停保留。\n',
 '| 臂 | dev自然交付（3seed，各64） | 独立自然holdout交付（各64） | holdout条件直接返程率 | Hunter难/中/易安全胜均值（各64） | Tactical安全胜均值/64 |',
 '|---|---|---|---|---|---|']
 for arm,rows in d['arms'].items():
  lines.append(f"| {arm} | {[r['dev']['episode_counts']['delivery'] for r in rows]} | {[r['holdout']['episode_counts']['delivery'] for r in rows]} | {[round(r['native']['direct_custody_delivery'],4) for r in rows]} | {[round(np.mean([r['Hunter'][t]['safe_win'] for r in rows]),2) for t in ['hard','normal','easy']]} | {np.mean([r['Tactical']['safe_win'] for r in rows]):.2f} |")
 lines+=['\n父点为共享的一份模型，不伪造三个独立父点；dev/独立自然holdout交付分别 '+str(d['parent']['dev']['episode_counts']['delivery'])+'/64、'+str(d['parent']['holdout']['episode_counts']['delivery'])+'/64。三个处理效应仅相对同seed训练A0，原父点变化还包含共同训练配方。',
 '\n| 处理 vs 同seed A0 | 独立自然holdout交付均差（pp） | 训练seed×对局95% CI（pp） | Bank直接交付均差（pp） | Bank训练seed×来源局95% CI（pp） |',
 '|---|---:|---|---:|---|']
 for arm,r in d['paired_to_same_seed_control'].items():
  a=r['holdout']['delivery'];b=r['Bank']['direct_custody_delivery']
  lines.append(f"| {arm} | {100*a['mean']:.2f} | {[round(100*x,2) for x in a['seed_and_source_episode_CI']]} | {100*b['mean']:.2f} | {[round(100*x,2) for x in b['seed_and_source_episode_CI']]} |")
 lines+=['\n仅3训练seed，区间是有条件的筛选不确定性，不证明安全等效。既有Bank仅map806/P0/far且前轮已使用；新自然holdout与本轮train/dev seed分离，但仍是同一地图，不能据此宣称跨图泛化。Bank-P倒退/等待为固定1步干预，完整保留即时死亡/丢包，不按幸存筛样。',
 '\n| 臂 | dev自然u16/u32尾部交付均值（3seed） | 最差最终holdout交付/64 | 最差保存点Bank直接返程率 | 实际道具净奖励（3seed） |',
 '|---|---|---:|---:|---|']
 for arm,rows in d['arms'].items():lines.append(f"| {arm} | {[r['natural_tail_delivery_mean'] for r in rows]} | {min(r['holdout']['episode_counts']['delivery'] for r in rows)} | {min(r['native_worst_saved_direct'] for r in rows):.4f} | {[round(r['train']['actual_item_reward_net'],4) for r in rows]} |")
 lines+=['\n## 机会归一化与道具使用\n',
 '| 臂 | holdout归因首控/原生窗口次数 | 库存窗口取包/窗口次数 | 同次窗口取包最终交付/窗口取包 | 首动作一致/资格次数 | 持续推进/目标ticks | 自然死亡/192 |',
 '|---|---|---|---|---|---|---|']
 for arm,rows in d['arms'].items():
  c={k:sum(r['holdout']['funnel_counts'][k] for r in rows) for k in rows[0]['holdout']['funnel_counts']}
  lines.append(f"| {arm} | {c['attributed_control']}/{c['window_events']} | {c['window_stock_pickup']}/{c['window_events']} | {c['window_stock_delivery']}/{c['window_stock_pickup']} | {c['first_oracle_matches']}/{c['first_oracle_actions']} | {c['progress_ticks']}/{c['target_ticks']} | {sum(r['holdout']['episode_counts']['death'] for r in rows)}/192 |")
 lines+=['\n首控与窗口是两个事件计数，不是同cohort概率；窗口取包/窗口次数也只作机会强度，可能包含单窗口多次取包。真正按取包来源跟踪的交付另列，不能把总体tick比率解释成整链成功率。',
 '\n| 臂/持包阶段 | 库存ticks | 释放请求/真实合法释放 | 自吃/归因敌命中 | 完整事件/正净收益事件/伤害或无益事件 | 净返程位移或追击延缓（格，proxy） |',
 '|---|---:|---|---|---|---:|']
 for arm,rows in d['arms'].items():
  for family in ['banana','glue']:
   vals=[r['holdout']['items'].get('carry_'+family,{}) for r in rows]
   total=lambda key:sum(v.get(key,0) for v in vals)
   lines.append(f"| {arm}/{family} | {total('inventory_ticks')} | {total('release_attempts')}/{total('legal_releases')} | {total('self_contacts')}/{total('enemy_hits')} | {total('completed')}/{total('beneficial')}/{total('harm')} | {total('net_home_distance' if family=='banana' else 'pursuit_delay_cells'):.4f} |")
 lines+=['\n该表取事件当时阶段，放置、接触和结算可能跨阶段，不以此强算单道具cohort命中率；自吃不自动代表自己放置且有益。JSON保留combat/window/carry全部阶段，完整净收益为物理proxy，不能等同action Q。',
 '\n| 臂 | 有奖励替换差的outer updates（各seed/32） | 最终policy与同seed A0相同 |',
 '|---|---|---|']
 for arm,rows in d['arms'].items():lines.append(f"| {arm} | {[r['train']['replacement_nonzero_updates'] for r in rows]} | {[r['train']['policy_matches_paired_control'] for r in rows]} |")
 lines+=['\n完整计数、机会归一化、死亡/自伤、道具分阶段、每seed u1/8/16/24/32 native曲线、全部固定首偏航、Bank-P与配对CI保存在 `runs/combat_item_20261023/v1/FINAL_COMPARISON.json`。原生固定第一来源repeat0的回归方向/实际奖励分量/本模型Critic重建128步GAE保存在各最终点 `fixed_dev_trajectory.pkl`；它不是历史optimizer优势、全动作Q，也不把“距离回退”冒充全安全集合偏离。JS固定seed Review及逐状态hash/动作/阶段/目标/势能沿唯一Review协议保存，JS近似奖励GAE不冒充JAX实测优势。',
 '\n道具日志分清legacy item_net与替换delta，两者相加才是实际道具奖励；gross hypothetical payoff不是A0/C实际收益，不能用来推断能力。各阶段原始机会/库存/释放/接触/命中/完整结算/净位移/追击延缓/伤害/交付均另列；事件发生阶段可能与放置阶段不同，不把这两种分母强行相除。',
 '\n实际对手current/ring/fixed=0/0/100%，所有被选对手为JAX native rule tier3，战斗老师tier4，不是生产JS Hunter或Python Tactical；不能宣称动态联赛已暴露。冻结配置与实际行数/稳态吞吐均在JSON，编译期显存不算有效训练。Actor/Critic独立，辅助战斗梯度不直接作用Critic；不作共享主干干扰结论。']
 path=R/'reports/combat_item_experiment_20261023.md'
 lines.append('\n![Fixed paired results](combat_item_experiment_20261023.png)')
 with path.open('a') as f:f.write('\n'.join(lines)+'\n')
 with (R/'docs/training_experiment_ledger.md').open('a') as f:f.write('\nCOMBAT-ITEM-20261023-v1固定12run、CPU/JS与Bank/Bank-P均完成；原子验收及完整配对结果见当天报告和FINAL_COMPARISON。诊断GPU停止，不恢复旧队列；无发布/晋升。\n')


if __name__=='__main__':
 assert read(B/'TRAINING_COMPLETE.json')['passed'] and read(B/'NATIVE_BANK_COMPLETE.json')['passed'] and read(B/'CPU_EVALUATION_COMPLETE.json')['passed']
 atomic_acceptance();fixed_replay_links();d=final_comparison();plot(d);markdown(d)
 atomic_json(B/'ANALYSIS_COMPLETE.json',dict(passed=True,comparison_sha256=sha(B/'FINAL_COMPARISON.json'),report='reports/combat_item_experiment_20261023.md',old_continuous_resume=False))
 notify('战斗/真实取包/道具实验全部固定训练与评估完成；完整多seed/保留集/安全结果已保存，进入最终资源释放核验，不自动续训。')
