"""Recomputable paired native/JS cohorts; missing results are never zero."""
from pathlib import Path
import json,statistics
import numpy as np
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/combat_mix_v1'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None

def ci(values,seed=20261028):
 if not values:return None
 x=np.asarray(values,float);rng=np.random.default_rng(seed)
 means=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
 return dict(mean=float(x.mean()),ci95_percentile=np.quantile(means,[.025,.975]).tolist(),training_seeds=len(x),
  min=float(x.min()),max=float(x.max()),limitation='Only3 paired training seeds; percentile bootstrap descriptive and low resolution.')

def native_summary():
 results={};inputs={}
 for done in sorted((BASE/'eval/native').glob('*/COMPLETE.json')):
  d=read(done)
  path=done.parent/'SUMMARY.json'
  if not d or not d.get('passed'):continue
  assert sha(path)==d['summary_sha256']
  s=read(path);inputs[str(path.relative_to(ROOT))]=sha(path);modes={}
  for row in s['rows']:
   c=row['passive_cohort'];modes[row['action_mode']]=dict(games=row['games'],
    delivery=c['at_least_one_delivery_rate'],own_deaths=c['own_deaths_per_match'],
    death_window=c['windows']['death'],sugar_window=c['windows']['native_syrup'],
    carry=c['carry_episodes'],phase=c['phases'],termination=c['termination_counts'],
    legacy_undiscounted=row['legacy']['mean_undiscounted'],
    legacy_discounted=row['legacy']['mean_discounted_gamma_0_995'])
  results[done.parent.name]=modes
 return results,inputs

def main():
 from scripts.combat_mix_exposure_summary import main as exposure_summary
 exposure_summary()
 p=read(BASE/'PROTOCOL.json');native,hashes=native_summary();paired={}
 for arm in p['arms']:
  if arm=='control':continue
  for mode in ('categorical','greedy'):
   for split in ('dev','holdout'):
    rows=[]
    for seed in p['seeds']:
     candidate=native.get(f'{arm}_s{seed}_u000320_{split}',{}).get(mode)
     control=native.get(f'control_s{seed}_u000320_{split}',{}).get(mode)
     parent=native.get(f'parent_s{seed}_u000160_{split}',{}).get(mode)
     if not all((candidate,control,parent)):continue
     rows.append(dict(seed=seed,games=candidate['games'],
      delivery_delta_vs_control=candidate['delivery']['value']-control['delivery']['value'],
      delivery_delta_vs_parent=candidate['delivery']['value']-parent['delivery']['value'],
      deaths_delta_vs_control=candidate['own_deaths']['value']-control['own_deaths']['value'],
      carry_delta_vs_control=(candidate['carry']['same_episode_delivery_rate']['value']-control['carry']['same_episode_delivery_rate']['value'])
       if candidate['carry']['same_episode_delivery_rate']['value'] is not None and control['carry']['same_episode_delivery_rate']['value'] is not None else None))
    paired[f'{arm}:{mode}:{split}']=dict(complete_paired_training_seeds=len(rows),rows=rows,
     delivery_vs_control=ci([r['delivery_delta_vs_control'] for r in rows]),
     delivery_vs_parent=ci([r['delivery_delta_vs_parent'] for r in rows]),
     deaths_vs_control=ci([r['deaths_delta_vs_control'] for r in rows]),
     decision='INCOMPLETE' if len(rows)!=3 else 'FIXED_ENDPOINT_COMPARISON_ONLY_NO_PROMOTION')
 js={}
 for path in (BASE/'eval/js').glob('*/SUMMARY.json'):
  marker=read(path.parent/'COMPLETE.json')
  if not marker:continue
  assert sha(path)==marker['summary_sha256']
  v=read(path);hashes[str(path.relative_to(ROOT))]=sha(path)
  js[path.parent.name]=dict(checkpoint=v['checkpoint'],policy_sha256=v['policy_sha256'],
    hunter=v['hunter_js'],natural={mode:{k:r.get(k) for k in ['games','window_events','window_stock_pickup',
      'window_stock_delivery','delivery','death','stock_pickup_free_alive','stock_pickup_dead','stock_pickup_native_sugar']}
      for mode,r in v['natural_js'].items()})
 result=dict(native=native,paired_training_seed_comparisons=paired,production_JS=js,input_sha256=hashes,
  missing_data='Absent cohorts remain absent/null; no failure or incomplete game imputed as zero.',
  metric_scope='Native rule3 and production JS Hunter separate. No claim argmax/categorical equivalent. Final three seeds, parent+matched control required.',
  teacher=False,no_training_control=True)
 atomic_json(BASE/'PAIRED_RESULTS.json',result)
 report=['# COMBAT-MIX 固定终点配对结果（20261028）','',
  '结果按冻结u320、三seed配对，不择峰。JAX与JS、categorical与greedy、dev与holdout分别输出；缺数据不按0。完整分子分母与输入SHA见 runs/combat_mix_v1/PAIRED_RESULTS.json。','',
  '|比较|已齐配对seed|交付差 vs control|交付差 vs 父|','|---|---|---|---|']
 for name,r in paired.items():
  a=r['delivery_vs_control'];b=r['delivery_vs_parent']
  report.append(f"|{name}|{r['complete_paired_training_seeds']}/3|{a['mean'] if a else '缺数据'}|{b['mean'] if b else '缺数据'}|")
 report+=['','只有3训练seed，训练seed bootstrap区间证据弱；目前不宣称总体最佳或发布。局级配对原始episodes保留，可重算；未完成条件明确标记。','',
  '连泡奖励/候选数量从训练exposure原始字段独立列出，不把塑形收益等同能力。legacy_mix为旧设置包复原，不作单因素归因。']
 (ROOT/'reports/combat_mix_paired_20261028.md').write_text('\n'.join(report)+'\n')
if __name__=='__main__':main()
