"""Aggregate completed production-JS expert-validation rows without rerunning."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha

R=Path(__file__).resolve().parents[1]
B=R/'runs/expert_validation_20261025/B_js/verified_v2'
MODELS=['parent','candidate','online'];MODES=['categorical','greedy']
def merge_numeric(dst,src):
    for k,v in src.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):dst[k]=dst.get(k,0)+v
        elif isinstance(v,dict):merge_numeric(dst.setdefault(k,{}),v)
def wilson(success,n,z=1.959963984540054):
    if n<=0:return [None,None]
    p=success/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]
def paired_delivery_comparison(left,right,seed=20261025):
    """Paired game-level difference with mutually exclusive outcome counts."""
    seeds=sorted(set(left)&set(right))
    if not seeds: raise ValueError('no shared paired seeds')
    l=np.asarray([int(left[s]['delivery']>0) for s in seeds],dtype=float)
    r=np.asarray([int(right[s]['delivery']>0) for s in seeds],dtype=float)
    diffs=l-r
    rng=np.random.default_rng(seed)
    boot=np.mean(diffs[rng.integers(0,len(diffs),(20000,len(diffs)))],axis=1)
    return {'paired_games':len(seeds),'left_only_deliveries':int(np.sum((l==1)&(r==0))),
      'right_only_deliveries':int(np.sum((l==0)&(r==1))),
      'both_delivered':int(np.sum((l==1)&(r==1))),
      'neither_delivered':int(np.sum((l==0)&(r==0))),
      'mean_paired_delivery_difference':float(diffs.mean()),
      'paired_bootstrap95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
      'training_seed_scope':'one checkpoint seed per non-online model; game-seed bootstrap does not estimate training-seed variance'}
def main():
    rows=[];episodes={}
    for model in MODELS:
      for mode in MODES:
        p=B/f'{model}_{mode}'/'RESULT.json'
        d=json.loads(p.read_text());r=d['results']['natural_full2400'];eps=r['episodes'];episodes[(model,mode)]={e['seed']:e for e in eps}
        items={};exposure={};funnel={}
        for ep in eps:
          merge_numeric(funnel,ep['stage_funnel'])
          merge_numeric(exposure,ep.get('action_exposure',{}))
          merge_numeric(items,ep.get('item_metrics',{}))
        deliveries=sum(int(e['delivery']>0) for e in eps)
        deaths=sum(int(e['death']>0) for e in eps)
        row={'model':model,'mode':mode,'games':len(eps),'seed_start':min(e['seed'] for e in eps),'seed_end':max(e['seed'] for e in eps),
          'delivery_episodes':deliveries,'delivery_rate':deliveries/len(eps),'delivery_rate_wilson95':wilson(deliveries,len(eps)),
          'death_episodes':deaths,'deaths':sum(e['death'] for e in eps),'pickup_episodes':sum(int(e['pickup']>0) for e in eps),
          'acquisition_funnel':{k:sum(e.get(k,0) for e in eps) for k in (
            'return_started','window_stock_pickup','stock_pickup_dead','stock_pickup_native_sugar',
            'stock_pickup_free_alive','stock_pickup_expired_window','stock_pickup_free_no_history',
            'window_stock_delivery','free_stock_delivery','nonstock_pickup','nonstock_delivery',
            'delivery_without_tracked_acquisition')},
          'window_events':sum(e['window_events'] for e in eps),'first_control':sum(e.get('first_control',0) for e in eps),
          'attributed_control':sum(e.get('attributed_control',0) for e in eps),'native_causal_kill':sum(e.get('native_causal_kill',0) for e in eps),
          'oracle_opportunities':sum(e['oracle_opportunities'] for e in eps),'oracle_matches':sum(e['oracle_matches'] for e in eps),
          'first_oracle_actions':sum(e['first_oracle_actions'] for e in eps),'first_oracle_matches':sum(e['first_oracle_matches'] for e in eps),
          'stage_funnel':funnel,'action_exposure':exposure,'item_metrics':items,
          'mean_entropy_by_phase':{k:{'move':v['move_entropy_sum']/max(1,v['ticks']),'ability':v['ability_entropy_sum']/max(1,v['ticks']),'ticks':v['ticks']} for k,v in exposure.items()},
          # The production evaluator seals the exact model JSON it loaded.
          # Some row documents predate the optional source_model_sha256 alias;
          # do not make summary generation depend on that redundant field.
          'model_json_sha256':d['model_json_sha256']}
        rows.append(row)
    paired={}
    for mode in MODES:
      for left,right in [('candidate','online'),('parent','online'),('candidate','parent')]:
        paired[f'{left}_minus_{right}_{mode}']=paired_delivery_comparison(
          episodes[(left,mode)],episodes[(right,mode)])
    out={'schema':'qqt.expert-validation-js-summary/v1','games_per_row':50,'rows':rows,'paired_delivery_comparisons':paired,
      'inference':'native production JS TransformerModel/ONNX WASM-CPU; teacher off; categorical and argmax rows use paired seeds with independent actor/opponent RNG streams',
      'limitation':'One checkpoint per model and only 50 paired games per row; intervals describe game-seed uncertainty, not training-seed uncertainty.',
      'source_sha256':{'scripts/expert_validation_js_eval.js':sha(R/'scripts/expert_validation_js_eval.js'),'scripts/dagger_rl_route.js':sha(R/'scripts/dagger_rl_route.js'),'scripts/ppo_scale_item_metrics_v4.js':sha(R/'scripts/ppo_scale_item_metrics_v4.js')}}
    atomic_json(B/'SUMMARY.json',out);print(json.dumps({'rows':[(x['model'],x['mode'],x['delivery_episodes'],x['death_episodes']) for x in rows],'paired':paired},indent=2))
if __name__=='__main__':main()
