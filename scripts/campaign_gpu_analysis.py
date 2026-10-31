"""Audit actual GPU probe trajectories before recommending a common backend."""
import json
import os
from pathlib import Path
import pickle
import numpy as np
from scripts.population_run_state import atomic_json,sha

os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'runs/gpu_efficiency_20261026/performance_probe_v1'
OUT=ROOT/'runs/authorized_campaign_20261026/GPU_analysis'


def compare_witness(a,b,*,exact=False):
    """Numeric field parity plus exact metadata, without numeric coercion."""
    import jax
    from scripts.gpu_efficiency_parity import compare_arrays
    left,ls=jax.tree.flatten_with_path(a);right,rs=jax.tree.flatten_with_path(b)
    if ls!=rs:return dict(passed=False,reason='tree structure differs')
    rows=[]
    for (path,x),(_,y) in zip(left,right):
        x,y=np.asarray(x),np.asarray(y)
        if x.dtype.kind not in 'biufc' or y.dtype.kind not in 'biufc':
            same=x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y)
            r=dict(passed=bool(same),bitwise_equal=bool(same),exact_required=True,
                   shape_equal=x.shape==y.shape,dtype_equal=x.dtype==y.dtype,max_abs_error=None)
        else:r=compare_arrays(x,y,exact=exact or x.dtype.kind not in 'fc')
        rows.append(dict(field=jax.tree_util.keystr(path),**r))
    return dict(passed=all(x['passed'] for x in rows),leaf_count=len(rows),
                bitwise_equal=all(x['bitwise_equal'] for x in rows),fields=rows)

def main():
    m=json.loads((B/'MANIFEST.json').read_text());records={}
    for label in m['process_order']:
        p=B/label/'COMPLETE.json';r=json.loads(p.read_text())
        assert r['passed'] and r['new_environment_steps']==131072 and r['new_optimizer_steps']==256
        assert r['manifest_sha256']==sha(B/'MANIFEST.json')
        assert r['final_witness_sha256']==sha(B/label/'FINAL_WITNESS.pkl')
        warm=r['rows'][2:]
        records[label]=dict(env_per_second=len(warm)*16384/sum(x['cycle_seconds'] for x in warm),
            whole_cycle_with_witness_env_per_second=len(warm)*16384/sum(x['total_cycle_with_passive_witness_seconds'] for x in warm),
            warm_cycles=[x['cycle_seconds'] for x in warm],process_wall_seconds=r['process_wall_seconds'],
            first_cycle_seconds=r['rows'][0]['cycle_seconds'],rows=r['rows'],result_sha256=sha(p))
    reference=json.loads((B/'baseline_cold/COMPLETE.json').read_text())
    base=pickle.load((B/'baseline_cold/FINAL_WITNESS.pkl').open('rb'));checks=[]
    for label in m['process_order'][1:]:
        other=json.loads((B/label/'COMPLETE.json').read_text())
        digests=[dict(iteration=i+1,**{field:a[field]==b[field] for field in ('snapshot_digest','batch_digest','diag_digest')})
                 for i,(a,b) in enumerate(zip(reference['rows'],other['rows']))]
        final=pickle.load((B/label/'FINAL_WITNESS.pkl').open('rb'))
        fields=compare_witness(base,final)
        exact={name:compare_witness(x,y,exact=True) for name,x,y in (
            ('actions',base['batch'][2],final['batch'][2]),
            ('masks_labels',base['batch'][6:],final['batch'][6:]),
            ('reward',base['diag'][3],final['diag'][3]),
            ('PRNG',base['snapshot']['keys'],final['snapshot']['keys']))}
        all_digest=all(all(v for k,v in row.items() if k!='iteration') for row in digests)
        checks.append(dict(variant=label,per_iteration_digests=digests,final_fields=fields,final_exact=exact,
            optimization_eligible=all_digest and fields['passed'] and all(c['passed'] for c in exact.values()),
            limitation=None if all_digest else 'Intermediate digest mismatch; only final arrays retained. No semantic-preserving adoption without additional field witnesses.'))
        del final
    samples={}
    for label in m['process_order']:
        p=ROOT/'runs/authorized_campaign_20261026/gpu_samples'/f'GPU-P1-{label}.jsonl'
        rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        samples[label]=dict(samples=len(rows),per_device=[dict(index=i,
            mean_util=float(np.mean([x['gpus'][i]['utilization_percent'] for x in rows])) if rows else None,
            mean_power=float(np.mean([x['gpus'][i]['power_watts'] for x in rows if x['gpus'][i]['power_watts'] is not None])) if rows else None)
            for i in range(8)],duty_cycle='sampled activity proxy; not integrated kernel time')
    eligible=[x['variant'] for x in checks if x['optimization_eligible']]
    selected='baseline'
    for variant in ('hoisted_updater','resident_params_adam'):
        labels=[variant+'_cold',variant+'_cache_warm_fresh']
        if all(x in eligible for x in labels):
            gains=[records[labels[0]]['env_per_second']/records['baseline_cold']['env_per_second'],
                records[labels[1]]['env_per_second']/records['baseline_cache_warm_fresh']['env_per_second']]
            if min(gains)>1.02:selected=variant;break
    OUT.mkdir(exist_ok=True,parents=True)
    atomic_json(OUT/'RESULT.json',dict(records=records,parity=checks,gpu_samples=samples,
        selected_common_implementation=selected,applied_to_both_future_arms=True,
        performance_gate='Technical semantics only; no ability gate',no_policy_adoption=True))
    atomic_json(OUT/'COMPLETE.json',dict(passed=True,semantic_eligible=eligible,
        selected_common_implementation=selected,result_sha256=sha(OUT/'RESULT.json')))
    print(json.dumps(dict(selected=selected,semantic_eligible=eligible)))

if __name__=='__main__':main()
