"""Independent per-game trace recomputation of diagnostic metrics."""
import json,math,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/jitter_followup_u800')
def read(p):return json.loads(p.read_text())
def point(value):
    # JSON.stringify(TypedArray) stores numeric keys rather than a JSON array.
    return [value[str(i)] for i in range(2)] if isinstance(value,dict) else value
def main():
    cfg=read(B/'PROTOCOL.json');checks=[];h0comparison=[]
    for arm in cfg['inference_arms']:
        for start in range(0,64,8):
            stem=B/arm/f'shard_{start:03d}.json';metrics=read(Path(str(stem)+'.diagnostic.json'));trace=read(Path(str(stem)+'.trace.json'));out=read(Path(str(stem)+'.outcome.json'))
            assert trace['model_sha256']==metrics['model_sha256']==cfg['model']['model_sha256'];assert out['mode']=='greedy' and not out['teacher']
            assert len(trace['episodes'])==len(metrics['episodes'])==len(out['episodes'])==8
            for e in trace['episodes']:
                rr=e['rows'];m=next(m for m in metrics['episodes'] if m['seed']==e['seed']);o=next(m for m in out['episodes'] if m['seed']==e['seed']);assert m['seat']==e['seat']==o['seat'];assert len(rr)==m['ticks']==o['ticks'] and m['done']
                events=[i for i in range(1,len(rr)) if rr[i]['controlled'] and rr[i-1]['controlled'] and (rr[i-1]['move'],rr[i]['move']) in [(2,3),(3,2)]]
                assert len(events)==m['lr_reversals'] and [z-a for a,z in zip(events,events[1:])]==m['reversal_intervals']
                assert sum(r['controlled'] and r['move']<4 for r in rr)==m['move_ticks'];stall=0;osc=0
                for i in range(0,len(rr)-49,50):
                    rows=rr[i:i+50];p=[point(r['pos']) for r in rows]+[point(rows[-1]['after'])];path=sum(math.dist(a,z) for a,z in zip(p,p[1:]));net=math.dist(p[0],p[-1]);eligible=all(r['controlled'] and r['next_controlled'] for r in rows)
                    if eligible and path>1e-6 and net/path<.1:osc+=50
                    initial=rows[0]['home_static'];bp=initial>=0 and any(0<=r['after_home_static']<initial for r in rows)
                    if eligible and all(r['carry'] and r['next_carry'] for r in rows) and rows[0]['home_center_distance']-rows[-1]['after_home_center_distance']<.1 and not bp:stall+=50
                assert (stall,osc)==(m['carry_stall_ticks'],m['oscillation_ticks'])
                for k in ['placed','deaths','self_deaths','causal_kills','winner','done','stored']:assert m[k]==o[k]
                checks.append(dict(arm=arm,seed=e['seed'],seat=e['seat'],ticks=len(rr),passed=True))
                if arm=='H0':
                    prior=Path('runs/bomb_threat_v1/eval/bt_slot01_c000015_u000800')/f'natural_greedy_{start:03d}.json.outcome.json'
                    if prior.exists():
                        d=next(x for x in read(prior)['episodes'] if x['seed']==e['seed'])
                        fields=['ticks','winner','placed','causal_kills','deaths','done','stored'];assert all(d[k]==m[k] for k in fields)
                        h0comparison.append(dict(seed=e['seed'],source=str(prior),source_sha256=sha(prior),matched_fields=fields))
    assert len(checks)==192 and len({(c['arm'],c['seed']) for c in checks})==192
    atomic_json(B/'RECOMPUTED.json',dict(status='PASS',games=192,ticks=sum(r['ticks'] for r in checks),checks=checks,H0_prior_production_outcome_comparison=h0comparison,unix=time.time()))
    print(json.dumps(dict(status='PASS',games=192,H0_prior_matched=len(h0comparison))))
if __name__=='__main__':main()
