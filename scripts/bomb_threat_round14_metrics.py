"""Independent raw-outcome recount, without altering frozen training ABI."""
import json
import time
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round14 import BASE,OUT,read


def recount():
    from scripts.bomb_threat_eval import summarize
    folder=BASE/'eval/bt_slot05_c000026_u000640_holdout'
    gate=read(folder/'GATE_METRICS.json');metrics={};differences={};files={}
    controls={}
    for mode in ('greedy','categorical'):
        rows=[];primary=[]
        for start in range(0,64,8):
            p=folder/f'natural_{mode}_{start:03d}.json'
            side=p.with_name(p.name+'.outcome.json')
            rows+=read(side)['episodes'];primary+=read(p)['results']['natural_full2400']['episodes']
            files[p.name]=sha(p);files[side.name]=sha(side)
        assert len(rows)==len(primary)==64 and len({r['seed'] for r in rows})==64
        measured=summarize(rows);expected=gate['natural_js_full2400'][mode]
        differences[mode]={k:dict(raw=v,formal=expected.get(k)) for k,v in measured.items()
            if expected.get(k)!=v}
        metrics[mode]=measured
        controls[mode]=dict(attributed_first_control=sum(r['attributed_first_control'] for r in primary),
            all_first_control=sum(r['first_control'] for r in primary),
            causal_kills=sum(r['causal_kill'] for r in primary),
            actual_bombs=measured['actual_bombs'])
        controls[mode]['attributed_control_events_per_bomb']=controls[mode]['attributed_first_control']/measured['actual_bombs']
        controls[mode]['causal_kill_events_per_bomb']=measured['causal_kills']/measured['actual_bombs']
    assert not any(differences.values()),differences
    result=dict(status='PASS',raw_recomputed_modes=metrics,formal_differences=differences,
        control_conversion=controls,files=files,missing=['unique bomb ID to victim hit linkage'],
        limitations=['first control events per bomb are not unique-bomb hit probability',
                     'observational ratios do not estimate incremental causal benefit of placing a bomb'],unix=time.time())
    atomic_json(OUT/'RAW_RECOUNT.json',result)
    return result


if __name__=='__main__':print(json.dumps(recount()['formal_differences']))
