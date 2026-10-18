"""Read completed frozen evaluations; never write to a training source."""
import hashlib
import json
import math
import sys
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select(source):
    source=Path(source);rows=[];protocol=None
    for run in sorted(source.glob('*v38*')):
        p=run/'authority_je10/PROTOCOL.json'
        if not p.exists():continue
        h=sha(p)
        if protocol is None:protocol=h
        if h!=protocol:raise ValueError('different frozen protocol')
        for p in sorted(run.glob('authority_je10/g*/COMPLETE.json')):
            c=json.loads(p.read_text());s=json.loads((p.parent/'SUMMARY.json').read_text())
            if not c['passed']:continue
            for name,h in c['shards'].items():
                if sha(p.parent/name)!=h:raise ValueError('evaluation hash mismatch')
            cp=run/f"formal/bun392_bridge_pfsp8/gen{s['generation']:06d}/policy.pt"
            if sha(cp)!=c['checkpoint_sha256'] or s['policy_sha256']!=c['checkpoint_sha256']:raise ValueError('checkpoint identity')
            route=s['route'];hunter=s['hunter_js'];tactical=s['tactical_python']
            values=[route['natural_full2400']['delivery_episodes']/16,sum(v['delivery_episodes'] for k,v in route.items() if k!='natural_full2400')/112,tactical['safe_win']/64,sum(v['safe_win'] for v in hunter.values())/192]
            unsafe=tactical.get('self_episodes',0)+sum(v.get('self_episodes',0) for v in hunter.values())+sum(v.get('self_death',0) for v in route.values())
            rows.append(dict(round=run.name,generation=s['generation'],checkpoint_sha256=sha(cp),complete_sha256=sha(p),summary_sha256=sha(p.parent/'SUMMARY.json'),metrics=values,self_episodes=unsafe,balanced_score=math.prod(values)**.25,route={k:{q:v.get(q) for q in ['games','pickup_episodes','delivery_episodes','death','self_death']} for k,v in route.items()},tactical=tactical,hunter=hunter))
    rows.sort(key=lambda r:(r['self_episodes']!=0,-r['balanced_score'],r['round'],r['generation']))
    return dict(schema='qqt.je10-release-selection/v1',protocol_sha256=protocol,rule='zero observed self harm first; maximize geometric mean of natural delivery/16, seven route delivery/112, Tactical safe kills/64, pooled Hunter safe kills/192; ties lexical round then generation. Exploratory selection, not significance or admission.',candidates=rows,selected={k:rows[0][k] for k in ['round','generation','checkpoint_sha256']})
if __name__=='__main__':
    result=select(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['selected'], 'completed candidates',len(result['candidates']))
