"""Independent raw-source coordinate/action/window recomputation, CPU only."""
import json,math,time
from pathlib import Path
from collections import Counter
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/p0_jitter_u800')
def main():
    cfg=json.loads((B/'PROTOCOL.json').read_text());results=json.loads((B/'RESULTS.json').read_text());checks=[]
    for g in cfg['games']:
        p=B/f"game_{g['index']:03d}.json";d=json.loads(p.read_text());source=Path(cfg['source'])/f"game_{g['index']:03d}"
        tr=json.loads((source/'TRACE.json').read_text());cm=json.loads((source/'COMPLETE.json').read_text());assert sha(source/'TRACE.json')==cm['trace_sha256']
        rr=d['rows'];assert len(rr)==len(tr['rows'])==d['ticks'];rev=[];categories=Counter()
        for t,(r,s) in enumerate(zip(rr,tr['rows'])):
            assert r['tick']==s[0] and r['pos']==s[24] and r['move']==s[11] and r['bomb_legal']==s[15] and r['bomb_probability']==s[16]
            assert r['carry']==bool(s[1] and not s[2] and s[10]==1)
            if t+1<len(rr):assert r['after']==rr[t+1]['pos']
            if t and r['controlled'] and rr[t-1]['controlled'] and (rr[t-1]['move'],r['move']) in [(2,3),(3,2)]:rev.append(t)
        assert rev==d['reversals']['events'];assert [b-a for a,b in zip(rev,rev[1:])]==d['reversals']['intervals']
        for w in d['windows']:
            a,z=w['start'],w['end'];rows=rr[a:z];pts=[r['pos'] for r in rows]+[rows[-1]['after']]
            path=sum(math.dist(x,y) for x,y in zip(pts,pts[1:]));net=math.dist(pts[0],pts[-1]);assert math.isclose(path,w['path'],abs_tol=1e-9);assert math.isclose(net,w['net'],abs_tol=1e-9)
            eligible=all(r['controlled'] and r['next_controlled'] for r in rows);carry=all(r['carry'] and r['next_carry'] for r in rows)
            stall=eligible and carry and rows[0]['home_center_distance']-rows[-1]['after_home_center_distance']<.1 and not (rows[0]['home_static']>=0 and any(0<=r['after_home_static']<rows[0]['home_static'] for r in rows))
            osc=eligible and path>1e-6 and net/path<.1
            legal=[r for r in rows if r['bomb_legal']]
            if stall:cat='持包停滞'
            elif osc:cat='原地振荡'
            elif eligible and path>=1 and net/path>=.1 and legal and sum(r['bomb_probability'] for r in legal)/len(legal)<.1:cat='正常游走，只是概率低'
            else:cat='其他'
            assert w['raw_stall']==stall and w['raw_oscillation']==osc and w['category']==cat;categories[cat]+=50
        summary=next(r for r in results['games'] if r['index']==g['index']);assert all(summary['category_ticks'][k]==categories[k] for k in summary['category_ticks'])
        checks.append(dict(index=g['index'],ticks=len(rr),source_trace_sha256=sha(source/'TRACE.json'),audit_sha256=sha(p),categories=dict(categories)))
    atomic_json(B/'RECOMPUTED.json',dict(status='PASS',independent_source_coordinate_action_and_50tick_check=True,games=len(checks),ticks=sum(r['ticks'] for r in checks),checks=checks,unix=time.time()))
    print(json.dumps(dict(status='PASS',games=len(checks),ticks=sum(r['ticks'] for r in checks))))
if __name__=='__main__':main()
