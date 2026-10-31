"""Read-only artifact acceptance for the frozen native recovery inventory.

No process control, request creation, model selection, or training authority.
Run with --watch to validate each completed point once and retain SHA evidence.
"""
import argparse,json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/iter_pbt_v1'

def read(path):return json.loads(path.read_text())

def verify(name,inventory):
    request=read(BASE/'eval_requests'/f'{name}.json')
    out=BASE/'eval'/name;cp=ROOT/request['checkpoint']
    marker=read(cp/'COMMITTED.json')
    assert sha(cp/'policy.pt')==marker['policy_sha256']
    from scripts.checkpoint_policy_artifact import verify_policy
    verify_policy(cp)
    value=read(out/'SUMMARY.json');complete=read(out/'COMPLETE.json')
    assert complete['passed'] and sha(out/'SUMMARY.json')==complete['summary_sha256']
    assert value['policy_sha256']==marker['policy_sha256'] and value['checkpoint']==request['checkpoint']
    assert value['teacher'] is False and value['engine_separation'] is True
    assert value['seed']==(20261101 if request['split']=='dev' else 20261001)
    for f,h in complete['files'].items():assert sha(out/f)==h,(name,f)
    # All already-completed production JS shards must remain byte-identical.
    for f,h in inventory.get('JS_files_sha256',{}).items():assert sha(out/f)==h,(name,'JS replayed or changed',f)
    nc=read(out/'native/NATIVE_COMPLETE.json');nv=read(out/'native/NATIVE_SUMMARY.json')
    assert nc['passed'] and sha(out/'native/NATIVE_SUMMARY.json')==nc['summary_sha256']
    assert nv['policy_sha256']==marker['policy_sha256'] and value['natural_jax']==nv['natural_jax']
    games=64 if request['split']=='dev' else 304
    modes=['categorical'] if request['split']=='dev' else ['categorical','greedy']
    trace_count=0
    for mode in modes:
        episodes=[]
        for f,h in nc['files'].items():
            p=out/'native'/f;assert sha(p)==h
            if f.startswith(mode+'_'):episodes.extend(read(p)['episodes'])
        assert len(episodes)==games and {e['episode_index'] for e in episodes}==set(range(games))
        for e in episodes:
            assert sha(ROOT/e['provenance']['trace'])==e['provenance']['trace_sha256']
            trace_count+=1
        counts=nv['natural_jax'][mode]
        assert counts['games']==games and counts['capture_episodes']==sum(e['events']['capture']>0 for e in episodes)
        assert counts['death_episodes']==sum(e['events']['death']>0 for e in episodes)
    return dict(summary_sha256=sha(out/'SUMMARY.json'),native_summary_sha256=nc['summary_sha256'],
                checkpoint=request['checkpoint'],policy_sha256=marker['policy_sha256'],
                native_trace_count=trace_count,JS_reused_unchanged=True,unix=time.time())

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    guard=lock(BASE/'NATIVE_RECOVERY_VERIFY.lock')
    target=BASE/'NATIVE_RECOVERY_ARTIFACT_VALIDATION.json'
    old=read(target) if target.exists() else {};validated=old.get('validated',{})
    queue=read(BASE/'NATIVE_EVAL_RECOVERY_QUEUE.json');inv={e['name']:e for e in queue['inventory']}
    while True:
        for name in queue['names']:
            if name in validated or not (BASE/'eval'/name/'COMPLETE.json').exists():continue
            try:validated[name]=verify(name,inv[name])
            except Exception as e:
                atomic_json(target,dict(status='BLOCKED',failed=name,error=repr(e),validated=validated,unix=time.time()))
                raise
        remaining=[n for n in queue['names'] if n not in validated]
        atomic_json(target,dict(status='PASS' if not remaining else 'RUNNING',validated=validated,remaining=remaining,unix=time.time()))
        print(json.dumps(dict(validated=len(validated),remaining=len(remaining),unix=time.time())),flush=True)
        if not remaining or not args.watch:return
        time.sleep(20)
if __name__=='__main__':main()
