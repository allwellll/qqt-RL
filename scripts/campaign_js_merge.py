"""Passive verification of every frozen production-JS game after completion."""
import argparse
import json
from pathlib import Path
import time
from scripts.population_run_state import atomic_json,sha,lock
from scripts.campaign_js_parallel import tasks,validate_batch,truncation_coverage

def verify(protocol,root):
    spec=json.loads((root/protocol).read_text());out=root/spec['namespace']
    guard=lock(out/'js_merge_verification.lock')
    complete=json.loads((out/'COMPLETE.json').read_text())
    assert complete['passed'] and complete['summary_sha256']==sha(out/'SUMMARY.json')
    summary=json.loads((out/'SUMMARY.json').read_text())
    expected=[];sources=[];coverage=dict(available=0,missing=0,flagged=0);termination=dict(explicit=0,missing=0)
    for name,mode,start,end,path in tasks(spec,out):
        value=json.loads(path.read_text());episodes=validate_batch(value,spec,name,start,end)
        for key,count in truncation_coverage(episodes).items():coverage[key]+=count
        for episode in episodes:
            expected.append((name,mode,episode['seed']))
            termination['explicit' if 'termination' in episode else 'missing']+=1
        sources.append(dict(path=str(path.relative_to(root)),sha256=sha(path),model=name,action_mode=mode,seeds=[e['seed'] for e in episodes]))
    assert [(r['model'],r['action_mode'],r['episode']['seed']) for r in summary['rows']]==expected
    assert len(expected)==len(set(expected))==complete['total_games']
    for model in spec['models'].values():
        assert sha(root/model['model_json'])==model['model_json_sha256']
    value=dict(passed=True,total_games=len(expected),protocol_sha256=sha(root/protocol),summary_sha256=sha(out/'SUMMARY.json'),seed_sets_exact=True,source_batches=sources,raw_sha_verified=True,artificial_truncation_field_coverage=coverage,termination_field_coverage=termination,missing_fields_never_imputed=True,no_new_games_run=True,unix=time.time())
    atomic_json(out/'PARALLEL_MERGE_VERIFIED.json',value);return value

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('protocol');a=p.parse_args()
    print(json.dumps({k:v for k,v in verify(a.protocol,Path(__file__).resolve().parents[1]).items() if k!='source_batches'}))

if __name__=='__main__':main()
