"""Verify immutable per-game caches after completion, without rerunning games."""
import argparse
import json
from pathlib import Path
import time
from scripts.population_run_state import atomic_json,sha,lock

def verify(protocol_path,condition,kind,root):
    protocol=json.loads((root/protocol_path).read_text())
    spec=protocol['checkpoints'][condition] if kind=='native' else protocol['conditions'][condition]
    output=root/protocol['namespace']/condition
    guard=lock(output/'merge_verification.lock')
    completed=json.loads((output/'COMPLETE.json').read_text())
    assert completed['passed']
    assert completed['summary_sha256']==sha(output/'SUMMARY.json')
    assert completed['manifest_sha256']==sha(output/'MANIFEST.json')
    games=protocol['games_per_condition'] if kind=='native' else spec['games_per_mode']
    seed_start=protocol['dev_seed_start'] if kind=='native' else spec['seed_start']
    state=root/(spec['state'] if kind=='native' else spec['checkpoint']+'/state.pkl')
    state_hash=sha(state);sources=[];truncations=0;terminations={}
    for mode in ('categorical','greedy'):
        seen=[]
        for start in range(0,games,8):
            path=output/'batches'/f'{mode}_{start:05d}.json'
            batch=json.loads(path.read_text())
            assert batch['passed']
            assert batch.get('source_state_sha256',batch.get('state_sha256'))==state_hash
            rows=batch['episodes'];expected=list(range(start,min(start+8,games)))
            assert [e['episode_index'] for e in rows]==expected
            assert [e['seed'] for e in rows]==[seed_start+i for i in expected]
            for row in rows:
                provenance=row['provenance'];assert provenance['state_sha256']==state_hash
                assert sha(root/provenance['trace'])==provenance['trace_sha256']
                assert isinstance(row['artificial_truncation'],bool)
                truncations+=int(row['artificial_truncation'])
                terminations[row['termination']]=terminations.get(row['termination'],0)+1
            seen.extend(expected)
            sources.append(dict(path=str(path.relative_to(root)),sha256=sha(path),action_mode=mode,seeds=[e['seed'] for e in rows]))
        assert seen==list(range(games))
    assert completed['total_games']==2*games
    value=dict(passed=True,total_games=2*games,protocol=protocol_path,protocol_sha256=sha(root/protocol_path),state_sha256=state_hash,summary_sha256=sha(output/'SUMMARY.json'),source_batches=sources,seed_sets_exact=True,duplicate_seeds_within_condition=0,trace_sha_verified=True,artificial_truncations=truncations,termination_counts=terminations,all_raw_rows_retained=True,no_new_games_run=True,unix=time.time())
    atomic_json(output/'PARALLEL_MERGE.json',value)
    return value

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('protocol');parser.add_argument('condition');parser.add_argument('kind',choices=['native','checkpoint']);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    value=verify(a.protocol,a.condition,a.kind,root)
    print(json.dumps({k:value[k] for k in ('passed','total_games','seed_sets_exact','trace_sha_verified','artificial_truncations')}))

if __name__=='__main__':main()
