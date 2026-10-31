"""Resumable full natural-game matrix; retains each complete source batch."""
import argparse
import json
import os
from pathlib import Path
from scripts.population_run_state import atomic_json,sha

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('protocol');p.add_argument('model');a=p.parse_args()
    protocol=json.loads(Path(a.protocol).read_text());spec=protocol['checkpoints'][a.model]
    root=Path(__file__).resolve().parents[1]
    os.environ.update(FULLMATCH_OUTPUT=str(root/protocol['namespace']/a.model),
        FULLMATCH_GAMES_PER_MODE=str(protocol['games_per_condition']),
        FULLMATCH_SEED_START=str(protocol['dev_seed_start']),FULLMATCH_MODELS='')
    from scripts import expert_reward_alignment_fullmatch as F
    F.MODELS={a.model:spec}
    original_manifest=F.write_manifest
    def registered_manifest():
        manifest=original_manifest()
        manifest.update(id=protocol['id']+':'+a.model,
            completion=f"{F.GAMES_PER_MODE} paired games per checkpoint/action mode; no result selection",
            incomplete_D_coverage='All D fixed u320 endpoints selected before this evaluation; no peak selection',
            campaign_protocol=a.protocol,campaign_protocol_sha256=sha(a.protocol),
            caveat='layout0/P0 native-JAX rule3; not cross-map/seat generalization or JS Hunter')
        atomic_json(F.MANIFEST,manifest);return manifest
    F.write_manifest=registered_manifest
    original_batch=F.run_batch
    def durable_batch(params,model,mode,indices,scan,names):
        output=F.OUT/'batches'/f'{mode}_{indices[0]:05d}.json'
        if output.exists():
            cached=json.loads(output.read_text())
            assert cached['source_state_sha256']==sha(F.R/spec['state'])
            for ep in cached['episodes']:
                assert sha(F.R/ep['provenance']['trace'])==ep['provenance']['trace_sha256']
            return cached['episodes']
        result=original_batch(params,model,mode,indices,scan,names)
        atomic_json(output,dict(passed=True,source_state_sha256=sha(F.R/spec['state']),episodes=result))
        return result
    F.run_batch=durable_batch
    if (F.OUT/'COMPLETE.json').exists():
        done=json.loads((F.OUT/'COMPLETE.json').read_text())
        if done.get('passed'):
            assert done['manifest_sha256']==sha(F.MANIFEST)
            assert done['summary_sha256']==sha(F.OUT/'SUMMARY.json')
            return
    F.main()

if __name__=='__main__':main()
