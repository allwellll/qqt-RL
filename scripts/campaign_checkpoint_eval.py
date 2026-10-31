"""Passive CPU full-game checkpoint consumer, independent of GPU scheduling."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]


def validate_source(spec, root=ROOT):
    checkpoint = root / spec['checkpoint']
    committed = json.loads((checkpoint / 'COMMITTED.json').read_text())
    for filename, field in [('state.pkl', 'state_sha256'), ('policy.pt', 'policy_sha256')]:
        if sha(checkpoint / filename) != committed[field]:
            raise ValueError(f'Checkpoint hash mismatch: {filename}')
    if spec.get('queue'):
        queue = json.loads((root / spec['queue']).read_text())
        if queue['stage'] != 'formal' or queue['checkpoint'] != spec['checkpoint']:
            raise ValueError('Only registered formal checkpoint requests are accepted')
        if any(queue[key] != committed[key] for key in ('state_sha256', 'policy_sha256')):
            raise ValueError('Queue/checkpoint atomic hash mismatch')
    return committed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('protocol'); p.add_argument('condition'); args = p.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if hasattr(os, 'sched_getaffinity'):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[-protocol['max_cpu_cores']:])
    for filename, expected in protocol['source_sha256'].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(f'Frozen evaluation source changed: {filename}')
    spec = protocol['conditions'][args.condition]
    committed = validate_source(spec)
    output = ROOT / protocol['namespace'] / args.condition
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
        FULLMATCH_OUTPUT=str(output), FULLMATCH_GAMES_PER_MODE=str(spec['games_per_mode']),
        FULLMATCH_SEED_START=str(spec['seed_start']), FULLMATCH_MODELS='')
    from scripts import expert_reward_alignment_fullmatch as F
    F.MODELS = {args.condition: dict(state=spec['checkpoint']+'/state.pkl',
        checkpoint=spec['checkpoint'], run_seed=spec['run_seed'])}
    original = F.write_manifest
    def write_manifest():
        value = original()
        value.update(id=protocol['id']+':'+args.condition,
            campaign_protocol=args.protocol, campaign_protocol_sha256=sha(args.protocol),
            split=spec['split'], completion=f"{spec['games_per_mode']} games per action mode",
            incomplete_D_coverage='Independent reward-task-semantics V2 checkpoint evaluation',
            evaluation_reward_caveat=protocol['reward_scoring_caveat'],
            registered_parent=spec['parent'], no_training_gate=True)
        atomic_json(F.MANIFEST, value)
        return value
    F.write_manifest = write_manifest
    original_batch = F.run_batch
    def batch(params, model, mode, indices, scan, names):
        path = output/'batches'/f'{mode}_{indices[0]:05d}.json'
        if path.exists():
            value = json.loads(path.read_text())
            assert value['state_sha256'] == committed['state_sha256']
            for episode in value['episodes']:
                assert sha(ROOT/episode['provenance']['trace']) == episode['provenance']['trace_sha256']
            return value['episodes']
        episodes = original_batch(params, model, mode, indices, scan, names)
        atomic_json(path, dict(passed=True, state_sha256=committed['state_sha256'], episodes=episodes))
        return episodes
    F.run_batch = batch
    if (output/'COMPLETE.json').exists():
        done = json.loads((output/'COMPLETE.json').read_text())
        assert done['passed'] and done['summary_sha256'] == sha(output/'SUMMARY.json')
        assert done['manifest_sha256'] == sha(F.MANIFEST)
        return
    F.main()


if __name__ == '__main__':
    main()
