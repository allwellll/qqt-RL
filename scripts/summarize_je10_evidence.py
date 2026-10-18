"""Read frozen JE10 evidence without importing training or loading checkpoints.

Only COMPLETE points are admitted. Every listed shard is verified before any
metric is exported. Active STATE is sampled separately and is never a score.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

POINTS = (0, 4, 8, 16, 32, 64, 96, 128)
TASKS = ('native_death_stock', 'native_sugar_stock', 'real_drop', 'carry_near',
         'carry_mid', 'carry_far', 'recovery_threat', 'natural_full2400')


def stable_bytes(path):
    path = Path(path)
    first = path.read_bytes()
    if path.read_bytes() != first:
        raise ValueError(f'changing evidence: {path.name}')
    return first


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    data = stable_bytes(path)
    return json.loads(data), sha(data)


def verified_point(directory):
    directory = Path(directory)
    complete, complete_hash = load(directory / 'COMPLETE.json')
    if complete.get('passed') is not True:
        raise ValueError('point is not complete')
    for name, expected in complete['shards'].items():
        target = directory / name
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError('shard escapes point directory')
        if sha(stable_bytes(target)) != expected:
            raise ValueError(f'shard hash mismatch: {name}')
    summary, summary_hash = load(directory / 'SUMMARY.json')
    if complete['shards'].get('SUMMARY.json') != summary_hash:
        raise ValueError('summary must be covered by COMPLETE')
    if (summary['generation'] != complete['generation'] or
            summary['policy_sha256'] != complete['checkpoint_sha256']):
        raise ValueError('checkpoint identity mismatch')
    return summary, dict(complete_sha256=complete_hash,
                         summary_sha256=summary_hash, shards=complete['shards'])


def route_summary(route):
    from scripts.analyze_control_route_curve import route_rates, route_strata
    if route['games'] != 16 or len(route['episodes']) != 16:
        raise ValueError('route denominator must be 16')
    seeds = [e['seed'] for e in route['episodes']]
    if seeds != list(range(2026101800, 2026101816)):
        raise ValueError('unexpected route seeds')
    for key, event in [('pickup_episodes', 'pickup'), ('delivery_episodes', 'delivery')]:
        if route[key] != sum(e[event] > 0 for e in route['episodes']):
            raise ValueError(f'episode count mismatch: {key}')
    fields = ('games', 'pickup_episodes', 'delivery_episodes', 'death', 'self_death',
              'oracle_opportunity_episodes', 'first_oracle_match_episodes',
              'oracle_opportunities', 'oracle_matches', 'target_ticks', 'progress_ticks',
              'first_control', 'attributed_first_control', 'final_pop',
              'active_contact_pop', 'causal_kill', 'surviving_kill')
    return dict(counts={k: route.get(k) for k in fields},
                rates=route_rates(route), strata=route_strata(route),
                stage_funnel=route.get('stage_funnel', {}))


def snapshot(source_root):
    root = Path(source_root)
    base = root / 'runs/je10_bun_20261018'
    result = dict(schema='je10.documentation-evidence/v1',
                  cutoff_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Read-only observation; no evaluation or checkpoint loading',
                  rounds=[], source_documents={})
    for filename in ('reports/je10_training_20261018.md', 'docs/training_experiment_ledger.md'):
        result['source_documents'][filename] = sha(stable_bytes(root / filename))
    for directory in sorted(base.glob('control_route_v38_r*')):
        declaration, declaration_hash = load(directory / 'ROUND_DECLARATION.json')
        plan, plan_hash = load(directory / 'plan.json')
        state, state_hash = load(directory / 'formal/bun392_bridge_pfsp8/STATE.json')
        protocol, protocol_hash = load(directory / 'authority_je10/PROTOCOL.json')
        row = dict(name=directory.name, declaration_sha256=declaration_hash,
                   plan_sha256=plan_hash, protocol_sha256=protocol_hash,
                   seed=plan['seed'], change=declaration['change'],
                   technical_seed=declaration['seeds']['technical'],
                   architecture=plan['architecture'],
                   plan={k: plan[k] for k in ('generations', 'fresh_ticks_per_generation',
                         'total_fresh_ticks', 'combat_fraction', 'bridge_fraction',
                         'ring_interval', 'route_pbrs', 'dagger')},
                   state={k: state[k] for k in ('generation', 'fresh_ticks', 'finite',
                          'nonzero_update', 'unix')}, state_sha256=state_hash,
                   points=[], terminal=None, successor=None)
        latest, latest_hash = load(directory / 'formal/bun392_bridge_pfsp8/LATEST_CHECKPOINT.json')
        committed, committed_hash = load(root / latest['checkpoint'] / 'COMMITTED.json')
        assert committed_hash == latest['committed_sha256']
        assert committed['passed'] and committed['replicas_exact']
        row['latest_checkpoint'] = dict(generation=committed['generation'],
            fresh_ticks=committed['fresh_ticks'], policy_sha256=committed['policy_sha256'],
            state_sha256=committed['state_sha256'], committed_sha256=committed_hash,
            latest_marker_sha256=latest_hash)
        for filename, output_key in [('RUN_TERMINAL.json', 'terminal'),
                                     ('SUCCESSOR_RECIPE.json', 'successor'),
                                     ('NEXT_RECIPE_PROPOSAL.json', 'next_proposal')]:
            path = directory / filename
            if path.exists():
                value, digest = load(path)
                fields = ('completed', 'generation', 'fresh_ticks', 'compute_seconds',
                          'wall_seconds', 'gpu_release_verified',
                          'successor_formal_nonzero_verified', 'unix') if output_key == 'terminal' else (
                          'recipe', 'changed_factor', 'reason', 'seeds', 'predeclared_fallback')
                row[output_key] = dict(sha256=digest, **{k: value[k] for k in fields if k in value})
        for generation in POINTS:
            point_dir = directory / f'authority_je10/g{generation:06d}'
            if not (point_dir / 'COMPLETE.json').exists():
                continue
            summary, proof = verified_point(point_dir)
            assert summary['generation'] == generation
            assert all(summary['hunter_js'][t]['games'] == 64 for t in ('hard', 'normal', 'easy'))
            assert summary['tactical_python']['games'] == 64
            row['points'].append(dict(generation=generation,
                fresh_ticks=generation * plan['fresh_ticks_per_generation'],
                policy_sha256=summary['policy_sha256'], proof=proof,
                hunter_js=summary['hunter_js'], tactical_python=summary['tactical_python'],
                routes={task: route_summary(summary['route'][task]) for task in TASKS},
                exposure=summary['actual_training_exposure']))
        if row['terminal'] and row['terminal']['completed']:
            assert [p['generation'] for p in row['points']] == list(POINTS)
        # Completed-run diagnostics are archived as selected fields, never raw run files.
        if row['terminal']:
            path = directory / 'FULL_CURVE_ROOT_CAUSE_REVIEW.json'
            if path.exists():
                review, digest = load(path)
                row['diagnostic_review'] = dict(sha256=digest, **{
                    k: review[k] for k in ('training', 'joint_fit_curve', 'actual_preparation_curve',
                                           'return_fifo', 'fixed_review') if k in review})
        result['rounds'].append(row)
    result['read_completed_utc'] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source_root.resolve()):
        parser.error('output must be outside the read-only source tree')
    value = snapshot(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(f"verified {sum(len(r['points']) for r in value['rounds'])} fixed points")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
