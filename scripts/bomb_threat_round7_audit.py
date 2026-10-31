"""Read-only ITER2 round7 reward audit; never changes a training recipe."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/bomb_threat_v1')


def summarize(rows, components):
    """Keep paid reward units separate from unrecorded event-linked funnels."""
    if not rows or any(row.get('finite') is not True for row in rows):
        raise ValueError('finite native update rows required')
    required = ('transport_shape', 'transport_capture', 'control_entry_count',
                'control_residual_count', 'route_shaped_reward')
    if any(name not in components for name in required):
        raise ValueError('native component ABI is incomplete')
    parts = [row['exposure']['native_parts'] for row in rows]
    if any(len(p) != len(components) for p in parts):
        raise ValueError('native component ABI width mismatch')
    sums = {name: sum(p[i] for p in parts) for i, name in enumerate(components)}
    total = sum(row['exposure']['total_reward'] for row in rows)
    return dict(
        update_range=[rows[0]['new_update'], rows[-1]['new_update']], updates=len(rows),
        environment_ticks=sum(row['actual_environment_steps_this_update'] for row in rows),
        collector_resets=sum(row['exposure']['resets'] for row in rows),
        bank_resets=sum(row['exposure']['bank_resets'] for row in rows),
        delivery_events=sum(row['exposure']['captures'] for row in rows),
        reward_total=total, transport_capture_paid=sums['transport_capture'],
        transport_capture_payment_units=sums['transport_capture'] / 24.,
        transport_capture_share_of_net_reward=sums['transport_capture'] / total if total > 0 else None,
        transport_shape_paid=sums['transport_shape'],
        first_eligible_stock_payment_units=sums['transport_shape'] / .75,
        route_shaped_reward=sums['route_shaped_reward'],
        attributed_native_control_entry_events=sums['control_entry_count'],
        attributed_native_control_death_residual_events=sums['control_residual_count'],
        raw_kill_window_count=None, raw_window_pickup_count=None,
        same_window_pickup_delivery_count=None,
        limitations=[
            'Training mixture contains natural, combat-scene and certified Bank resets; not full natural games.',
            'control_entry_count counts attributed native sugar entries, not all death windows.',
            'control_residual_count counts rewarded attributed controlled deaths, not all causal kills.',
            'Payment units are reward amount divided by the frozen coefficient; not raw pickup/capture events.',
            'Existing logs do not record window IDs or window-to-pickup-to-delivery linkage.',
            'Episode/reset counts and Bank counts are not event-specific source denominators.',
        ])


def freeze(base=BASE):
    out = base / 'iter2/round7_audit'
    manifest_path = out / 'MANIFEST.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    slots = json.loads((base / 'SLOTS.json').read_text())
    active = slots['1']
    run = active['run']
    candidates = {}
    # Active continuation and its immediate parent are sufficient for the last
    # three 40-update points; completed aliases never replace a newer lineage.
    runs = [run]
    if active.get('source'):
        parent = Path(active['source']['checkpoint']).parent.name
        if parent not in runs:
            runs.append(parent)
    for name in reversed(runs):
        for marker in (base / 'formal' / name).glob('u*/COMMITTED.json'):
            cm = json.loads(marker.read_text())
            u = cm['update']
            if u > 0 and u % 40 == 0:
                candidates[u] = (marker, cm)
    selected = sorted(candidates)[-3:]
    if len(selected) != 3:
        raise ValueError('three committed native update windows required')
    windows = []
    for u in selected:
        marker, cm = candidates[u]
        cp = marker.parent
        if sha(cp / 'policy.pt') != cm['policy_sha256']:
            raise ValueError('atomic policy hash mismatch')
        run_path = cp.parent
        records = {}
        for line in (run_path / 'updates.jsonl').read_text().splitlines():
            row = json.loads(line)
            if u - 40 < row['new_update'] <= u:
                if row['new_update'] in records:
                    raise ValueError('duplicate native update')
                records[row['new_update']] = row
        if set(records) != set(range(u - 39, u + 1)):
            raise ValueError('missing committed update rows')
        snapshot = out / f'{run_path.name}_u{u:06d}_rows.json'
        atomic_json(snapshot, [records[k] for k in sorted(records)])
        windows.append(dict(checkpoint=str(cp), update=u,
                            policy_sha256=cm['policy_sha256'],
                            commit_sha256=sha(marker), rows=str(snapshot), rows_sha256=sha(snapshot)))
    sources = [
        'scripts/bomb_threat_iter2_diagnostics_train.py',
        'scripts/bomb_threat_collector_diagnostics_v1.py',
        'scripts/carrier_mechanism_engine.py', 'scripts/combat_item_engine.py',
        'scripts/je10_ops.py', 'scripts/je09_stage_f_reward.py',
        'scripts/je05_bun_transport.py', 'scripts/je09_route_pbrs.py',
    ]
    manifest = dict(status='FROZEN_READ_ONLY', decision='.hermes/STEER_20261029_1530.md',
                    observed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                    selection='latest three committed 40-update points on active slot01 A3 lineage',
                    windows=windows, source_sha256={p: sha(Path(p)) for p in sources},
                    new_environment_steps=0, optimizer_steps=0, training_changed=False)
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    manifest = freeze()
    if args.freeze_only:
        print(json.dumps(manifest, ensure_ascii=False))
        return
    for source, expected in manifest['source_sha256'].items():
        if sha(Path(source)) != expected:
            raise ValueError('frozen audit source changed: ' + source)
    # CPU process only. Install the same native adapter to obtain its actual
    # component ABI, without running an environment transition or optimizer.
    from scripts import ppo_scale_longrun_train as P
    from scripts.bomb_threat_runtime import install
    P.T.configure()
    P.C.install_provenance()
    install()
    from scripts import combat_item_engine as T
    components = list(T.BASE_COMPONENTS[:-1]) + [
        'banana_completed', 'banana_benefit', 'banana_pay',
        'glue_completed', 'glue_benefit', 'glue_pay', 'item_extra', 'combined_total']
    windows = []
    for item in manifest['windows']:
        if sha(Path(item['rows'])) != item['rows_sha256']:
            raise ValueError('frozen row snapshot changed')
        windows.append(dict(item, **summarize(json.loads(Path(item['rows']).read_text()), components)))
    result = dict(status='BLOCKED_REWARD_PREMISE_CONTRADICTED',
                  manifest='runs/bomb_threat_v1/iter2/round7_audit/MANIFEST.json',
                  manifest_sha256=sha(BASE / 'iter2/round7_audit/MANIFEST.json'),
                  existing_delivery_reward=24., existing_first_eligible_stock_reward=.75,
                  proposed_delivery_increment=.10, proposed_increment_over_existing_delivery=.10 / 24.,
                  prerequisite_no_delivery_or_stock_reward=False,
                  native_components=components, windows=windows,
                  raw_funnel_status='UNAVAILABLE_IN_FROZEN_TRAINING_LOGS',
                  new_environment_steps=0, optimizer_steps=0, reward_changed=False,
                  A4_started=False, slot07_continues_A3=True)
    atomic_json(BASE / 'iter2/round7_audit/RESULT.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
