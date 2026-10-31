"""Recompute passive evaluation from immutable numeric traces, without JAX/GPU.

Usage: <PYTHON> -m scripts.passive_evaluation_report <evaluation-namespace>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import passive_evaluation_metrics as metrics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paired_difference(parent, candidate, *, bootstrap_seed=20261026):
    """Game-paired bootstrap only; it does not estimate training-seed variance."""
    a, b = np.asarray(parent, float), np.asarray(candidate, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('paired arrays must have identical one-dimensional shape')
    if not a.size:
        return dict(paired_games=0, mean_delta=None, game_pair_bootstrap_ci95=None)
    delta = b-a
    rng = np.random.default_rng(bootstrap_seed)
    means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(-1)
    return dict(paired_games=int(a.size), mean_delta=float(delta.mean()),
                game_pair_bootstrap_ci95=np.quantile(means, [.025,.975]).tolist(),
                bootstrap_replicates=10000, bootstrap_seed=bootstrap_seed,
                parent_only_positive=int(np.sum((a>0)&(b<=0))),
                candidate_only_positive=int(np.sum((b>0)&(a<=0))),
                both_positive=int(np.sum((a>0)&(b>0))),
                both_nonpositive=int(np.sum((a<=0)&(b<=0))),
                independent_unit='paired source game, not tick/event',
                training_seed_variance_estimated=False)


def paired_ratio_difference(parent_num, parent_den, candidate_num, candidate_den,
                            *, bootstrap_seed=20261026):
    """Pool event denominators and bootstrap paired whole source games."""
    arrays = [np.asarray(v, float) for v in
              (parent_num, parent_den, candidate_num, candidate_den)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('event counts must have identical one-dimensional shape')
    pn, pd, cn, cd = arrays
    if any(not np.isfinite(a).all() or (a < 0).any() for a in arrays) or \
            (pn > pd).any() or (cn > cd).any():
        raise ValueError('event numerators must lie between zero and denominators')
    pr, cr = metrics.ratio(pn.sum(), pd.sum()), metrics.ratio(cn.sum(), cd.sum())
    result = dict(paired_games=len(pn), parent_numerator=int(pn.sum()),
        parent_denominator=int(pd.sum()), candidate_numerator=int(cn.sum()),
        candidate_denominator=int(cd.sum()), parent_rate=pr, candidate_rate=cr,
        mean_delta=cr-pr if pr is not None and cr is not None else None,
        game_pair_bootstrap_ci95=None, bootstrap_replicates=10000,
        bootstrap_defined_replicates=0, bootstrap_seed=bootstrap_seed,
        independent_unit='paired source game, not tick/event',
        training_seed_variance_estimated=False)
    if not pn.size:
        return result
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(pn), (10000, len(pn)))
    pden, cden = pd[indices].sum(-1), cd[indices].sum(-1)
    valid = (pden > 0) & (cden > 0)
    result['bootstrap_defined_replicates'] = int(valid.sum())
    # Preserve undefined replicates rather than silently reporting a conditional
    # interval that drops zero-opportunity source-game resamples.
    if valid.all():
        differences = cn[indices].sum(-1)/cden - pn[indices].sum(-1)/pden
        result['game_pair_bootstrap_ci95'] = np.quantile(differences, [.025,.975]).tolist()
    else:
        result['ci_unavailable_reason'] = 'Some paired source-game resamples have zero opportunity denominators.'
    return result


def transition_scope(rows):
    """Expose residual reward-settlement ticks after the first native outcome."""
    terminal = next((i for i,r in enumerate(rows) if r.get('winner') in (-1,0,1)), None)
    cutoff = terminal+1 if terminal is not None else len(rows)
    tail = rows[cutoff:]
    return dict(valid_transitions=len(rows), native_gameplay_transitions=cutoff,
        native_terminal_observed=terminal is not None,
        settlement_tail_transitions=len(tail),
        tail_own_deaths=sum(bool(r.get('own_death')) for r in tail),
        tail_deliveries=sum(bool(r.get('own_capture')) for r in tail),
        scope='Detailed cohort exposure includes the residual reward-settlement tail; outcome is the first native terminal winner.')


def recompute_episode(ep, manifest, root=ROOT):
    """Verify raw rewards and rebuild all supported passive denominators."""
    provenance = ep['provenance']
    path = root/provenance['trace']
    if sha(path) != provenance['trace_sha256']:
        raise ValueError('numeric trace hash mismatch')
    with np.load(path, allow_pickle=False) as saved:
        col = provenance['telemetry_column']
        indices = saved['episode_indices'].tolist()
        if indices[col] != ep['episode_index']:
            raise ValueError('trace column does not identify this episode')
        active = saved['active'][:,col].astype(bool)
        names = saved['component_names'].tolist()
        tick_names = saved['telemetry_names'].tolist()
        raw = saved['telemetry'][active,col]
        rows = []
        for tick in raw:
            row = dict(zip(tick_names, map(float, tick)))
            row['phase'] = metrics.PHASES[int(row.pop('phase_code'))]
            rows.append(row)
        errors = {}
        for arm in ('legacy','minimal'):
            parts = np.asarray(saved[f'{arm}_parts'][active,col], np.float64)
            reward = np.asarray(saved[f'{arm}_reward'][active,col], np.float64)
            terms = manifest['reward_scoring'][f'exclusive_{arm}_terms']
            if arm == 'minimal':
                terms = ('transport_capture','settled_kill','global_safety',
                         'control_entry_reward','control_residual_reward','route_shaped_reward')
            recomposed = parts[:,[names.index(n) for n in terms]].sum(-1)
            total = parts[:,names.index('combined_total')]
            max_error = float(np.max(np.abs(recomposed-reward), initial=0))
            total_error = float(np.max(np.abs(total-reward), initial=0))
            if max(max_error,total_error) > manifest['reward_scoring']['per_transition_decomposition_tolerance']:
                raise ValueError('raw exclusive reward terms do not recompose actual reward')
            errors[arm] = dict(max_abs_exclusive_to_actual_reward=max_error,
                               max_abs_total_to_actual_reward=total_error,
                               undiscounted_return=float(reward.sum()),
                               discounted_return=float(np.dot(
                                   reward, manifest['gamma']**np.arange(reward.size))))
    original = ep['passive_metrics']
    winner = next((int(r['winner']) for r in rows if r['winner'] in (-1,0,1)), None)
    computed = metrics.summarize_game(original['match_id'], rows,
        termination=ep['termination'], winner=winner,
        model=original['model'], action_mode=original['action_mode'])
    if computed != original:
        raise ValueError(f"passive metrics do not recompute exactly: {original['match_id']}")
    errors['transition_scope'] = transition_scope(rows)
    return computed, errors


def build_report(namespace, root=ROOT):
    namespace = Path(namespace)
    manifest = json.loads((namespace/'MANIFEST.json').read_text())
    complete = json.loads((namespace/'COMPLETE.json').read_text())
    summary = json.loads((namespace/'SUMMARY.json').read_text())
    if sha(namespace/'MANIFEST.json') != complete['manifest_sha256'] or \
            sha(namespace/'SUMMARY.json') != complete['summary_sha256']:
        raise ValueError('completed evaluation hash mismatch')
    expected_seeds = manifest['paired_seed_list']
    expected_rows = {(model,mode) for model in manifest['checkpoints']
                     for mode in manifest['action_modes']}
    if {(r['model'],r['action_mode']) for r in summary['rows']} != expected_rows:
        raise ValueError('model/action-mode coverage incomplete')
    report_rows, raw_errors, games_by_condition = [], [], {}
    for row in summary['rows']:
        if [ep['seed'] for ep in row['episodes']] != expected_seeds:
            raise ValueError('seed coverage differs from frozen paired manifest')
        games, scopes = [], []
        for ep in row['episodes']:
            game, errors = recompute_episode(ep, manifest, root)
            games.append(game)
            raw_errors.append(errors)
            scopes.append(errors['transition_scope'])
        key = (row['model'], row['action_mode'])
        games_by_condition[key] = games
        report_rows.append(dict(model=key[0],action_mode=key[1],
            policy_sha256=manifest['checkpoints'][key[0]]['policy_sha256'],
            state_sha256=manifest['checkpoints'][key[0]]['state_sha256'],
            engine=manifest['engine'],opponent=manifest['opponent'],
            seat=manifest['seat'],split=manifest['split'],
            games=len(games),cohort=metrics.summarize_cohort(games),
            transition_scope={key:sum(s[key] for s in scopes) for key in
                ('valid_transitions','native_gameplay_transitions',
                 'settlement_tail_transitions','tail_own_deaths','tail_deliveries')},
            rewards={a:row[a] for a in ('legacy','minimal')}))
    parent, candidate = 'parent_r1_s2751_u064','minimal_s2851_u064'
    pairs = []
    for mode in manifest['action_modes']:
        if (parent,mode) not in games_by_condition or (candidate,mode) not in games_by_condition:
            continue
        a, b = games_by_condition[parent,mode], games_by_condition[candidate,mode]
        pair = dict(action_mode=mode,parent=parent,candidate=candidate,
                    game_seeds=expected_seeds)
        for metric, getter in (
                ('delivery_match', lambda g:g['full_game']['at_least_one_delivery']['numerator']),
                ('own_death_events', lambda g:g['full_game']['own_deaths']['count']),
                ('native_match_win', lambda g:g['full_game']['net_wins']['numerator'])):
            pair[metric] = paired_difference([getter(g) for g in a],[getter(g) for g in b])
        for name, getter in (
                ('death_window_pickup', lambda g:g['window_funnel']['by_kind']['death']['pickup_per_closed_window']),
                ('syrup_window_pickup', lambda g:g['window_funnel']['by_kind']['native_syrup']['pickup_per_closed_window']),
                ('same_carry_delivery', lambda g:g['carry_episodes']['same_episode_delivery_rate'])):
            ar, br = [getter(g) for g in a], [getter(g) for g in b]
            pair[name] = paired_ratio_difference(
                [r['numerator'] for r in ar], [r['denominator'] for r in ar],
                [r['numerator'] for r in br], [r['denominator'] for r in br])
        pair['seed_rows'] = [dict(seed=seed,
            parent_delivery=a[i]['full_game']['delivery_events']['count'],
            candidate_delivery=b[i]['full_game']['delivery_events']['count'],
            parent_deaths=a[i]['full_game']['own_deaths']['count'],
            candidate_deaths=b[i]['full_game']['own_deaths']['count'])
            for i,seed in enumerate(expected_seeds)]
        pairs.append(pair)
    mode_pairs = []
    for model in manifest['checkpoints']:
        if (model,'categorical') not in games_by_condition or (model,'greedy') not in games_by_condition:
            continue
        a, b = games_by_condition[model,'categorical'], games_by_condition[model,'greedy']
        mode_pairs.append(dict(model=model, delta='greedy minus categorical',
            delivery_match=paired_difference(
                [g['full_game']['at_least_one_delivery']['numerator'] for g in a],
                [g['full_game']['at_least_one_delivery']['numerator'] for g in b]),
            own_death_events=paired_difference(
                [g['full_game']['own_deaths']['count'] for g in a],
                [g['full_game']['own_deaths']['count'] for g in b])))
    coverage = metrics.coverage_matrix()
    return dict(schema='qqt.passive-recomputable-acceptance/v2',
        namespace=namespace.relative_to(root).as_posix(),
        source_manifest_sha256=sha(namespace/'MANIFEST.json'),
        source_summary_sha256=sha(namespace/'SUMMARY.json'),
        metrics_source_sha256=sha(root/'scripts/passive_evaluation_metrics.py'),
        report_source_sha256=sha(root/'scripts/passive_evaluation_report.py'),
        all_games_recomputed_exactly=True, games=sum(r['games'] for r in report_rows),
        rows=report_rows,paired_comparisons=pairs,action_mode_comparisons=mode_pairs,
        reward_max_abs_exclusive_to_actual={arm:max(e[arm]['max_abs_exclusive_to_actual_reward']
            for e in raw_errors) for arm in ('legacy','minimal')},
        reward_max_abs_total_to_actual={arm:max(e[arm]['max_abs_total_to_actual_reward']
            for e in raw_errors) for arm in ('legacy','minimal')},
        coverage=coverage,
        undefined_metrics={group:{key:dict(value=None,reason=reason) for key,reason in coverage[group].items()}
                           for group in ('missing_data','unavailable')},
        no_training_or_queue_control=True,
        limitations=['One candidate training seed at intermediate u064; no final reward-package inference.',
                     'Native JAX rule3, not JS Hunter or Python Tactical.',
                     'Fixed natural layout0 seat0 diagnostic dev; no independent holdout.',
                     'Paired intervals describe game variation only; no holdout tuning or promotion.',
                     'Detailed phase/window/carry/item exposure includes residual reward-settlement transitions; transition_scope reports the tail separately.',
                     'Combat/free-enemy phase requires empty hands and a free opponent, not own aliveness; own-respawn exposure is not separately serialized.',
                     'Window/carry/item rates are opportunity descriptive metrics, not counterfactual effects.'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('namespace',type=Path)
    args = parser.parse_args()
    path = args.namespace.resolve()
    report = build_report(path)
    output = path/'ACCEPTANCE.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(output)
    print(json.dumps(dict(games=report['games'],recomputed=True,output=output.relative_to(ROOT).as_posix(),
                          sha256=sha(output)),indent=2))


if __name__ == '__main__':
    main()
