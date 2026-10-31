import hashlib

import numpy as np
import pytest

from scripts import passive_evaluation_metrics as metrics
from scripts import passive_evaluation_report as report


def test_pairing_keeps_game_units_and_exact_zero_difference():
    result = report.paired_difference([1,0,1],[1,0,1])
    assert result['mean_delta'] == 0
    assert result['game_pair_bootstrap_ci95'] == [0,0]
    assert result['both_positive'] == 2
    assert result['both_nonpositive'] == 1
    assert result['training_seed_variance_estimated'] is False
    assert report.paired_difference([],[])['mean_delta'] is None
    with pytest.raises(ValueError):
        report.paired_difference([1],[1,2])


def test_offline_trace_recompute_checks_rewards_and_hashes(tmp_path):
    tick_names = ['tick','phase_code','winner','own_carried_before',
                  'own_carried_after','legacy_reward','minimal_reward']
    raw = np.array([[0,0,-2,-1,-1,0,0],[1,0,-1,-1,-1,0,0]],np.float32)[:,None,:]
    names = ['transport_capture','settled_kill','global_safety',
             'control_entry_reward','control_residual_reward','route_shaped_reward',
             'combined_total']
    p = tmp_path/'trace.npz'
    np.savez(p,telemetry=raw,telemetry_names=tick_names,component_names=names,
             episode_indices=[0],active=np.ones((2,1),bool),
             legacy_reward=np.zeros((2,1)),minimal_reward=np.zeros((2,1)),
             legacy_parts=np.zeros((2,1,7)),minimal_parts=np.zeros((2,1,7)))
    rows = []
    for raw_row in raw[:,0]:
        row = dict(zip(tick_names,map(float,raw_row)))
        row['phase'] = metrics.PHASES[int(row.pop('phase_code'))]
        rows.append(row)
    original = metrics.summarize_game('g',rows,termination='game_timeout',winner=-1,
                                      model='p',action_mode='greedy')
    ep = dict(episode_index=0,termination='game_timeout',passive_metrics=original,
              provenance=dict(trace='trace.npz',trace_sha256=report.sha(p),telemetry_column=0))
    manifest = dict(gamma=.995,reward_scoring=dict(
        exclusive_legacy_terms=names[:-1],exclusive_minimal_terms=[],
        per_transition_decomposition_tolerance=1e-4))
    recomputed, errors = report.recompute_episode(ep,manifest,root=tmp_path)
    assert recomputed == original
    assert errors['legacy']['max_abs_exclusive_to_actual_reward'] == 0
    ep['provenance']['trace_sha256'] = hashlib.sha256(b'wrong').hexdigest()
    with pytest.raises(ValueError,match='trace hash mismatch'):
        report.recompute_episode(ep,manifest,root=tmp_path)


def test_missing_action_probabilities_have_no_fake_zero_measurements():
    game = metrics.summarize_game('g',[dict(tick=0,phase='carrying')],
        termination='game_timeout',winner=-1,model='p',action_mode='greedy')
    result = metrics.summarize_cohort([game])
    probability = result['phases']['carrying']['action_probability_means']['ability_prob_2']
    assert probability == dict(value=None,observed_ticks=0,total_ticks=1)


def test_paired_event_rates_pool_denominators_and_cluster_on_source_games():
    result = report.paired_ratio_difference([1, 0], [1, 9], [1, 0], [1, 1])
    assert result['parent_rate'] == .1
    assert result['candidate_rate'] == .5
    assert result['mean_delta'] == .4
    assert result['independent_unit'] == 'paired source game, not tick/event'
    assert result['bootstrap_defined_replicates'] == 10000
    assert result['training_seed_variance_estimated'] is False


def test_zero_event_denominators_do_not_become_bootstrap_zero_rates():
    result = report.paired_ratio_difference([0, 0], [0, 0], [1, 0], [1, 1])
    assert result['parent_rate'] is None
    assert result['candidate_rate'] == .5
    assert result['mean_delta'] is None
    assert result['game_pair_bootstrap_ci95'] is None
    assert result['bootstrap_defined_replicates'] == 0
    with pytest.raises(ValueError):
        report.paired_ratio_difference([2], [1], [0], [1])


def test_native_terminal_and_residual_reward_tail_are_separate_scopes():
    rows = [dict(winner=-2, own_death=1), dict(winner=0, own_capture=1),
            dict(winner=-2, own_death=1)]
    result = report.transition_scope(rows)
    assert result['native_gameplay_transitions'] == 2
    assert result['settlement_tail_transitions'] == 1
    assert result['tail_own_deaths'] == 1
    assert result['tail_deliveries'] == 0
    assert report.transition_scope([dict(winner=-2)])['native_terminal_observed'] is False
