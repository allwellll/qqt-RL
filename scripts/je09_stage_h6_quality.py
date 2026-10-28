"""Machine-readable H6 DAgger quality and fixed-gold audit report."""
import argparse
import json
import os
import time
from pathlib import Path

from scripts.je09_stage_h6_fragments import Fragment, current_fragment
from scripts.population_run_state import atomic_json

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = Path(os.environ.get('JE09_NAMESPACE',
    'runs/je09_bun_20261016/stage_h6_full_trajectory_fragment_dagger_v28'))
SMOKE_NAME = os.environ.get('JE09_SMOKE_NAME', 'production_smoke_stage_h6')
ARM = 'bun392_bridge_pfsp8'

GOLD = [
    ('free_enemy_no_rush', Fragment.COMBAT_CONTROL, dict()),
    ('slow_glue_negative', Fragment.COMBAT_CONTROL, dict(glue_only=True)),
    ('banana_negative', Fragment.COMBAT_CONTROL, dict(banana_only=True)),
    ('attributed_sugar_window_loose', Fragment.A_LOOSE_RECOVERY, dict(attributed_window=True, loose_attributed=True)),
    ('attributed_sugar_window_stock', Fragment.B_CAGE_STOCK, dict(attributed_window=True, stock_in_enemy_cage=True)),
    ('death_window_stock', Fragment.B_CAGE_STOCK, dict(enemy_alive=False, stock_in_enemy_cage=True)),
    ('carry_return', Fragment.CARRY_RETURN, dict(carrying_enemy=True)),
    ('cage_delivery_edge', Fragment.CAGE_DELIVERY, dict(carrying_enemy=True, at_own_cage=True)),
    ('window_not_open', Fragment.COMBAT_CONTROL, dict(enemy_alive=True)),
    ('window_open_object_not_seen', Fragment.AMBIGUOUS, dict(attributed_window=True)),
    ('unreachable', Fragment.AMBIGUOUS, dict(reachable=False)),
    ('unclear_drop_provenance', Fragment.AMBIGUOUS, dict(provenance_clear=False)),
    ('terminal_state', Fragment.AMBIGUOUS, dict(terminal=True)),
    ('doomed_no_safe_action', Fragment.AMBIGUOUS, dict(legal_safe_actions=False)),
]


def gold_audit():
    common = dict(terminal=False, learner_alive=True, learner_trapped=False,
        teammate_harmed=False, enemy_alive=True, enemy_trapped=False,
        attributed_window=False, carrying_enemy=False, loose_attributed=False,
        stock_in_enemy_cage=False, at_own_cage=False, reachable=True,
        legal_safe_actions=True, provenance_clear=True)
    rows = []; classes = sorted({int(expected) for _, expected, _ in GOLD})
    matrix = {str(t): {str(p): 0 for p in classes} for t in classes}
    for name, expected, changes in GOLD:
        predicted = current_fragment(**(common | changes))
        matrix[str(int(expected))][str(int(predicted))] += 1
        rows.append(dict(case=name, expected=expected.name, predicted=predicted.name,
                         correct=predicted == expected, label='fixed semantic gold; not model-output accuracy'))
    tp = {}; precision = {}; recall = {}
    for c in classes:
        true_pos = matrix[str(c)][str(c)]
        pred_pos = sum(matrix[str(t)][str(c)] for t in classes)
        actual_pos = sum(matrix[str(c)].values())
        tp[str(c)] = true_pos
        precision[str(c)] = true_pos / pred_pos if pred_pos else None
        recall[str(c)] = true_pos / actual_pos if actual_pos else None
    return dict(total=len(rows), correct=sum(r['correct'] for r in rows),
        localization_precision_proxy=sum(r['correct'] for r in rows) / len(rows),
        localization_coverage=sum(r['predicted'] != 'AMBIGUOUS' for r in rows) / len(rows),
        confusion_matrix=matrix, per_class_precision=precision, per_class_recall=recall,
        rows=rows)


def build_report():
    events = []
    for run_name in [SMOKE_NAME, 'formal']:
        run_root = ROOT / NAMESPACE / run_name
        if run_name == 'formal': run_root = ROOT / NAMESPACE
        event_file = run_root / 'formal' / ARM / 'events.jsonl'
        if event_file.exists():
            for line in event_file.read_text().splitlines():
                if line.strip():
                    row = json.loads(line); row['_run'] = run_name; events.append(row)
    dagger_rows = [row.get('dagger', {}) for row in events]
    sum_key = lambda key: sum(float(row.get(key, 0)) for row in dagger_rows)
    visited = sum(float(row.get('visited', 0)) for row in dagger_rows)
    names = ['a_loose', 'b_stock', 'carry_return', 'cage_delivery']
    def event_value(event, key):
        row = event.get('dagger', {})
        if key == 'teacher_eligible': return float(row.get('eligible', 0))
        if key == 'teacher_proposed': return sum(float(row.get(f'{name}_proposed', 0)) for name in names)
        if key == 'teacher_accepted': return float(row.get('query', 0))
        return float(row.get(key, 0))
    eligible = sum(event_value(row, 'teacher_eligible') for row in events)
    proposed = sum(event_value(row, 'teacher_proposed') for row in events)
    accepted = sum(event_value(row, 'teacher_accepted') for row in events)
    gradient = sum(float(row.get('dagger', {}).get('query', 0)) for row in events)
    fragments = {}
    for name in names:
        candidates = sum_key(f'{name}_candidate_ticks')
        proposed_n = sum_key(f'{name}_proposed')
        accepted_n = query_n = sum(float(row.get('dagger', {}).get(f'{name}_query', 0)) for row in events)
        fragments[name] = dict(
            candidate_ticks=candidates,
            candidate_episodes=sum_key(f'{name}_candidate_episodes'),
            proposed=proposed_n, accepted=accepted_n, queries=query_n,
            query_episodes=sum_key(f'{name}_query_episodes'),
            accepted_per_proposed=accepted_n / proposed_n if proposed_n else None,
            query_per_candidate=query_n / candidates if candidates else None)
    action_sizes = {str(i): sum_key(f'action_set_size_{i}') for i in range(1, 6)}
    legal_queries = sum_key('legal_query')
    opponent_roles = {}
    for role_id, role_name in [(0, 'native_strong_pfsp'), (1, 'weak_courier'), (2, 'weak_guard')]:
        rows = [row.get('dagger', {}).get('opponent_course', {}).get(str(role_id), {}) for row in events]
        keys = ['visited', 'eligible', 'query', 'candidate_a', 'candidate_b',
                'candidate_carry_return', 'candidate_cage_delivery', 'path_a', 'path_b',
                'carry_return', 'cage_delivery', 'opponent_native_custody_ticks']
        opponent_roles[role_name] = {key: sum(float(item.get(key, 0)) for item in rows) for key in keys}
    targeted_course = {}
    for target_id in range(5):
        entries = [row.get('dagger', {}).get('targeted_course', {}).get(str(target_id), {}) for row in events]
        targeted_course[str(target_id)] = dict(
            name=next((item.get('name') for item in entries if item.get('name')), None),
            visited=sum(float(item.get('visited', 0)) for item in entries),
            candidate=sum(float(item.get('candidate', 0)) for item in entries),
            query=sum(float(item.get('query', 0)) for item in entries),
            weak_courier=sum(float(item.get('weak_courier', 0)) for item in entries),
            weak_guard=sum(float(item.get('weak_guard', 0)) for item in entries),
            difficulty={str(level):sum(float(item.get('difficulty', {}).get(str(level), 0)) for item in entries)
                        for level in range(4)},
            native_transitions=[sum(float(item.get('native_transitions', [0]*8)[i]) for item in entries)
                                for i in range(8)])
    trainer_exit = ROOT / NAMESPACE / 'formal' / ARM / 'TRAINER_EXIT.json'
    report = dict(schema='je09.h6.dagger-quality/v1', generated_unix=time.time(),
        experiment_id=(json.loads((ROOT / NAMESPACE / 'ROUND_DECLARATION.json').read_text())['experiment_id']
            if (ROOT / NAMESPACE / 'ROUND_DECLARATION.json').exists() else 'JE09-STAGE-H6-FULL-TRAJECTORY-FRAGMENT-DAGGER-20261016-v28'),
        generation=max((int(e.get('generation', 0)) for e in events), default=0),
        status='complete' if trainer_exit.exists() else 'monitoring',
        funnel=dict(full_scene_ticks=sum_key('full_ticks'), full_episodes_completed=sum_key('full_episode_ends'),
            visited_ticks=visited, candidate_fragments=eligible, clearly_located=None,
            ambiguous_or_unreachable=None, teacher_proposed=proposed,
            strict_accepted=accepted, gradient_queries=gradient,
            location_coverage=None,
            teacher_acceptance=accepted / proposed if proposed else None,
            accepted_count_is_query_lower_bound=True,
            location_status='Runtime collector does not preserve pre-verifier reachable/provenance localization bits; omitted rather than inferred from verifier quota.',
            final_supervision_coverage=gradient / eligible if eligible else None,
            localization_precision_proxy='fixed gold only; see fixed_gold_audit',
            noneligible_gradient_pollution=sum_key('noneligible_label')),
        fragments=dict(per_fragment=fragments,
            combat_control=dict(dagger_queries=0, policy='audit only; existing PPO/native combat expert/history retention')),
        teacher_quality=dict(action_set_size_histogram=action_sizes,
            legal_action_rate=legal_queries / gradient if gradient else None,
            temporal_hazard_acceptance_lower_bound=accepted / proposed if proposed else None,
            rollout_survival_rate=None,
            task_progress_rate=None,
            old_drop_expert_consistency=None,
            old_return_expert_consistency=None,
            verifier_rejected=None,
            verifier_rejected_upper_bound=max(proposed - accepted, 0),
            rollout_survival_and_expert_consistency_status='not emitted by runtime collector; null means unmeasured, not pass',
            refusal_reason_breakdown=dict(
                strict_verifier_reject=None,
                strict_verifier_reject_upper_bound=max(proposed - accepted, 0),
                AMBIGUOUS=None, UNREACHABLE=None, DOOMED=None,
                UNCLEAR_PROVENANCE=None, FUTURE_DEPENDENT=None,
                status='reason-specific counters are not yet emitted; do not infer category from aggregate rejects')),
        runtime_scope=dict(runs=[row['_run'] for row in events],
            generations=sorted({int(row.get('generation', 0)) for row in events}),
            opponent_course=opponent_roles, targeted_course=targeted_course),
        fixed_gold_audit=gold_audit(),
        h5cr_baseline=dict(visited=67108864, eligible=1895745, eligible_fraction=0.028249,
            query=972034, query_of_eligible=0.512745, query_of_visited=0.014484,
            rejected_or_unlabeled_fraction_of_eligible=0.487255,
            query_distribution=dict(A=53986, A_fraction=0.055539, B=396771,
                B_fraction=0.408186, RETURN=521277, RETURN_fraction=0.536274),
            noneligible_pollution=0,
            interpretation='historical event-gate coverage only; not gold-label precision'))
    return report


def main():
    report = build_report()
    out = ROOT / NAMESPACE / 'quality'
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'DAGGER_QUALITY.json', report)
    lines = ['# H6 DAgger Quality', '',
        f"Generation: {report['generation']}",
        f"Ticks visited: {report['funnel']['visited_ticks']:.0f}",
        f"Eligible/proposed/accepted/gradient: {report['funnel']['candidate_fragments']:.0f}/"
        f"{report['funnel']['teacher_proposed']:.0f}/{report['funnel']['strict_accepted']:.0f}/"
        f"{report['funnel']['gradient_queries']:.0f}",
        f"Fixed semantic-gold localization precision proxy: {report['fixed_gold_audit']['localization_precision_proxy']:.3f}; "
        f"coverage: {report['fixed_gold_audit']['localization_coverage']:.3f}",
        'COMBAT_CONTROL has zero H6 DAgger queries; it remains on PPO/safety-expert/history retention.',
        'Weak opponent targets are stratified independently across A, B, return and delivery; see runtime_scope.targeted_course.',
        'Runtime survival, progress, old-expert consistency, and reason-specific refusals are null until instrumented.',
        '', 'The fixed audit is a semantic fixture set, not model-output precision. H5CR values are event-gate coverage, not gold accuracy.', '']
    (out / 'DAGGER_QUALITY.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['once', 'watch'], nargs='?', default='once')
    args = parser.parse_args()
    if args.command == 'once': main()
    else:
        seen = None
        while True:
            report = build_report()
            token = (report['generation'], report['funnel']['visited_ticks'])
            if token != seen:
                main(); seen = token
            time.sleep(10)
