"""Read-only H6 event summarizer with separate proposed, accepted and query counts."""
import json
import os
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = Path(os.environ["JE09_NAMESPACE"])
SMOKE = os.environ["JE09_SMOKE_NAME"]
ARM = "bun392_bridge_pfsp8"
FRAGMENTS = ["a_loose", "b_stock", "carry_return", "cage_delivery"]
TARGETS = ["none", *FRAGMENTS]


def read_events():
    rows = []
    for label, path in [
        ("smoke", ROOT / NAMESPACE / SMOKE / "formal" / ARM / "events.jsonl"),
        ("formal", ROOT / NAMESPACE / "formal" / ARM / "events.jsonl"),
    ]:
        if path.exists():
            rows.extend(dict(json.loads(line), _source=label)
                         for line in path.read_text().splitlines() if line.strip())
    return rows


def total(rows, key):
    return sum(float(row.get("dagger", {}).get(key, 0)) for row in rows)


def rate(a, b):
    return a / b if b else None


def build():
    rows = read_events()
    visited = total(rows, "visited")
    full_ticks = total(rows, "full_ticks")
    eligible = total(rows, "teacher_eligible")
    proposed = total(rows, "teacher_proposed")
    accepted = total(rows, "teacher_accepted")
    queries = total(rows, "gradient_query")
    fragments = {}
    for name in FRAGMENTS:
        candidate = total(rows, f"{name}_candidate_ticks")
        prop = total(rows, f"{name}_proposed")
        acc = total(rows, f"{name}_accepted")
        query = total(rows, f"{name}_query")
        fragments[name] = dict(candidate_ticks=candidate,
            candidate_episodes=total(rows, f"{name}_candidate_episodes"),
            proposed=prop, accepted=acc, gradient_query=query,
            query_episodes=total(rows, f"{name}_query_episodes"),
            acceptance=rate(acc, prop), query_per_candidate=rate(query, candidate))
    roles = {}
    for role_id, name in [(0, "strong_or_pfsp"), (1, "weak_courier"), (2, "weak_guard")]:
        groups = [r.get("dagger", {}).get("opponent_course", {}).get(str(role_id), {}) for r in rows]
        keys = ["visited", "eligible", "query", "candidate_a", "candidate_b",
                "candidate_carry_return", "candidate_cage_delivery", "path_a", "path_b",
                "carry_return", "cage_delivery", "opponent_native_custody_ticks"]
        roles[name] = {key: sum(float(g.get(key, 0)) for g in groups) for key in keys}
    targets = {}
    for target_id, name in enumerate(TARGETS):
        groups = [r.get("dagger", {}).get("targeted_course", {}).get(str(target_id), {}) for r in rows]
        targets[name] = dict(
            visited=sum(float(g.get("visited", 0)) for g in groups),
            candidate=sum(float(g.get("candidate", 0)) for g in groups),
            query=sum(float(g.get("query", 0)) for g in groups),
            weak_courier=sum(float(g.get("weak_courier", 0)) for g in groups),
            weak_guard=sum(float(g.get("weak_guard", 0)) for g in groups),
            native_transitions=[sum(float(g.get("native_transitions", [0] * 8)[i]) for g in groups)
                                for i in range(8)])
    action_sets = {str(i): total(rows, f"action_set_size_{i}") for i in range(1, 6)}
    noneligible = total(rows, "noneligible_label")
    return dict(schema="je09.h6.dagger-quality-live/v1", generated_unix=time.time(),
        experiment_id=json.loads((ROOT / NAMESPACE / "ROUND_DECLARATION.json").read_text())["experiment_id"],
        generations=sorted({int(r.get("generation", 0)) for r in rows}),
        source_runs=sorted({r["_source"] for r in rows}),
        funnel=dict(visited_ticks=visited, full_scene_ticks=full_ticks,
            full_episode_ends=total(rows, "full_episode_ends"), candidates=eligible,
            explicit_localization=None, ambiguous_or_rejected_reasons=None,
            proposed=proposed, strict_accepted=accepted, gradient_query=queries,
            location_coverage=None, location_precision_proxy=None,
            teacher_acceptance=rate(accepted, proposed),
            supervision_coverage=rate(queries, eligible),
            noneligible_gradient_pollution=noneligible,
            missing_localization_fields="Collector does not emit causal localization/rejection reason bits; null is unmeasured."),
        fragments=fragments,
        combat_control=dict(teacher_queries=total(rows, "combat_control_dagger_query"),
            policy="Audit-only; battle PPO, native safety expert and retention."),
        teacher_quality=dict(action_set_size_histogram=action_sets,
            legal_query_rate=rate(total(rows, "legal_query"), queries),
            temporal_hazard_acceptance=None, counterfactual_survival=None,
            task_progress=None, drop_expert_consistency=None,
            return_expert_consistency=None,
            reason="One-step labels are filtered through native survival and transition checks; rollout metrics are not emitted."),
        opponent_roles=roles, targeted_weak_course=targets,
        fixed_gold_audit=dict(scope="fixed semantic fixtures, not learner trajectory or model recognition accuracy",
            cases=14, correct=14, explicit=9,
            precision_proxy=1.0, coverage=9/14,
            per_case_path="DAGGER_QUALITY.json.fixed_gold_audit.rows"),
        h5cr_baseline=dict(visited=67108864, eligible=1895745,
            eligible_fraction=0.028249, query=972034, query_of_eligible=0.512745,
            query_of_visited=0.014484, unlabeled_fraction=0.487255,
            A=53986, B=396771, RETURN=521277, noneligible_pollution=0,
            interpretation="Historical event-gate coverage, not gold-label precision."))


def main():
    report = build()
    root = ROOT / NAMESPACE / "quality"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "DAGGER_QUALITY_LIVE.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# H6 DAgger Quality Live", "",
        f"Generations: {report['generations']}",
        f"Visited/full ticks: {report['funnel']['visited_ticks']:.0f}/{report['funnel']['full_scene_ticks']:.0f}",
        f"Candidate/proposed/accepted/query: {report['funnel']['candidates']:.0f}/"
        f"{report['funnel']['proposed']:.0f}/{report['funnel']['strict_accepted']:.0f}/"
        f"{report['funnel']['gradient_query']:.0f}",
        f"Acceptance/query coverage: {report['funnel']['teacher_acceptance']}/{report['funnel']['supervision_coverage']}",
        f"Fixed semantic gold proxy: {report['fixed_gold_audit']['correct']}/14; coverage 9/14. Not model accuracy.",
        "COMBAT_CONTROL DAgger queries: 0; battle remains PPO/safety expert/history.",
        "Uninstrumented localization reasons, rollout survival/progress and expert agreement are null, not passes.", ""]
    (root / "DAGGER_QUALITY_LIVE.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    if not args.watch:
        main()
    else:
        last = None
        while True:
            rows = read_events()
            marker = (len(rows), rows[-1].get("generation") if rows else 0)
            if marker != last:
                main()
                last = marker
            time.sleep(max(args.interval, 1))
