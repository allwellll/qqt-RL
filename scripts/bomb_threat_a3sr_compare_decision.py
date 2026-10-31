"""Read-only experiment decision report; never changes BEST or a web model."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round15 import read
BASE=Path('runs/bomb_threat_v1');OUT=BASE/'fresh256_a3sr_vs_u800_20261030'
REPORT=Path('reports/bomb_threat_v1_fresh256_a3sr_20261030.md')
PROMOTION=BASE/'iter2/a3sr/promotion_20261030'


def compare(values):
    result={}
    for role in ('a3sr_u640','a3sr_u1280'):
        tiers={};missing=[]
        for tier in ('hard','normal','easy'):
            candidate=values[tier][role];baseline=values[tier]['u800']
            if candidate['games']!=256 or baseline['games']!=256:
                missing.append(tier+'_256_COMPLETE');continue
            c,b=candidate['by_seat']['P1'],baseline['by_seat']['P1']
            assert c['games']==b['games']==128
            checks=dict(win_minus_loss=c['win_minus_loss']>=b['win_minus_loss'],
                causal_kills=c['causal_kills']>=b['causal_kills'],deaths=c['deaths']<=b['deaths'],
                self_deaths=c['self_deaths']<=b['self_deaths'],unilateral_zero=c['zero_bomb_games']<=b['zero_bomb_games'],
                draw_fraction=c['draw_fraction']<=b['draw_fraction'],
                same_chain_window_delivery=c['funnel']['window_stock_delivery']>=b['funnel']['window_stock_delivery'])
            tiers[tier]=dict(candidate_WL=c['win_minus_loss'],baseline_WL=b['win_minus_loss'],
                difference=c['win_minus_loss']-b['win_minus_loss'],absolute_at_least10=c['win_minus_loss']>=10,
                difference_at_least10=c['win_minus_loss']-b['win_minus_loss']>=10,
                amended_per_tier_condition=c['win_minus_loss']>=b['win_minus_loss']-5,
                descriptive_dominance_checks=checks,descriptive_dominance=all(checks.values()))
        total=sum(r['candidate_WL'] for r in tiers.values()) if not missing else None
        old=sum(r['baseline_WL'] for r in tiers.values()) if not missing else None
        passed=not missing and total>old and all(r['amended_per_tier_condition'] for r in tiers.values())
        result[role]=dict(status='MISSING' if missing else 'REPORT_ELIGIBLE' if passed else 'NOT_ELIGIBLE',
            missing=missing,tiers=tiers,P1_three_tier_sum=total,u800_P1_three_tier_sum=old,
            criterion='20261030 promotion decision: sum>u800 and eachtier>=u800-5; report orchestration only',
            observable_dominance=None if missing else all(r['descriptive_dominance'] for r in tiers.values()),
            no_best_change=True,no_publication=True)
    return result


def refresh():
    source=read(OUT/'RESULTS.json',{})
    if not source:return
    results=compare(source['metrics'])
    report=dict(status='COMPLETE' if source['status']=='COMPLETE' else 'RUNNING',
        source_sha256=sha(OUT/'RESULTS.json'),decisions=results,
        decision='.hermes/DECISION_promote_a3sr_20261030.md',decision_sha256=sha(Path('.hermes/DECISION_promote_a3sr_20261030.md')),
        limitations=['descriptive matched fixed-seed evidence, not significance or universal noninferiority',
                    'same old64 geometry pool, not256 unique maps; JS Hunter is not a human'],unix=time.time())
    atomic_json(OUT/'WEB_CANDIDATE_REVIEW.json',report)
    if REPORT.exists():
        text=REPORT.read_text().split('\n## 最新推广决定下的网页候选口径')[0]
        text+='\n## 最新推广决定下的网页候选口径\n\n'
        text+='`DECISION_promote_a3sr_20261030`覆盖原网页候选规则：P1三档胜负差合计严格高于u800，且每档不低于u800减5；只报编排方，不自动替换网页或BEST。原“全面不差”和+10仍分别报告描述性结果，不再作为当前推广/迭代门禁。\n\n'
        for role,d in results.items():
            text+=f"- {role}: **{d['status']}**；P1三档合计 {d['P1_three_tier_sum']} vsu800 {d['u800_P1_three_tier_sum']}；完整描述性逐项不差={d['observable_dominance']}；缺数{d['missing']}。\n"
            for tier,r in d['tiers'].items():
                text+=f"  - {tier}: P1 W−L {r['candidate_WL']:+d} vsu800 {r['baseline_WL']:+d}，差{r['difference']:+d}，绝对≥+10={r['absolute_at_least10']}，相对差≥+10={r['difference_at_least10']}，新单档界限通过={r['amended_per_tier_condition']}；原描述性不差缺口={[k for k,v in r['descriptive_dominance_checks'].items() if not v]}。\n"
        text+='\n各候选新样本齐全后才作完整判定；没有从单档、单座位或半程择优。原始逐局配对表与所有输入SHA可在RESULTS.json复算。\n'
        REPORT.write_text(text)
    if source['status']=='COMPLETE' and not (OUT/'FINAL_NOTIFICATION.json').exists():
        atomic_json(OUT/'FINAL_NOTIFICATION.json',report)
        from scripts.iter_pbt_owner import notify
        notify('A3SR_U800_FINAL_DECISION',f"A3SR u640/u1280同u800 seed fresh256完成：{ {k:d['status'] for k,d in results.items()} }；P1分档与运包失败证据报告齐全，仅报编排方，未替换网页。",report=str(REPORT))
    return report

if __name__=='__main__':refresh()
