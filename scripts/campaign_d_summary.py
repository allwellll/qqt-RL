"""Read immutable D endpoints/curves and paired parents, without training."""
from pathlib import Path
import json
from scripts.population_run_state import atomic_json,sha

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'runs/expert_reward_20261025/D_ablation_v1'
OUT=ROOT/'runs/authorized_campaign_20261026/D_summary'

def read(p):return json.loads(Path(p).read_text())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    endpoints=[];curves=[];parents=[]
    sources=read(D/'SOURCE_MANIFEST.json')['sources']
    for seed in (2026102851,2026102852,2026102853):
        parent=ROOT/sources[str(seed)]['checkpoint']
        pr=read(parent/'retention/COMPLETE.json')
        parents.append(dict(seed=seed,source=sources[str(seed)],evaluation=pr,
            protocol_sha256=pr.get('protocol_sha256'),evidence=str((parent/'retention/COMPLETE.json').relative_to(ROOT))))
        for arm in ('legacy','minimal'):
            run=D/'formal'/f'{arm}_s{seed}';cp=run/'u000320';done=read(run/'COMPLETE.json')
            assert done['new_update']==320 and done['environment_steps']==5242880
            assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
            evaluation=read(D/'eval_done'/f'{arm}_s{seed}_u000320.json')
            assert evaluation['passed'] and evaluation['result']['policy_sha256']==done['policy_sha256']
            final=evaluation['result'];holdout=read(cp/'retention/HOLDOUT_COMPLETE.json')
            assert holdout['passed'] and holdout['policy_sha256']==done['policy_sha256']
            endpoints.append(dict(arm=arm,seed=seed,checkpoint=done['saved_checkpoint'],
                state_sha256=done['state_sha256'],policy_sha256=done['policy_sha256'],
                natural_delivery=final['natural_delivery_episodes'],natural_games=final['natural_games'],
                holdout_delivery=holdout['delivery_episodes'],holdout_games=holdout['games'],
                holdout_death_events=holdout['death_events'],tactical=final['tactical'],hunter=final['hunter']))
            for path in sorted((D/'eval_done').glob(f'{arm}_s{seed}_*.json')):
                x=read(path);r=x['result'];curves.append(dict(arm=arm,seed=seed,new_update=x['request']['new_update'],
                    level=r['level'],delivery=r.get('natural_delivery_episodes',r.get('delivery_episodes')),
                    games=r.get('natural_games',r.get('games')),evidence=str(path.relative_to(ROOT))))
    pairs=[]
    for seed in (2026102851,2026102852,2026102853):
        a,b=[next(x for x in endpoints if x['seed']==seed and x['arm']==arm) for arm in ('legacy','minimal')]
        pairs.append(dict(seed=seed,dev_delivery_delta=b['natural_delivery']-a['natural_delivery'],
            holdout_delivery_delta=b['holdout_delivery']-a['holdout_delivery'],
            holdout_death_event_delta=b['holdout_death_events']-a['holdout_death_events']))
    aggregate={arm:dict(dev_deliveries=[r['natural_delivery'] for r in endpoints if r['arm']==arm],
        holdout_deliveries=[r['holdout_delivery'] for r in endpoints if r['arm']==arm],
        worst_dev=min(r['natural_delivery'] for r in endpoints if r['arm']==arm),
        worst_holdout=min(r['holdout_delivery'] for r in endpoints if r['arm']==arm)) for arm in ('legacy','minimal')}
    atomic_json(OUT/'SUMMARY.json',dict(endpoints=endpoints,paired_seed_deltas=pairs,aggregate=aggregate,
        all_curves=curves,parents=parents,parent_comparison_caveat='Parent protocol hashes must match before paired inference.',
        confounds=['minimal capture native +2 vs legacy transport +24',
        'legacy pending capture/terminal clawback timing differs from immediate minimal events',
        'kill/death semantics and auxiliary channels are a package intervention'],
        no_peak_selection=True,no_cross_reward_scale_ranking=True,independent_training_pairs=3))
    lines=['# D六run固定终点汇总（20261026）','',
        '六run各320更新/5,242,880新增on-policy步，全部state/policy hash与最终评估、独立holdout验收通过。保留全部曲线与最差seed，不择峰。',
        '', '| arm | seed | dev交付/64 | holdout交付/64 | holdout死亡事件 |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['arm']} | {r['seed']} | {r['natural_delivery']} | {r['holdout_delivery']} | {r['holdout_death_events']} |" for r in endpoints]
    lines += ['', 'D minimal不是仅删除塑形：交付+24→+2，同时从transport pending/clawback结算改为原生事件即时计酬；还含其他奖励包差异，不能归因单个辅助项。',
        '', '父点原始评估、Tactical与Hunter三难度、全曲线及配对差值见 runs/authorized_campaign_20261026/D_summary/SUMMARY.json。父点协议若不一致，不强做显著性结论；新300局矩阵独立执行。低死亡若伴随低行动/低交付，不称安全改善。']
    (ROOT/'reports/D_paired_final_20261026.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'COMPLETE.json',dict(passed=True,runs=6,summary_sha256=sha(OUT/'SUMMARY.json')))
    print(json.dumps(dict(aggregate=aggregate,pairs=pairs)))

if __name__=='__main__':main()
