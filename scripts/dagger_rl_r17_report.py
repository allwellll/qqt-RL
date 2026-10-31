"""Consume all new fixed results against reused, paired, immutable controls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')
import pickle
import json
import subprocess
import sys
import numpy as np
import jax
from scripts.dagger_rl_r17_run import B, R, PRIOR, read, notify
from scripts.population_run_state import atomic_json, sha
from scripts.terminal_checkpoint_restore import digest
from scripts.combat_item_report import natural, paired_uncertainty


def supplements(p):
    """Read all fixed traces, actual rewards/GAE and clustered Bank arrays."""
    stages, traces, bank_arrays = {}, [], {}
    for seed in p['paired_seeds']:
        base=B/'formal'/f'sequential_anchor_s{seed}'
        rows=[json.loads(l) for l in (base/'updates.jsonl').read_text().splitlines()]
        stages[base.name]={}
        for mode in ['dagger','ppo']:
            selected=[r for r in rows if r['mode']==mode]
            stages[base.name][mode]=dict(
                reward_components_sum=np.asarray([r['reward_components'] for r in selected]).sum(0).tolist(),
                actual_training_GAE_mean=[r['GAE_mean'] for r in selected],
                actual_training_GAE_variance=[r['GAE_variance'] for r in selected],
                loss=[r['loss'] for r in selected],
                goal_queries=[read(base/f'u{r["update"]:03d}/COMMITTED.json')['metrics']['goal_guard_telemetry'] for r in selected],
                scope='queries are ticks, not retained replay loss dose or independent opportunities')
    paths=[(B/'formal/no_update/u000','dev'),(B/'formal/no_update/u000','holdout')]
    paths += [(B/'formal'/f'sequential_anchor_s{s}'/f'u{u:03d}',split)
              for s in p['paired_seeds'] for u,split in [(16,'dev'),(32,'dev'),(32,'holdout')]]
    for cp,split in paths:
        folder=cp/('holdout/retention' if split=='holdout' else 'retention')
        d=read(cp/'retention'/('natural_holdout64.json' if split=='holdout' else 'natural_full64.json'))
        assert d['model_json_sha256']==sha(folder/'model/model.json')
        episodes=d['results']['natural_full2400']['episodes']
        seed=p['evaluation']['natural_holdout_seed' if split=='holdout' else 'natural_dev_seed']
        assert sorted(e['seed'] for e in episodes)==list(range(seed,seed+64))
        files=list(folder.glob('fixed-*.trace.json'));assert len(files)==1
        t=read(files[0]);first=t['first_deviation'];item=dict(checkpoint=str(cp.relative_to(B)),split=split,
            first_geometric_disagreement=first,trace_sha256=sha(files[0]),
            episodes_entered_return=sum(e.get('stage_funnel',{}).get('C_RETURN',{}).get('target_ticks',0)>0 for e in episodes))
        if first is not None:
            ix=[i for i,x in enumerate(t['diagnostic_ticks']) if x['state_before_sha256']==first['state_hash']]
            assert len(ix)==1
            i=ix[0];tick=t['diagnostic_ticks'][i]
            item.update(approximate_reward_components=tick['approximate_reward_components'],
                zero_value_discounted_shaping=tick['approximate_reward_gae'],
                next_zero_value_discounted_shaping=t['diagnostic_ticks'][i+1]['approximate_reward_gae'] if i+1<len(t['diagnostic_ticks']) else None)
        traces.append(item)
    subprocess.run(['node','scripts/dagger_rl_direction_trace_audit.js',str(B.relative_to(R))],cwd=R,check=True)
    native=read(B/'FIXED_DEV_DIRECTION_AUDIT_v2.json');assert native['passed'] and native['completed']==7
    for label in ['native','bankP_backward','bankP_wait']:
        load=lambda cp:pickle.load((cp/f'{label}_holdout.pkl').open('rb'))
        parent=load(B/'formal/no_update/u000')
        arms={name:[load(root/'formal'/f'{arm}_s{s}'/'u032') for s in p['paired_seeds']]
              for name,root,arm in [('kind_guard',B,'sequential_anchor'),('legacy_seq',PRIOR,'sequential_anchor'),('PPO',PRIOR,'ppo_anchor')]}
        metrics={}
        for metric in ['direct_custody_delivery','delivery','death','drop']:
            means={name:[np.asarray(x[metric]).reshape(32,8).mean(-1) for x in a] for name,a in arms.items()}
            ref=np.asarray(parent[metric]).reshape(32,8).mean(-1)
            metrics[metric]={name:paired_uncertainty(np.stack([a-b for a,b in zip(means['kind_guard'],other)]))
                             for name,other in [('vs_legacy_sequential',means['legacy_seq']),('vs_PPO',means['PPO']),('vs_parent',[ref]*3)]}
        bank_arrays[label]=metrics
    return dict(actual_stage_rewards_GAE=stages,fixed_trace_links=traces,
        native_direction_audit_sha256=sha(B/'FIXED_DEV_DIRECTION_AUDIT_v2.json'),
        source_clustered_Bank_contrasts=bank_arrays,environment_steps_adopted=0,optimizer_steps=0,
        limits='JS zero-Value shaping is not actual GAE/Q. Native one-step contrasts are not future '
               'Q or safety. Bank reconstruction is not historical optimizer advantage. '
               'Prior-used map806/P0/allfar32 source games x8 continuations, not cross-map confirmation.')


def run():
    for name in ['TRAINING_COMPLETE.json', 'CPU_EVALUATION_COMPLETE.json',
                 'NATIVE_BANK_COMPLETE.json', 'FIXED_FIT_COMPLETE.json']:
        assert read(B/name)['passed']
    p = read(B/'PROTOCOL.json'); source = pickle.load((B/'inputs/parent.pkl').open('rb'))
    parent, parent_binary = natural(PRIOR/'formal/no_update/u000', True)
    results, binaries, controls, atoms, ppo_binaries = [], [], [], [], []
    for seed in p['paired_seeds']:
        base = B/'formal'/f'sequential_anchor_s{seed}'
        control = PRIOR/'formal'/base.name
        rows = [json.loads(line) for line in (base/'updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows] == list(range(1, 33))
        assert all(r['finite'] and r['update_l2']['actor'] > 0 for r in rows)
        assert all((r['update_l2']['critic'] == 0) == (r['mode'] == 'dagger') for r in rows)
        assert all((r['dagger_coefficient'] == 0) == (r['mode'] == 'ppo') for r in rows)
        for u in p['checkpoints']:
            cp = base/f'u{u:03d}'; m = read(cp/'COMMITTED.json')
            assert sha(cp/'state.pkl') == m['state_sha256'] and sha(cp/'policy.pt') == m['policy_sha256']
            x = pickle.load((cp/'state.pkl').open('rb'))
            assert digest(pickle.load((cp/'policy.pt').open('rb'))) == digest(x['params']['actor'])
            assert int(x['adam'][1][0].count) == int(source['adam'][1][0].count)+u*32
            assert all(np.isfinite(a).all() for a in jax.tree.leaves((x['params'], x['adam'])))
            if u == 0:
                z = pickle.load((control/'u000/state.pkl').open('rb'))
                keys = ['params', 'adam', 'states', 'aux', 'keys', 'auxiliary']
                assert digest(tuple(x[k] for k in keys)) == digest(tuple(z[k] for k in keys))
            if u <= 16:
                assert digest(x['params']['critic']) == digest(source['params']['critic'])
                for name in ['mu', 'nu']:
                    assert digest(getattr(x['adam'][1][0], name)['critic']) == digest(getattr(source['adam'][1][0], name)['critic'])
            atoms.append(dict(checkpoint=str(cp.relative_to(B)), state_sha256=m['state_sha256'], policy_sha256=m['policy_sha256']))
        final = base/'u032'; dev, _ = natural(final); mid, _ = natural(base/'u016')
        holdout, binary = natural(final, True); old, old_binary = natural(control/'u032', True)
        ppo=PRIOR/'formal'/f'ppo_anchor_s{seed}'/'u032'
        ppo_holdout,ppo_binary=natural(ppo,True);ppo_binaries.append(ppo_binary)
        binaries.append(binary); controls.append(old_binary)
        fit = read(base/'FIXED_PRE_POST_SWITCH_FIT.json'); assert fit['passed']
        for u, h in fit['reference_state_sha256'].items():
            assert h == read(base/f'u{int(u):03d}/COMMITTED.json')['state_sha256']
        telemetry = {str(u): read(base/f'u{u:03d}/COMMITTED.json')['metrics']['goal_guard_telemetry'] for u in range(1, 33)}
        results.append(dict(seed=seed, dev=dev, midpoint=mid, tail_dev_delivery=(dev['episode_counts']['delivery']+mid['episode_counts']['delivery'])/2,
            holdout=holdout, matched_legacy_control=old, contextual_PPO_control=ppo_holdout,
            PPO_retention=read(ppo/'retention/COMPLETE.json'),
            legacy_sequential_retention=read(control/'u032/retention/COMPLETE.json'),
            retention=read(final/'retention/COMPLETE.json'),
            fixed_fit=fit, goal_guard_query_telemetry=telemetry,
            source_exposure=[dict(update=r['update'], mode=r['mode'], PPO=r['source_exposure'], auxiliary=r['auxiliary_source_exposure']) for r in rows],
            item_denominators=[dict(update=r['update'], mode=r['mode'], PPO=r['item_opportunity_denominators'], auxiliary=r['auxiliary_item_opportunity_denominators']) for r in rows],
            actual_opponents=dict(current=0, ring=0, fixed=524288, JAX_rule3=524288),
            stages=dict(dagger_optimizer_steps=512, fresh_PPO_environment_steps_adopted=131072, PPO_optimizer_steps=512),
            Bank=read(final/'native_holdout.json'), Bank_control=read(control/'u032/native_holdout.json'),
            BankP={k: read(final/f'bankP_{k}_holdout.json') for k in ['backward', 'wait']},
            curves={str(u): read(base/f'u{u:03d}/native_dev.json') for u in [1, 8, 16, 17, 24, 32]}))
    contrasts = {key: paired_uncertainty(np.stack([np.asarray(a[key],float)-np.asarray(c[key],float)
                  for a,c in zip(binaries,controls)])) for key in parent_binary}
    vs_parent = {key: paired_uncertainty(np.stack([np.asarray(a[key],float)-np.asarray(parent_binary[key],float)
                 for a in binaries])) for key in parent_binary}
    assert len(atoms) == 21
    supplement=supplements(p)
    atomic_json(B/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json',dict(passed=True,**supplement))
    atomic_json(B/'ATOMIC_ACCEPTANCE.json', dict(passed=True, declared_checkpoints=21, rows=atoms))
    output = dict(passed=True, results=results, paired_vs_reused_legacy_sequential=contrasts,
                  vs_untrained_parent=vs_parent, parent=parent, reused_R16_complete_comparison_sha256=sha(PRIOR/'FINAL_COMPARISON.json'),
                  contextual_vs_PPO={key:paired_uncertainty(np.stack([np.asarray(a[key],float)-np.asarray(c[key],float)
                      for a,c in zip(binaries,ppo_binaries)])) for key in parent_binary},
                  worst_holdout_delivery=min(x['holdout']['episode_counts']['delivery'] for x in results),
                  supplement_sha256=sha(B/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json'),
                  no_peak_selection=True, no_promotion=True,
                  limits='Three original paired training seeds reused, not independent confirmation. Same prior-used map806/P0/allfar Bank. '
                         'Own-label fit and query ticks are not independent capability. Original geometric direction disagreement remains a proxy; '
                         'one-step distance is not Q/GAE. Improvement against a degraded control alone is insufficient.')
    atomic_json(B/'FINAL_COMPARISON.json', output)
    lines = ['\nR17固定三配对终态（有效目标kind门控，控制复用R16，非独立新seed确认）：',
             '| seed | 新处理holdout交付/64 | 原顺序对照/64 | 新处理死亡对局 | dev尾部 |', '|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['seed']} | {r['holdout']['episode_counts']['delivery']} | {r['matched_legacy_control']['episode_counts']['delivery']} | {r['holdout']['episode_counts']['death']} | {r['tail_dev_delivery']} |")
    lines.append(f"\n父点holdout交付{parent['episode_counts']['delivery']}/64；保留、完整曲线、真实道具链与query门控计数见FINAL_COMPARISON，不能对退化control择峰宣称整体改善。旧暂停保留，无发布。")
    for name in ['reports/dagger_rl_iteration_20261025.md', 'docs/training_experiment_ledger.md']:
        with (R/name).open('a') as f:
            f.write('\n'.join(lines)+'\n')
    atomic_json(B/'ANALYSIS_COMPLETE.json', dict(passed=True, comparison_sha256=sha(B/'FINAL_COMPARISON.json')))
    notify('R17新处理三配对完整结果已冻结，按未训练父/匹配PPO/原顺序对照及最差点评估，不自动晋升。')


if __name__ == '__main__':
    run()
