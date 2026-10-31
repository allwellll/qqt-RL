"""Unique bounded CPU owner for preregistered native item-chain diagnosis."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import concurrent.futures
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.population_run_state import atomic_json, sha, lock

R = Path(__file__).resolve().parents[1]
PREVIOUS = R / 'runs/dagger_rl_20261024/r10_release_qualification'
B = R / 'runs/dagger_rl_20261024/r11_native_item_mechanism'


def read(path):
    return json.loads(Path(path).read_text())


def freeze():
    assert read(PREVIOUS / 'REPAIRED_CONCLUSION_COMPLETE.json')['passed']
    assert not read(PREVIOUS / 'EVIDENCE_MEASUREMENT_HOLD.json')['hold']
    decision = read(B / 'SCIENTIFIC_SELECTION.json')
    assert decision['execute_native_mechanism']
    assert decision['full_evidence_sha256'] == sha(PREVIOUS / 'COMPLETE_EVIDENCE_SYNTHESIS.json')
    if (B / 'PROTOCOL.json').exists(): return read(B / 'PROTOCOL.json')
    models = ['no_update/u000', 'control_s2026102581/u032', 'qualified_route_s2026102581/u032']
    selected = []
    for model in models:
        cp = PREVIOUS / 'formal' / model
        episodes = read(cp / 'retention/natural_full64.json')['results']['natural_full2400']['episodes']
        groups = {}
        for category, field in [('release','legal_releases'), ('inventory','inventory_ticks')]:
            groups[category] = [e['game'] for e in episodes
                               if e['item_metrics'].get('carry_banana',{}).get(field,0)>0][:4]
        for game in sorted(set(groups['release'] + groups['inventory'])):
            selected.append(dict(model=model, game=game, seed=episodes[game]['seed'],
                                 team=episodes[game]['team'],
                                 categories=[c for c in groups if game in groups[c]],
                                 expected_episode=episodes[game],
                                 policy_sha256=sha(cp / 'policy.pt'),
                                 model_json_sha256=sha(cp / 'retention/model/model.json')))
    sources = ['scripts/dagger_rl_native_mechanism.py','scripts/dagger_rl_native_bank_preload.js',
               'scripts/dagger_rl_js_bank.js','scripts/dagger_rl_native_chain.js',
               'scripts/dagger_rl_native_teacher.py','scripts/dagger_rl_oracle.py',
               'scripts/dagger_rl_route.js','scripts/dagger_rl_metrics_v2.js','web/sim.js']
    protocol = dict(id='DAGGER-RL-20261024-R11-NATIVE-ITEM-MECHANISM',
        scientific_question=read(PREVIOUS / 'NEXT_SCIENTIFIC_QUESTIONS.json')['if_no_credible_full_improvement'],
        previous_comparison_sha256=sha(PREVIOUS / 'FINAL_COMPARISON.json'),
        previous_full_evidence_sha256=sha(PREVIOUS / 'COMPLETE_EVIDENCE_SYNTHESIS.json'),
        source_selection='first<=4 source games per model/category in ascending game/tick; actual release or any carrier banana inventory; overlaps retained',
        selected_sources=selected, original_episodes_per_model=64,
        maximum_states=24, maximum_branch_steps=9216,
        maximum_source_replay_steps=485376, maximum_total_physics_steps=494592,
        parity_clone_step_budget=24, new_independent_source_episodes=0, optimizer_steps=0,
        adopted_updates=0, uses_holdout=False, old_continuous_resume=False,
        teacher='original JAX item forecast/eligibility with actual native masks; native observable route fallback explicitly not JAX temporal safety parity',
        source_seal={s:sha(R / s) for s in sources})
    atomic_json(B / 'PROTOCOL.json', protocol)
    return protocol


def source_job(case):
    key = case['model'].replace('/','_')
    target = B / 'sources' / key / f"game{case['game']:03d}"
    target.mkdir(parents=True, exist_ok=True)
    model = PREVIOUS / 'formal' / case['model'] / 'retention/model/model.json'
    assert sha(model) == case['model_json_sha256']
    output = target / 'RESULT.json'
    if not output.exists():
        env = dict(os.environ, NODE_OPTIONS='--require=./scripts/dagger_rl_native_bank_preload.js',
                   DAGGER_NATIVE_BANK_OUT=str(target.relative_to(R)),
                   DAGGER_NATIVE_BANK_CATEGORIES=','.join(case['categories']),
                   JE10_ROUTE_TASK='natural_full2400', JE10_ROUTE_GAMES='64',
                   JE10_ROUTE_START=str(case['game']), JE10_ROUTE_END=str(case['game']+1),
                   JE10_ROUTE_SEED_START='2026102480')
        with (target / 'source.log').open('a') as stream:
            subprocess.run(['node','scripts/dagger_rl_route.js',str(model.relative_to(R)),
                            str(output.relative_to(R)),str((PREVIOUS / 'cpu_eval_inputs/regions.json').relative_to(R))],
                           cwd=R, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = read(output)['results']['natural_full2400']
    assert result['episodes'] == [case['expected_episode']], 'natural source changed episode/action-derived outcomes'
    banks = []
    for category in case['categories']:
        bank = target / f'{category}.bank'
        assert bank.exists(), (case['model'], case['game'], category, 'expected actual causal state absent')
        banks.append(dict(path=str(bank.relative_to(R)),sha256=sha(bank),category=category,
                          model=str(model.relative_to(R)),source_model=case['model']))
    receipt = dict(passed=True, source_game=case['game'],source_seed=case['seed'],
                   full_episode_exact=True, source_physics_steps=result['ticks'],
                   maximum_clone_parity_steps=len(banks), banks=banks)
    atomic_json(target / 'SOURCE_COMPLETE.json', receipt)
    return receipt


def branch_job(entry):
    bank = R / entry['path']; assert sha(bank)==entry['sha256']
    target = bank.with_suffix('.branches.json')
    if not target.exists():
        with target.with_suffix('.log').open('a') as stream:
            subprocess.run(['node','scripts/dagger_rl_native_chain.js',entry['path'],entry['model'],
                            str(target.relative_to(R))],cwd=R,env=os.environ,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    result = read(target); assert result['passed'] and result['real_native_next_step_parity']
    return dict(source_model=entry['source_model'], path=str(target.relative_to(R)),
                sha256=sha(target), seed=result['seed'], tick=result['tick'], category=result['category'],
                executed_branch_physics_steps=result['executed_branch_physics_steps'],
                original_forecast_accepted=result['committed_plan'] is not None,
                original_forecast_diagnostic=result['original_item_forecast_at_source']['diagnostic'],
                results=[{k:v for k,v in x.items() if k!='trace'} for x in result['results']])


def owner():
    B.mkdir(parents=True, exist_ok=True)
    unique = lock(B / 'owner.lock')
    protocol = freeze()
    amendment = read(B / 'TECHNICAL_SOURCE_AMENDMENT.json') if (B / 'TECHNICAL_SOURCE_AMENDMENT.json').exists() else None
    if amendment:
        assert amendment['original_protocol_sha256']==sha(B / 'PROTOCOL.json')
        assert amendment['scientific_selection_and_budgets_unchanged']
    for name, digest in protocol['source_seal'].items():
        replacement=amendment['replaced_seals'].get(name) if amendment else None
        if replacement:
            assert replacement['before']==digest
            digest=replacement['after']
        assert sha(R / name)==digest
    with (R / 'runs/je10_bun_20261018/evaluation.lock').open('a') as global_lock:
        fcntl.flock(global_lock.fileno(),fcntl.LOCK_EX)
        atomic_json(B / 'STATUS.json',dict(stage='SOURCE_REPLAY',controls_gpu=False,unix=time.time()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            sources = list(pool.map(source_job,protocol['selected_sources']))
        entries = [b for source in sources for b in source['banks']]
        assert len(entries)<=24
        atomic_json(B / 'SOURCE_BANK_COMPLETE.json',dict(passed=True,sources=sources,
            states=len(entries),next_step_parity_each=True,new_independent_source_episodes=0))
        from scripts.dagger_rl_r10_run import notify
        notify(f'R11生产JS真实持包Bank首批完成：{len(sources)}既有来源局、{len(entries)}状态，逐状态完整env/RNG/道具归因aux下一步parity；仅机制诊断、不训练。')
        atomic_json(B / 'STATUS.json',dict(stage='PAIRED_BRANCHES',controls_gpu=False,states=len(entries)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            branches = list(pool.map(branch_job,entries))
        source_steps=sum(s['source_physics_steps'] for s in sources)
        branch_steps=sum(s['executed_branch_physics_steps'] for s in branches)
        assert source_steps<=protocol['maximum_source_replay_steps']
        assert branch_steps<=protocol['maximum_branch_steps']
        assert source_steps+branch_steps+len(entries)<=protocol['maximum_total_physics_steps']
        result=dict(passed=True,protocol_sha256=sha(B / 'PROTOCOL.json'),source_episodes=len(sources),
                    states=len(entries),source_reexecuted_physics_steps=source_steps,
                    maximum_parity_clone_physics_steps=len(entries),executed_branch_physics_steps=branch_steps,
                    new_independent_source_episodes=0,optimizer_steps=0,adopted_updates=0,
                    rows=branches,no_policy_Q_claim=True,old_continuous_resume=False)
        atomic_json(B / 'COMPLETE.json',result)
        atomic_json(B / 'STATUS.json',dict(stage='MECHANISM_COMPLETE',controls_gpu=False))
        notify('R11有界生产JS动作链诊断终态：真实来源/原计划/逐步老师/学生闭环全部保留；需据具体缺陷选最小单变量，未自动恢复旧队列。')


if __name__=='__main__':
    try: owner()
    except Exception as error:
        B.mkdir(parents=True,exist_ok=True)
        atomic_json(B / 'TECHNICAL_FAILURE.json',dict(reason=str(error).replace(str(R),'<WORKTREE>'),unix=time.time()))
        raise
    finally:
        atomic_json(B / 'OWNER_EXIT.json',dict(controls_gpu=False,unix=time.time()))
