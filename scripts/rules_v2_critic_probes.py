"""Prepare and score version-bound common-state Critic probes, CPU only.

The probe sampler is deliberately not run while G1/G2 are blocked. Scoring
requires a genuine terminal-MC artifact and provenance, never a model's own V
as a bootstrap target. This module cannot start GPU training.
"""
from __future__ import annotations
import argparse
import json
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha

BASE=Path('runs/rules_v2_critic_20261027/v1')
PROTOCOL=Path('runs/authorized_campaign_20261026/protocols/rules_v2_critic_ab_v1.json')
IMPLEMENTATION=PROTOCOL.with_name('rules_v2_critic_ab_v1_implementation01.json')
RULE_VERSION='rules_v2_native_parity'
PHASES=('enemy_free_combat','native_control_window','carrying_return')


def read(path):return json.loads(Path(path).read_text())


def require_rules_ready(base=BASE):
    gates=read(Path(base)/'GATES.json')
    for name in ('G1','G2'):
        if gates[name]['status']!='PASS':
            raise RuntimeError(f'{name} blocked; old-rule data cannot stand in for new-rule probes')
    receipt=read(Path(base)/'RULE_VERSION.json')
    if receipt.get('rule_version')!=RULE_VERSION or not receipt.get('source_sha256'):
        raise ValueError('Missing new-rule source seal')
    for path,expected in receipt['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('New-rule source SHA mismatch')
    return receipt


def prepare(base=BASE):
    base=Path(base);original=read(PROTOCOL);config=read(IMPLEMENTATION)
    specs=[]
    for arm in original['arms']:
        for seed in original['seeds']:
            specs.append(dict(id=f'{arm}_s{seed}',checkpoint=str(base/'formal'/f'{arm}_s{seed}'/'u000320'),
                              arm=arm,seed=seed,update=320,environment_steps=5242880))
    manifest=dict(id='RULES-V2-COMMON-PROBES-20261027-V1',protocol=str(PROTOCOL),protocol_sha256=sha(PROTOCOL),
        rule_version=RULE_VERSION,ready=False,blocked_on=['G1','G2'],
        teacher=False,no_synthetic_carrier=True,no_GPU=True,
        source_model='same frozen parent_s2026102751_u064 before any new training',
        source_game_budget=768,source_games_per_split_mode=192,
        phases=list(PHASES),states_per_phase_mode_split=128,rollouts_per_state=4,
        action_modes=['categorical','greedy'],splits=['dev','holdout'],
        dev_seed_start=2026162600,holdout_seed_start=2026172600,
        selection='first causal eligible state of each phase per source game, then seed order; first128 per phase/mode/split; no fill of missing strata',
        source_game_isolation='same game seed across all models/modes assigned to one split',
        rollout_rng='fold_in(source seed, phase index, rollout index); independent repeat streams paired across models',
        gamma=.995,terminal_MC='complete rewards to actual terminal incl settlement tail; no V bootstrap',
        missing='undersupply/truncation/overflow listed explicitly; no early horizon counted as terminal',
        models=specs,untrained_parent=config['parent'],configuration=config['config'],
        native_full_game=dict(dev_games_per_mode=304,holdout_games_per_mode=304,
            opponent='fixed JAX rule3',seeds='same frozen split seeds as probes; natural full games recorded separately'),
        production_js=dict(engine='production JS',opponent='Hunter',tiers=['hard','normal','easy'],games_per_tier_mode=64,
            action_modes=['categorical','greedy'],seed_start=2026182600,teacher=False,
            writes_to_database=False,checkpoint='fixed u320 only; no peak selection'),
        limitations=['Common parent source distribution, not each candidate visitation distribution',
                    'Map/layout/seat coverage must come from actual source data; never assumed',
                    'Four MC repeats have residual noise; CI clustered by source game and paired training seed'])
    atomic_json(base/'PROBE_EVAL_MANIFEST.json',manifest)
    return manifest


def terminal_mc_returns(rewards,valid,terminal,gamma=.995):
    rewards=np.asarray(rewards,float);valid=np.asarray(valid,bool);terminal=np.asarray(terminal,bool)
    if rewards.shape!=valid.shape or rewards.shape!=terminal.shape or rewards.ndim!=3:
        raise ValueError('Expected state x rollout x tick arrays')
    lengths=valid.sum(-1)
    if (lengths==0).any():raise ValueError('Empty terminal MC trajectory')
    expected=np.arange(valid.shape[-1])<lengths[...,None]
    if not np.array_equal(valid,expected):raise ValueError('Noncontiguous valid MC mask')
    for index in np.ndindex(lengths.shape):
        n=int(lengths[index])
        if not terminal[index+(n-1,)] or terminal[index][:n-1].any():
            raise ValueError('Censored or invalid terminal MC; never bootstrap using tested V')
    if not np.isfinite(rewards[valid]).all():raise ValueError('Nonfinite MC reward')
    discount=np.power(gamma,np.arange(rewards.shape[-1]))
    return (np.where(valid,rewards,0.)*discount).sum(-1)


def score(payload,params_path,base=BASE):
    """Offline numerical scoring; sampler must supply sealed real-MC evidence."""
    version=require_rules_ready(base)
    payload=Path(payload);meta=read(payload.with_suffix('.json'))
    if meta.get('rule_version')!=RULE_VERSION or meta.get('rule_source_sha256')!=version['source_sha256']:
        raise ValueError('Probe rule version differs from training version')
    if meta.get('teacher') is not False or not meta.get('source_games'):
        raise ValueError('Missing genuine teacher-free source-game provenance')
    if meta.get('gamma')!=.995:raise ValueError('Critic and reward discount mismatch')
    if meta.get('action_mode') not in ('categorical','greedy') or meta.get('split') not in ('dev','holdout'):
        raise ValueError('Unregistered probe mode or split')
    if sha(payload)!=meta['payload_sha256'] or sha(params_path)!=meta['checkpoint_state_sha256']:
        raise ValueError('Probe or checkpoint SHA mismatch')
    import pickle
    import jax
    from scripts import rules_v2_critic as critic
    from scripts.carrier_mechanism_stats import calibration,cluster_interval
    assert jax.default_backend()=='cpu'
    with Path(params_path).open('rb') as f:saved=pickle.load(f)
    params=saved['params']
    with np.load(payload,allow_pickle=False) as z:
        if z['mc_rewards'].shape[1]!=4:raise ValueError('Frozen probe requires four MC repeats per state')
        samples=terminal_mc_returns(z['mc_rewards'],z['mc_valid'],z['mc_terminal'])
        obs,gv,ledger=z['obs'],z['global_state'],z['ledger']
        phases,source_ids=z['phase'],z['source_game_id']
        n=len(samples)
        if not n or obs.shape!=(n,24,13,15) or gv.shape!=(n,24) or ledger.shape!=(n,len(critic.LEDGER_FIELDS)):
            raise ValueError('Probe input shape differs from frozen ABI')
        if phases.shape!=(n,) or source_ids.shape!=(n,) or not np.isin(phases,PHASES).all():
            raise ValueError('Invalid probe phase or source identity')
        if not all(np.isfinite(x).all() for x in (obs,gv,ledger)):
            raise ValueError('Nonfinite probe inputs')
        prediction=np.asarray(jax.jit(critic.forward)(params,obs,gv,ledger)[2])
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite Critic prediction')
    rows=[]
    for phase in PHASES:
        mask=phases==phase
        if not mask.any():
            rows.append(dict(phase=phase,states=0,status='MISSING_SOURCE_COVERAGE'));continue
        result=calibration(prediction[mask],samples[mask])
        result['bias_source_game_cluster_CI']=cluster_interval(prediction[mask]-samples[mask].mean(-1),source_ids[mask])
        rows.append(dict(phase=phase,metrics=result,source_games=len(set(source_ids[mask].tolist())),
                         target='terminal MC only, not tested-V bootstrap',action_mode=meta['action_mode'],split=meta['split']))
    result=dict(passed=True,rows=rows,rule_version=RULE_VERSION,raw_sources=[
        dict(path=str(p),sha256=sha(p)) for p in (payload,payload.with_suffix('.json'),Path(params_path))],
        note='Four MC rollouts do not establish exact V; EV cannot identify constant bias')
    target=Path(base)/'probe_scores'/f'{meta["model"]}_{meta["split"]}_{meta["action_mode"]}.json'
    atomic_json(target,result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['prepare','score'])
    parser.add_argument('--payload');parser.add_argument('--checkpoint');args=parser.parse_args()
    result=prepare() if args.action=='prepare' else score(args.payload,args.checkpoint)
    print(json.dumps(dict(action=args.action,ready=result.get('ready'),rule_version=result['rule_version'])))


if __name__=='__main__':main()
