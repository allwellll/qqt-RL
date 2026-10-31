"""Recomputable campaign coverage and mechanism audits; no GPU or training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from scripts.population_run_state import atomic_json,sha


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('protocol');a=parser.parse_args()
    p=json.loads(Path(a.protocol).read_text());out=Path(p['namespace']);out.mkdir(parents=True,exist_ok=True)
    findings=[];missing=[];sources=[]
    if p['kind']=='bank_coverage':
        from scripts.carrier_mechanism_analysis import bank
        from scripts.carrier_mechanism_stats import validate_bank_record
        items=bank();groups={}
        for x in items:
            r=x['record'];validate_bank_record(r)
            assert len(x['prefix_actions'])==len(x['prefix_envkeys'])
            key=(r.get('map_id'),r.get('layout'),r.get('split'),r.get('seat',0),r.get('source_policy'))
            groups[str(key)]=groups.get(str(key),0)+1
        findings.append(dict(real_certified_states=len(items),source_episodes=len({x['record']['source_episode'] for x in items}),groups=groups,
            causal_inbound_evidence='stored actual prefix actions, environment RNG, before/after env and aux; validated source records',
            synthetic_carry_used=False))
        missing=['Existing Bank still map806/P0; expansion to independent maps/P1 and window/combat predecessors has not yet been collected. Coverage audit does not count as collection or generalization proof.']
    elif p['kind']=='ledger_observability':
        files=['scripts/dagger_rl_r4_train.py','scripts/ppo_scale_longrun_train.py','scripts/campaign_reward_v2.py','scripts/carrier_mechanism_engine.py','scripts/combat_item_engine.py']
        sources=[dict(path=f,sha256=sha(f)) for f in files]
        text=Path('scripts/dagger_rl_r4_train.py').read_text()
        assert 'compute_gae(r,v,nv,done,.995,.95)' in text
        findings=[dict(gae_gamma=.995,gae_lambda=.95,terminal_bootstrap='done masks GAE; raw nv evaluated before fresh reset; native game-end settlement latch distinct from horizon',
            ppo_only=True,DAgger=0,combat_KL=0,
            hidden_reward_ledgers=['paid','shape_paid','transport pending/balance','control credit balance','route phase/offset','campaign_task_paid'],
            critic_observability='Policy/critic receive encoded physical obs/global state, not full aux reward ledgers. Therefore identical physical observations can have different future clawbacks.',
            reward_change_Adam='V2 both arms inherit same matched R1 full Adam; treatment critic target changes with reward package, inherited value/Adam adaptation is not separately randomized')]
        missing=['Fixed common-state phase KL and independent MC Critic EV need new numerical probe; code audit alone is not their result.']
    elif p['kind']=='item_branch_inventory':
        root=Path('runs');paths=list(root.glob('dagger_rl_202610*/**/ACTION_CHAIN_AUDIT.json'))
        for f in paths:
            try:d=json.loads(f.read_text())
            except (OSError,ValueError):continue
            findings.append(dict(path=str(f),sha256=sha(f),examples=d.get('examples'),source_scope=d.get('scope',d.get('causal_branches_scope'))))
        missing=['Prior enemy-wait microbranches are observational/physical witnesses only. Same-opportunity responding-Hunter banana/glue net-benefit branches remain required; no causal success inferred from a hit.']
    else:raise ValueError('Unrecognized audit kind')
    atomic_json(out/'RESULT.json',dict(protocol=a.protocol,protocol_sha256=sha(a.protocol),findings=findings,missing=missing,
        scientific_acceptance=False,scope='Evidence/coverage audit only; required missing numerical experiments remain uncompleted',sources=sources,unix=time.time()))
    atomic_json(out/'AUDIT_COMPLETE.json',dict(passed=True,result_sha256=sha(out/'RESULT.json'),
        scientific_acceptance=False,scope='Bounded audit executed; not full scientific task acceptance'))


if __name__=='__main__':main()
