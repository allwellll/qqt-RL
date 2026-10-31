"""Resolve immutable R4 decisions in order; never mutate an earlier contract."""
import copy
import json
from pathlib import Path
from scripts.population_run_state import atomic_json,sha

BASE=Path('runs/r4_combat_20261031')
FILES=('r4_combat_v1.json','r4_combat_v1_amendment01_sugar_respawn.json',
       'r4_combat_v1_amendment02_carrier_conservative.json')


def effective(base=BASE, *, protocol_dir=None):
    directory=Path(protocol_dir) if protocol_dir is not None else base/'protocols'
    paths=[directory/name for name in FILES]
    original,sugar,carrier=[json.loads(p.read_text()) for p in paths]
    assert sugar['status']==carrier['status']=='FROZEN'
    result=copy.deepcopy(original)
    result['resolved_source_sha256']={str(p):sha(p) for p in paths}
    result['kill_arena']=dict(rule_version=sugar['ruleset'],max_ticks=sugar['horizon'],
        respawn=sugar['respawn'],terminal=sugar['terminal'],parity=sugar['parity'])
    k2=dict(sugar['k2'])
    for key in ('victim_carrying_learner_origin_bun','native_custody_potential_unchanged'):
        k2.pop(key)
    k2.update(carrier_potential=carrier['phi'],carrier_difference=carrier['difference'],
        victim_carrying_learner_origin_bun=carrier['kill_carrier'],
        carrier_forced=carrier['forced'],opponent_delivery=carrier['opponent_delivery'],
        carrier_runtime='scripts.r4_k2_carrier_reward',
        status='PREPARATION_NOT_FORMAL',activation_gate=carrier['activation_gate'])
    result['arms']['K2']=k2
    # K4's bun mode retains the R3 reward as specified in its own contract;
    # only its cage-free mode uses the latest sugar/respawn kill definition.
    result['arms']['K4'].update(bun_reward='R3 unchanged',arena_reward=dict(kill=1.,death=-1.,self_death=-1.5,
        forced=.2,terminal=1.),status='WAIT_PARITY_AND_COLLECTOR_ACCEPTANCE')
    result['status']='RESOLVED_PREPARATION_NOT_FORMAL'
    result['reward_versions']=dict(K2='r4-k2-carrier-conservative-v1',
        K4_arena=sugar['ruleset'],K1='parent R3 unchanged',K3='parent R3 unchanged')
    return result


if __name__=='__main__':
    result=effective();atomic_json(BASE/'EFFECTIVE_PROTOCOL.json',result)
    print(json.dumps(dict(status=result['status'],file=str(BASE/'EFFECTIVE_PROTOCOL.json'),sha256=sha(BASE/'EFFECTIVE_PROTOCOL.json'))))
