from pathlib import Path
from scripts.population_run_state import sha
from scripts import r4_combat_protocol as p


def test_latest_carrier_decision_replaces_old_exclusion_without_rewriting_history():
    directory=Path('docs/protocols/r4')
    files=[directory/f for f in p.FILES]
    before={str(f):sha(f) for f in files}
    result=p.effective(protocol_dir=directory);k=result['arms']['K2']
    assert k['carrier_potential']['enemy_carry']=='-.40-.80*clip(progress_home,0,1)'
    assert k['carrier_potential']['loose']==-.1
    assert k['carrier_forced']['value']==.1
    assert k['carrier_forced']['window_ticks']==20
    assert 'no additional kill' not in k['victim_carrying_learner_origin_bun']
    assert k['opponent_delivery']==-20
    assert 'native_custody_potential_unchanged' not in k
    assert result['arms']['K4']['bun_reward']=='R3 unchanged'
    assert result['kill_arena']['rule_version']=='kill-arena-sugar-respawn-v1'
    assert result['kill_arena']['terminal']['death_done'] is False
    assert before=={str(f):sha(f) for f in files}
