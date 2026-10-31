import json
import pytest
from scripts import bomb_threat_iter2_schedule as S


def test_formal_notification_requires_real_post_restore_update_not_launch():
    protocol=dict(chunk_start_update=960)
    assert not S.formal_update_recorded(962,'formal',protocol)
    assert not S.formal_update_recorded(963,'smoke',protocol)
    assert S.formal_update_recorded(963,'formal',protocol)


def test_a2_switch_only_complete_natural_boundary_and_preserves_full_source(tmp_path,monkeypatch):
    base=tmp_path/'runs/bomb_threat_v1';(base/'iter2').mkdir(parents=True)
    run='bt_slot07_c000015';out=base/'formal'/run;out.mkdir(parents=True)
    (base/'iter2/STEER1320_AUTHORIZATION.json').write_text(json.dumps(dict(status='AUTHORIZED_READY',A2=dict(boundary_run=run,boundary_update=960))))
    (base/'protocols').mkdir();recipe=dict(iter2_arm='A2',lr=.000075,curriculum_fraction=.1,opponent_weights=[.05,.35,.30,.15,.1,.05])
    (base/'protocols'/f'{run}.json').write_text(json.dumps(dict(arms={run:recipe})))
    cp=out/'u000960';cp.mkdir();(cp/'state.pkl').write_bytes(b'full');(cp/'policy.pt').write_bytes(b'actor')
    from scripts.population_run_state import sha
    done=dict(checkpoint=str(cp.relative_to(tmp_path)),update=960,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'))
    monkeypatch.setattr(S,'BASE',base);monkeypatch.setattr(S,'ROOT',tmp_path)
    def make(slot,chunk,source,new):
        assert slot==7 and source==done
        assert {k:v for k,v in new.items() if k not in ('entropy_fixed_005','entropy_restore_authorization')}==recipe
        assert new['entropy_fixed_005'] is True
        return base/'protocols'/'successor.json'
    monkeypatch.setattr(S,'make_protocol',make)
    entry=dict(run=run,iter2_arm='A2')
    assert S.apply_entropy_restore(7,entry,done,16) is None
    (out/'COMPLETE.json').write_text(json.dumps(done))
    result=S.apply_entropy_restore(7,entry,done,16)
    assert result['source']==done and result['entropy_restored_1320']
    assert S.apply_entropy_restore(1,entry,done,16) is None
    assert S.apply_entropy_restore(7,dict(entry,entropy_restored_1320=True),done,16) is None


def test_a3_bomb_kill_trigger_is_strict_three_cadence_points_and_no_missing_fill():
    from scripts.bomb_threat_steer1320 import a3_bomb_kill_observation
    arms=[dict(update=u,train_unix=u,metrics=dict(bombs_per_game=30.,causal_kills=20,draw_kill_difference=-1)) for u in (680,720,760)]
    controls=[dict(slot=s,arm=None,train_unix=u,metrics=dict(bombs_per_game=24.,causal_kills=20),marker=f'{s}/{u}') for u in (680,720,760) for s in ('0','3','6')]
    assert a3_bomb_kill_observation(arms,controls)['status']=='RESTORE_ORIGINAL_RECIPE'
    assert a3_bomb_kill_observation(arms[:2],controls)['status']=='PENDING'
    arms[1]['metrics']['bombs_per_game']=29.
    assert a3_bomb_kill_observation(arms,controls)['status']=='CONTINUE'
    arms[1]['metrics']['bombs_per_game']=30.
    arms[1]['metrics']['causal_kills']=21
    assert a3_bomb_kill_observation(arms,controls)['status']=='CONTINUE'
    assert a3_bomb_kill_observation(arms,[r for r in controls if r['slot']!='6'])['status']=='PENDING'
    arms[1]['update']=800
    assert a3_bomb_kill_observation(arms,controls)['status']!='RESTORE_ORIGINAL_RECIPE'


def test_d4_draw_stock_and_tick_denominators():
    from scripts.bomb_threat_steer1320 import summarize_d4
    records=[dict(seed=1,seat=1,ticks=2400,done=True,winner=None,stored=[[1,0],[0,1]],
                  causal_kills=[0,2],kills=[dict(tick=400,attacker=1),dict(tick=2300,attacker=1)],
                  natural=dict(delivery=0,pickup=0)),
             dict(seed=2,seat=0,ticks=2400,done=True,winner=None,stored=[[0,0],[0,1]],
                  causal_kills=[1,0],kills=[dict(tick=1800,attacker=0)],natural=dict(delivery=1,pickup=1)),
             dict(seed=3,seat=0,ticks=500,done=True,winner=0,stored=[[1,1],[0,0]],causal_kills=[1,0],kills=[dict(tick=100,attacker=0)],natural=dict(delivery=1,pickup=1))]
    result=summarize_d4(records)
    assert result['games']==3 and result['kill_draw_games']==2 and result['kill_draw_kills']==3
    assert result['kill_draw_stock_total_difference_histogram']=={'0':1,'-1':1}
    assert result['kill_draw_no_delivery_games']==1
    assert result['kill_draw_tick_bins']=={'1..600':1,'601..1200':0,'1201..1800':1,'1801..2400':1}
    assert result['kill_draw_missing_event_count']==0


def test_fixed_entropy_worker_only_changes_entropy_and_requires_complete_continuation():
    from pathlib import Path
    old=Path('scripts/bomb_threat_iter2_diagnostics_train.py').read_text()
    new=Path('scripts/bomb_threat_iter2_fixed_entropy_train.py').read_text()
    expected=old.replace('Policy-only ITER2 initialization, frozen-Actor Critic warmup and bounded PPO.',
        'STEER1320 full-state A2 continuation with fixed entropy; no reinitialization.')
    expected=expected.replace("    out=ROOT/protocol['namespace']/args.arm;",
        "    assert spec.get('entropy_fixed_005') is True and spec['iter2_arm']=='A2'\n"
        "    assert protocol.get('continuation_source'), 'fixed-entropy switch must preserve full current state'\n"
        "    out=ROOT/protocol['namespace']/args.arm;")
    expected=expected.replace("recipe.entropy(index,spec['iter2_arm']=='A2')","recipe.entropy(index,False)")
    begin=new.index("        atomic_json(out/'FULL_STATE_CONTINUATION.json'")
    end=new.index("\n    else:",begin)
    assert 'reinitialized=False,extra_warmup_updates=0' in new[begin:end]
    assert new[:begin]+new[end+1:]==expected
