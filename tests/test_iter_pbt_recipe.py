from scripts.iter_pbt_recipe import score,eligible,explore,replacements
import numpy as np

def test_joint_death_episodes_not_double_penalized_by_overlapping_causes():
    h={k:dict(safe_win=2,death_episode=1,self_episodes=1,trade_episodes=1,by_bot_episodes=1) for k in ('hard','normal','easy')}
    assert score(h)==6

def test_missing_or_low_delivery_donor_never_exploited():
    rows=[dict(slot=i,score=i*10,eligible=i<5,checkpoint=str(i)) for i in range(8)]
    assert replacements(rows[:7])==[]
    assert all(r['donor']<5 for r in replacements(rows))
    assert not eligible(.89,1) and eligible(.9,1)

def test_only_top_quarter_can_donate_even_if_other_slots_have_delivery():
    rows=[dict(slot=i,score=i*10,eligible=i<6,checkpoint=str(i)) for i in range(8)]
    assert replacements(rows)==[]
    rows[7]['eligible']=True
    changes=replacements(rows)
    assert len(changes)==2 and all(r['donor']==7 for r in changes)

def test_low_score_gap_does_not_replace():
    rows=[dict(slot=i,score=i/2,eligible=True,checkpoint=str(i)) for i in range(8)]
    assert replacements(rows)==[]

def test_exploration_only_authorized_fields_and_valid_mixture():
    r=dict(lr=7.5e-5,entropy_coefficient=.005,combat_shaping=.1,curriculum_fraction=.25,opponent_weights=[.2,.25,.2,.15,.1,.1])
    n=explore(r,123)
    assert set(n)==set(r)
    assert min(n['opponent_weights'])>=0 and abs(sum(n['opponent_weights'])-1)<1e-8
    assert n==explore(r,123) and any(np.isclose(n['lr'],v,rtol=1e-12,atol=0) for v in [6e-5,9.375e-5])

def test_inherited_fork_argv_is_not_an_independent_worker(tmp_path,monkeypatch):
    from scripts import iter_pbt_owner as owner
    from pathlib import Path
    original=Path
    monkeypatch.setattr(owner,'Path',lambda p:tmp_path if p=='/proc' else original(p))
    def proc(identity,parent):
        d=tmp_path/str(identity);d.mkdir()
        (d/'cmdline').write_bytes(b'python\0-m\0scripts.iter_pbt_train\0runs/iter_pbt_v1/protocols/slot00_c000001.json\0')
        (d/'stat').write_text(f'{identity} (python) R {parent} 0 0')
    proc(100,1);proc(101,100)
    assert owner.live_workers('scripts.iter_pbt_train')=={'slot00_c000001':100}
    proc(102,1)
    import pytest
    with pytest.raises(RuntimeError,match='duplicate independent'):
        owner.live_workers('scripts.iter_pbt_train')

def test_slot_cache_reuses_same_slot_files_without_moving_live_cache(tmp_path):
    from scripts.iter_pbt_owner import prepare_slot_cache
    first=prepare_slot_cache(0,'slot00_c000001',tmp_path)
    (first/'compiled-key').write_bytes(b'content-addressed')
    second=prepare_slot_cache(0,'slot00_c000002',tmp_path)
    other=prepare_slot_cache(1,'slot01_c000001',tmp_path)
    assert first.is_dir() and not first.is_symlink()
    assert second.is_symlink() and second.resolve()==first.resolve()
    assert (second/'compiled-key').read_bytes()==b'content-addressed'
    assert other.resolve()!=first.resolve()

def test_complete_cohort_uses_effective_generation_after_zero_update_retry(tmp_path,monkeypatch):
    import json
    from scripts import iter_pbt_owner as owner
    monkeypatch.setattr(owner,'BASE',tmp_path)
    (tmp_path/'protocols').mkdir()
    for g in range(8):
        attempt=1 if g==4 else 0
        name=f'slot{g:02d}_c{attempt:06d}'
        (tmp_path/'protocols'/f'{name}.json').write_text(json.dumps(dict(generation=0,chunk_start_update=320)))
        out=tmp_path/'eval'/f'{name}_u000400';out.mkdir(parents=True)
        (out/'SELECTION.json').write_text(json.dumps(dict(score=g*10,eligible=True,checkpoint=f'formal/{name}/u000400')))
    changes=owner.cohort_decisions(0)
    assert len(changes)==2 and {v['loser'] for v in changes}=={0,1}
    assert {v['donor'] for v in changes}=={6,7}
    assert owner.cohort_decisions(1)==[]

def test_only_chunk_end_points_enter_selection_queue():
    from scripts.iter_pbt_owner import is_selection_checkpoint
    assert not is_selection_checkpoint(321,320)
    assert not is_selection_checkpoint(322,320)
    assert not is_selection_checkpoint(360,320)
    assert is_selection_checkpoint(400,320)
    assert not is_selection_checkpoint(400,400)
    assert not is_selection_checkpoint(440,400)
