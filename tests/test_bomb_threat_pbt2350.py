from scripts.bomb_threat_pbt2350 import apply_switch
from scripts.bomb_threat_pbt2350_train import worker_source


def test_only_authorized_slot_and_exact_boundary_run():
    assert apply_switch(5,{'run':'not-target'}, {}, 1) is None
    assert apply_switch(4,{'run':'not-target'}, {}, 1) is None


def test_rekey_is_after_donor_parity_and_before_fresh_restore():
    source=worker_source()
    compile(source,'pbt-worker','exec')
    assert source.index("'donor next-step mismatch before rekey'")<source.index('keys=jax.random.split(jax.random.PRNGKey(seed),n)[None]')<source.index('    if args.verify_restore:')
    assert "critic_warmup_updates=0" in source


def test_full_boundary_selects_latest_atomic_and_preserves_recipe(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from scripts import bomb_threat_pbt2350 as S
    from scripts.population_run_state import atomic_json,sha
    monkeypatch.setattr(S,'ROOT',tmp_path);base=tmp_path/'runs/bomb_threat_v1';monkeypatch.setattr(S,'BASE',base)
    auth=base/'iter2/pbt2350/AMENDMENT.json';monkeypatch.setattr(S,'AUTH',auth)
    atomic_json(auth,dict(status='AUTHORIZED_AT_FULL_CHUNK',boundary_run='target',donor_run='donor'))
    recipe={'iter2_arm':'A3','lr':7.5e-5,'entropy_coefficient':.005,'opponent_weights':[.2,.25,.2,.15,.1,.1]}
    def commit(run,u):
        cp=base/'formal'/run/f'u{u:06d}';cp.mkdir(parents=True)
        (cp/'state.pkl').write_bytes(f'complete-state-{u}'.encode());(cp/'policy.pt').write_bytes(b'policy')
        cm=dict(passed=True,checkpoint=str(cp.relative_to(tmp_path)),state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),update=u,metrics={'finite':True})
        atomic_json(cp/'COMMITTED.json',cm);atomic_json(cp.parent/f'NEXT_STEP_u{u:06d}.json',{'digest':'token'})
        return cm
    done=commit('target',320);atomic_json(base/'formal/target/COMPLETE.json',done)
    commit('donor',40);latest=commit('donor',80)
    atomic_json(base/'protocols/target.json',{'chunk_start_update':0,'chunk_updates':320})
    atomic_json(base/'protocols/donor.json',{'arms':{'donor':recipe},'iter2_arm':'A3','source_sha256':{}})
    # Two immutable worker ABI inputs are required by the frozen new protocol.
    for file in ['scripts/bomb_threat_pbt2350.py','scripts/bomb_threat_pbt2350_train.py']:
        p=tmp_path/file;p.parent.mkdir(exist_ok=True);p.write_text('source')
    new=S.apply_switch(4,{'run':'target','iter2_arm':'A3'},done,24)
    proto=json.loads((base/'protocols'/f"{new['run']}.json").read_text())
    assert proto['continuation_source']==latest and proto['chunk_start_update']==80
    assert next(iter(proto['arms'].values()))==recipe
    assert proto['pbt_lineage']['critic_warmup_updates']==0
    assert proto['pbt_lineage']['target_seed']==2026112904
    assert new['pbt2350_applied'] and (base/'formal/target/u000320/state.pkl').exists()
