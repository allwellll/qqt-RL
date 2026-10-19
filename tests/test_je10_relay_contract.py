"""Technical handoff receipts, independent of all capability metrics."""
import pytest
from scripts.je10_supervise import gpu_release_receipt, formal_update_receipt


def test_release_requires_all_eight_devices_and_no_compute_clients():
    inventory='\n'.join(f'{i}, GPU-{i}' for i in range(8))
    assert gpu_release_receipt(inventory,'')['released'] is True
    assert gpu_release_receipt(inventory,'GPU-3, 123, 40000')['released'] is False
    assert gpu_release_receipt(inventory.rsplit('\n',1)[0],'')['released'] is False


def test_smoke_or_launch_alone_cannot_confirm_formal_nonzero():
    with pytest.raises(AssertionError):formal_update_receipt({}, {})
    row={'generation':1,'finite':True,'nonzero_update':True,
         'actor_update_l2':[1.]*8,'critic_update_l2':[2.]*8,'fresh_ticks':2097152}
    cp={'generation':1,'passed':True,'replicas_exact':True,'devices':8,'fresh_ticks':2097152}
    receipt=formal_update_receipt(row,cp)
    assert receipt['formal_nonzero_verified'] and receipt['devices']==8
    assert receipt['capability_gate'] is False
    row['actor_update_l2'][3]=0.
    with pytest.raises(AssertionError):formal_update_receipt(row,cp)


def test_formal_receipt_rejects_nan_or_wrong_device_checkpoint():
    row={'generation':1,'finite':True,'nonzero_update':True,
         'actor_update_l2':[1.]*8,'critic_update_l2':[2.]*8,'fresh_ticks':2097152}
    cp={'generation':1,'passed':True,'replicas_exact':True,'devices':4,'fresh_ticks':2097152}
    with pytest.raises(AssertionError):formal_update_receipt(row,cp)
    cp['devices']=8;row['critic_update_l2'][7]=float('nan')
    with pytest.raises(AssertionError):formal_update_receipt(row,cp)


def test_successor_host_fix_requires_predecessor_terminal(tmp_path,monkeypatch):
    import json
    from scripts import je10_supervise as relay
    monkeypatch.setattr(relay,'ROOT',tmp_path)
    worker=tmp_path/'scripts/je10_worker.py';worker.parent.mkdir()
    old="state_hash=hashlib.sha256(np.asarray(batch[0])[dev,tick,env_index].tobytes()+np.asarray(batch[1])[dev,tick,env_index].tobytes()).hexdigest()"
    original='            failures=[]\n                '+old+"\n            row['signal_decomposition']=dict(control=float(trace[...,1].sum()),route=float(trace[...,2].sum()),\n"
    worker.write_text(original)
    parent=tmp_path/'formal/arm/gen000128';parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):relay.install_successor_host_trace_fix(parent)
    assert worker.read_text()==original
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps({'generation':127,'atomic_stop':True}))
    with pytest.raises(AssertionError):relay.install_successor_host_trace_fix(parent)
    assert worker.read_text()==original
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps({'generation':128,'atomic_stop':False}))
    relay.install_successor_host_trace_fix(parent)
    assert 'observation_state_hash(batch[0],batch[1]' in worker.read_text()
    installed=worker.read_text();relay.install_successor_host_trace_fix(parent)
    assert worker.read_text()==installed


def test_execution_source_is_captured_before_reset_changes_family(tmp_path,monkeypatch):
    import json
    from scripts import je10_supervise as relay
    monkeypatch.setattr(relay,'ROOT',tmp_path)
    path=tmp_path/'scripts/je09_stage_h5c_ops.py';path.parent.mkdir()
    path.write_text('        other=curriculum.assisted_actions(other,mm[oppidx],assist,kwalk)\n        kinds=jnp.where(done,newkinds,kinds)\n            context=jnp.concatenate([context,trace],-1)\n')
    parent=tmp_path/'formal/arm/gen000128';parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):relay.install_successor_exposure_trace(parent)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps({'generation':128,'atomic_stop':False}))
    relay.install_successor_exposure_trace(parent);source=path.read_text()
    assert source.index('executed_source=')<source.index('kinds=jnp.where(done')
    assert 'trace,provenance' in source
    relay.install_successor_exposure_trace(parent)
    assert source==path.read_text()
