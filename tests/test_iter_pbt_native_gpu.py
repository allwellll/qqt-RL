from scripts.iter_pbt_native_gpu_policy import batches, reduce_concurrency, end_checkpoint

def test_holdout_304_is_one_batch_and_completed_indices_not_replayed():
    assert batches(304,set())==[list(range(304))]
    assert batches(304,set(range(8)))==[list(range(8,304))]
    assert batches(64,set(range(64)))==[]

def test_only_chunk_end_enters_new_evaluation_queue():
    assert end_checkpoint(400,320)
    assert not end_checkpoint(360,320)
    assert not end_checkpoint(322,320)

def test_drop_more_than_five_percent_reduces_concurrency():
    assert reduce_concurrency(8,20000,[19900,19950,19800])==8
    assert reduce_concurrency(8,20000,[18000,18100,18200])==4
    assert reduce_concurrency(1,20000,[18000]*3)==1
    assert reduce_concurrency(8,20000,[18000])==8

def test_gpu_modules_import_without_selecting_or_hiding_devices(monkeypatch):
    import os,importlib
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','training-sentinel')
    for name in ('scripts.iter_pbt_native_eval','scripts.iter_pbt_native_owner'):
        importlib.import_module(name)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='training-sentinel'

def test_batches_do_not_replay_partial_cpu_holdout():
    completed=set(range(0,304,2))
    planned=batches(304,completed)
    assert len(planned)==1
    assert set(planned[0]).isdisjoint(completed)
    assert set(planned[0])|completed==set(range(304))

def test_concurrency_recovers_only_after_cooldown_and_healthy_training():
    from scripts.iter_pbt_native_gpu_policy import recover_concurrency
    assert recover_concurrency(2,6,20000,[19900]*3,400,200,300)==2
    assert recover_concurrency(2,6,20000,[18000]*3,600,200,300)==2
    assert recover_concurrency(2,6,20000,[19900]*2,600,200,300)==2
    assert recover_concurrency(2,6,20000,[19000]*3,600,200,300)==3
    assert recover_concurrency(6,6,20000,[20000]*3,600,200,300)==6
    assert recover_concurrency(0,6,20000,[],600,200,300)==1

def test_dedicated_lanes_use_same_lock_as_temporary_drain():
    from scripts.iter_pbt_native_gpu_policy import gpu_lock_name, native_priority, dedicated_gpus
    assert gpu_lock_name(2,5)=='NATIVE_GPU2_lane5.lock'
    assert gpu_lock_name(1,None)=='NATIVE_GPU1.lock'
    assert dedicated_gpus({'dedicated_gpus':[2]})=={2}
    first={'name':'slot01_c000024_u001120_holdout','split':'holdout'}
    other={'name':'slot99_holdout','split':'holdout','priority':'best_candidate'}
    assert native_priority(first,[first['name']])<native_priority(other,[first['name']])

def test_native_owner_adopts_temporary_wrapper_without_signals(tmp_path):
    from scripts.iter_pbt_native_owner import native_workers, lane_busy
    p=tmp_path/'100';p.mkdir()
    (p/'cmdline').write_bytes(b'python\0runs/iter_pbt_v1/native_eval_lane.py\0runs/iter_pbt_v1/eval_requests/x.json\0--gpu\0'+b'2\0')
    (p/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=2\0DRAIN_LANE=4\0')
    (p/'stat').write_text('100 (python) R 1 0 0')
    assert native_workers(tmp_path)=={'x':dict(identity=100,gpu=2,lane=4)}
    from scripts.population_run_state import lock
    guard=lock(tmp_path/'NATIVE_GPU2_lane4.lock')
    assert lane_busy(tmp_path,2,4)
    guard.close()
    assert not lane_busy(tmp_path,2,4)
    formal=tmp_path/'101';formal.mkdir()
    (formal/'cmdline').write_bytes(b'python\0-m\0scripts.iter_pbt_native_eval\0runs/iter_pbt_v1/eval_requests/y.json\0--gpu\0'+b'3\0--lane\0'+b'5\0')
    (formal/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=3\0')
    (formal/'stat').write_text('101 (python) R 1 0 0')
    assert native_workers(tmp_path)['y']==dict(identity=101,gpu=3,lane=5)

def test_borrowed_gpu_waits_for_training_chunk_exit():
    from scripts.iter_pbt_native_gpu_policy import evaluation_slots
    config=dict(dedicated_gpus=[2,5],lanes_per_gpu=2)
    assert evaluation_slots(config,{0,1,3,4,5,6,7})==[(2,0),(2,1)]
    assert evaluation_slots(config,{0,1,3,4,6,7})==[(2,0),(2,1),(5,0),(5,1)]
    config['temporary_borrowed_gpus']=[5]
    assert evaluation_slots(config,(),recovery_pending=False)==[(2,0),(2,1)]

def test_idle_borrowed_card_returns_without_interrupting_other_lanes():
    from scripts.iter_pbt_native_gpu_policy import returnable_gpus
    config=dict(temporary_borrowed_gpus=[3,6])
    assert returnable_gpus(config,{2,3},True)==set()
    assert returnable_gpus(config,{2,3},False)=={6}
    assert returnable_gpus(config,{2},False)=={3,6}
