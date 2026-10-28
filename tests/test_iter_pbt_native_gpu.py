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
    assert reduce_concurrency(1,20000,[18000]*3)==0
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
