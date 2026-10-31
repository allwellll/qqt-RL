"""R3 performance-only custody handoff, frozen R3 reward/inputs preserved."""
import os
from pathlib import Path
from types import ModuleType
from scripts.r3_items_train import worker_source as previous_source


def worker_source():
    source = previous_source().replace("out/'RESTORE.json',dict(passed=True,fresh_process=True",
        "out/'R3_PERFORMANCE2_RESTORE.json',dict(passed=True,fresh_process=True")
    anchor = "        if index in (int(protocol['chunk_start_update'])+1,"
    assert source.count(anchor) == 1
    source=source.replace(anchor,"        if index in (start+1,start+2,start+3) or index in (int(protocol['chunk_start_update'])+1,")
    anchor='        before=P.one(par);began=time.monotonic()'
    assert source.count(anchor)==1
    source=source.replace(anchor,"""        # Optional whole-update scheduling, enabled only after measured
        # shared-card warm cycles exceed60s. No samples/updates are changed.
        import fcntl
        _stagger_started=time.monotonic();_stagger_file=None;_stagger_wait=0.
        _control=BASE/'controls/GPU_STAGGER.json'
        _schedule=json.loads(_control.read_text()) if _control.exists() else {}
        if _schedule.get('mode')=='whole_update_mutex' and str(_schedule['gpu'])==os.environ['JAXBOMB_DEVICE_OVERRIDE']:
            _stagger_file=(BASE/'GPU_STAGGER.lock').open('a')
            fcntl.flock(_stagger_file.fileno(),fcntl.LOCK_EX)
            _stagger_wait=time.monotonic()-_stagger_started
            if paused():save(index-1,dict(user_paused=True));return
"""+anchor+"\n        began=_stagger_started")
    anchor='        rollout=time.monotonic()-began'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'-_stagger_wait')
    anchor='            update_l2=delta,cycle_seconds=time.monotonic()-began,rollout_seconds=rollout,'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'stagger_wait_seconds=_stagger_wait,')
    anchor="        atomic_json(out/'PROGRESS.json',dict(iterations=index,last=row))"
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"""
        if _stagger_file is not None:
            fcntl.flock(_stagger_file.fileno(),fcntl.LOCK_UN);_stagger_file.close()
            time.sleep(.01)  # Let the other already-waiting slot take a turn.
""")
    return source


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'], JAX_PLATFORMS='cuda',
        JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    module=ModuleType('r3_performance2_worker')
    module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__)
    module.main()


if __name__=='__main__':
    main()
