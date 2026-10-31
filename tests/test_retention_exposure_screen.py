import json
from types import SimpleNamespace
from scripts import retention_exposure_screen as screen


def test_holdout_uses_new_seeds_and_namespace_and_never_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(screen, 'declaration', lambda: dict(
        evaluation_report='reports/retention_exposure_evaluation_20261020.md',
        dev_rng=20261019, test_rng=20261031,
        route_dev_seeds=list(range(2026102100,2026102132)),
        route_test_seeds=list(range(2026110100,2026110164))))
    monkeypatch.setattr(screen, 'ROOT', tmp_path)
    monkeypatch.setattr(screen.engine, 'env_for', lambda run, cpu: dict(CUDA_VISIBLE_DEVICES='' if cpu else '0,1,2,3,4,5,6,7'))
    calls=[]
    monkeypatch.setattr(screen.subprocess, 'run', lambda args, **kw: (calls.append((args,kw)) or SimpleNamespace(returncode=0)))
    run=tmp_path/'runs/r2/A1';run.mkdir(parents=True)
    assert screen.evaluation_job(run,'test',32)==0
    args,kw=calls[-1]
    assert args[1]==screen.SCRIPT and 'evalchild' in args
    assert kw['env']['CUDA_VISIBLE_DEVICES']==''
    assert kw['env']['JE10_EVAL_SEED']=='20261031'
    assert kw['env']['JE10_ROUTE_SEED_START']=='2026110100'
    assert kw['env']['JE10_ROUTE_GAMES']=='64'
    assert kw['env']['JE10_EVAL_NAMESPACE']=='authority_retention_test'
    assert screen.evaluation_job(run,'dev',8)==0
    assert calls[-1][1]['env']['JE10_ROUTE_GAMES']=='32'
    assert calls[-1][1]['env']['JE10_ROUTE_SEED_START']=='2026102100'


def test_candidate_install_is_explicit_and_only_full_arm(monkeypatch, tmp_path):
    from scripts import retention_core_exposure as candidate
    calls=[]
    monkeypatch.setattr(screen.engine,'configure',lambda run: calls.append('baseline') or 'worker')
    monkeypatch.setattr(candidate,'install',lambda: calls.append('full'))
    for fraction,expected in [(.5,['baseline']),(1.,['baseline','full'])]:
        calls.clear()
        (tmp_path/'plan.json').write_text(json.dumps(dict(core_history_fraction=fraction)))
        assert screen.configure(tmp_path)=='worker'
        assert calls==expected


def test_handoff_requires_all_fixed_training_release_but_not_cpu_thresholds(monkeypatch, tmp_path):
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    base=tmp_path/'runs/r1';base.mkdir(parents=True)
    monkeypatch.setattr(screen,'R1',base)
    specs=[dict(namespace=f'runs/r1/run{i}') for i in range(6)]
    (base/'RESUMPTION_DECLARATION.json').write_text(json.dumps(dict(runs=specs)))
    (base/'STATUS.json').write_text(json.dumps(dict(state='TRAINING_COMPLETE_EVALUATING')))
    for i,spec in enumerate(specs):
        run=tmp_path/spec['namespace'];run.mkdir()
        (run/'TRAINING_COMPLETE.json').write_text('{}')
        if i>0:(run/'GPU_RELEASE_RECEIPT.json').write_text('{}')
    assert screen.preceding_training_released()
    assert not (base/'SCREEN_COMPLETE.json').exists()
    (tmp_path/specs[-1]['namespace']/'GPU_RELEASE_RECEIPT.json').unlink()
    assert not screen.preceding_training_released()
