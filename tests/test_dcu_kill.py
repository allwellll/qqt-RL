"""Configuration and wall-time admission checks without starting a DCU."""
import subprocess
import sys
from pathlib import Path
import pytest
from scripts.train_dcu_kill import arguments,plan


@pytest.mark.parametrize('options',[
    ['--devices','0,0'],['--devices','-1'],['--envs','3'],['--ticks','0'],
    ['--minibatch','0'],['--minibatch','3'],['--hours','6'],['--critic-warmup','-1'],
    ['--seed','-1'],['--seed','4294967296'],
])
def test_invalid_geometry_rejected(options):
    with pytest.raises(SystemExit):arguments(['plan',*options])


def test_five_hour_eight_card_plan():
    row=plan(arguments(['plan']))
    assert row['wall_seconds']==18000
    assert row['global_envs']==64 and row['global_samples']==4096
    assert row['global_minibatch']==512
    assert row['includes_compilation_and_evaluation']


def test_smoke_is_bounded_and_training_is_explicit(tmp_path):
    with pytest.raises(SystemExit):arguments(['smoke','--out',str(tmp_path/'unused')])
    with pytest.raises(SystemExit):arguments(['train','--out',str(tmp_path/'unused')])
    args=arguments(['smoke','--devices','0','--envs','2','--ticks','4','--minibatch','4',
        '--max-updates','2','--out',str(tmp_path/'unused')])
    assert args.max_updates==2


def test_plan_does_not_import_jax_or_create_output(tmp_path):
    out=tmp_path/'no-output'
    code='import sys; from scripts.train_dcu_kill import main; main(["plan", "--out",'+repr(str(out))+']); assert "jax" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not out.exists()
