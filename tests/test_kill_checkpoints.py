import json
from pathlib import Path
import pickle
from unittest.mock import patch

import pytest

from qqt_rl.training.checkpoints import CheckpointStore


def test_retention_preserves_initial_latest_and_other_directories(tmp_path):
    other=tmp_path/'old-project';other.mkdir();(other/'untouched').write_text('keep')
    store=CheckpointStore(tmp_path,keep=3,min_free_gib=0)
    weights=dict(actor={'w':[1,2]},critic={'w':[3,4]},optimizer={'secret':'never-save'})
    for index in range(7):store.publish(index,weights,dict(schema='qqt.kill-actor-critic/v1'))
    assert sorted(p.name for p in tmp_path.glob('u*'))==['u000000','u000005','u000006']
    assert json.loads((tmp_path/'LATEST.json').read_text())['checkpoint']=='u000006'
    assert json.loads((tmp_path/'STORAGE.json').read_text())['checkpoints']==3
    for directory in tmp_path.glob('u*'):
        assert {p.name for p in directory.iterdir()}=={'actor.pt','critic.pt','CHECKPOINT.json'}
        assert pickle.loads((directory/'critic.pt').read_bytes())==weights['critic']
    assert (other/'untouched').read_text()=='keep'


def test_failed_save_preserves_previous_latest(tmp_path):
    store=CheckpointStore(tmp_path,min_free_gib=0)
    weights=dict(actor=[1],critic=[2]);meta=dict(schema='qqt.kill-actor-critic/v1')
    store.publish(0,weights,meta)
    with patch('qqt_rl.training.checkpoints.durable_write',side_effect=OSError('disk full')):
        with pytest.raises(OSError):store.publish(1,weights,meta)
    assert json.loads((tmp_path/'LATEST.json').read_text())['checkpoint']=='u000000'
    assert not (tmp_path/'.u000001.tmp').exists()
    with patch('qqt_rl.training.checkpoints.shutil.disk_usage') as usage:
        usage.return_value.free=0
        with pytest.raises(OSError,match='reserve'):store.publish(1,weights,meta)
    assert (tmp_path/'u000000/actor.pt').exists()


def test_byte_budget_prunes_before_count_limit(tmp_path):
    store=CheckpointStore(tmp_path,keep=64,budget_gib=.00004,min_free_gib=0)
    for index in range(5):
        store.publish(index,dict(actor=b'a'*5000,critic=b'b'*5000),dict(schema='qqt.kill-actor-critic/v1'))
    receipt=json.loads((tmp_path/'STORAGE.json').read_text())
    assert receipt['checkpoint_bytes']<=receipt['budget_bytes']
    assert receipt['checkpoints']<5
    assert (tmp_path/'u000000').exists() and (tmp_path/'u000004').exists()
