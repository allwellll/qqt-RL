"""Release verification must reject corrupted or incompatible native weights."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.export_bun_release import export, validate_native, validate_reference
from scripts.check_publish_privacy import findings, scan


def test_native_contract_rejects_adapter_and_wrong_heads():
    p = dict(tok=(np.zeros((216, 392)), np.zeros(392)),
             state_w=np.zeros((24, 392)), blocks=[{}] * 4,
             heads=dict(wm=(np.zeros((392, 5)), np.zeros(5)),
                        wb=(np.zeros((392, 3)), np.zeros(3))))
    validate_native(p)
    p['heads']['wb'] = (np.zeros((392, 2)), np.zeros(2))
    with pytest.raises(ValueError, match='ability'):
        validate_native(p)


def test_public_reference_is_repository_relative():
    assert validate_reference('runs/example/gen003072') == 'runs/example/gen003072'
    for ref in ['/private/model', '../model', 'runs/../../model', 'C:\\private', 'https://host/model']:
        with pytest.raises(ValueError):
            validate_reference(ref)


def test_privacy_scope_ignores_untracked_local_files(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'public.txt').write_text('relative/source\n')
    (tmp_path / 'AGENTS.md').write_text('example-user /private/example-user\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'public.txt'], check=True)
    assert scan(tmp_path, ['example-user']) == []
    (tmp_path / 'public.txt').write_text('EXAMPLE-USER\n')
    assert scan(tmp_path, ['example-user'])
    assert scan(tmp_path, ['example-user'], ['AGENTS.md'])


def test_privacy_detects_machine_data_but_allows_runtime_pid():
    assert findings('owner PID ' + str(123456), [])
    private_ip = '.'.join(['10', '1', '2', '3'])
    assert findings(private_ip, [])
    assert not findings('pid = os.getpid(); pid0 is a player', [])


def test_corruption_is_rejected_before_unpickling(tmp_path):
    (tmp_path / 'policy.pt').write_bytes(b'not a pickle')
    (tmp_path / 'COMMITTED.json').write_text(json.dumps(dict(passed=True, policy_sha256='0'*64)))
    with pytest.raises(ValueError, match='hash'):
        export(tmp_path, tmp_path/'out', 'runs/example', 'raw', '2026-10-12T00:00:00Z', 'test')


def test_model_packed_numbers_are_not_pid_metadata(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    doc=dict(meta=dict(arch='transformer'), tensors={}, flat='PID'+str(12345))
    p=tmp_path/'model.json';p.write_text(json.dumps(doc))
    assert not scan(tmp_path,['example-user'],['model.json'])
    doc['meta']['source']='owner PID '+str(123456)
    p.write_text(json.dumps(doc))
    assert scan(tmp_path,['example-user'],['model.json'])


def test_web_export_fixed_timestamp_is_byte_reproducible(tmp_path):
    import pickle
    from scripts import export_bun_web_model
    e = 4
    p = dict(tok=(np.zeros((216,e)), np.zeros(e)), pos=[np.zeros((26,e))],
             state_w=np.zeros((24,e)),state_b=np.zeros(e),blocks=[],
             heads={k:(np.zeros((e,n)),np.zeros(n)) for k,n in [('wm',5),('wb',3),('wv',128)]})
    source = tmp_path/'policy.pt';source.write_bytes(pickle.dumps(p))
    targets = [tmp_path/'one.json',tmp_path/'two.json']
    for t in targets:
        subprocess.run([__import__('sys').executable, export_bun_web_model.__file__, str(source),
                        '--output',str(t),'--generated-at','2026-10-12T00:00:00Z'],check=True)
    assert targets[0].read_bytes() == targets[1].read_bytes()
