from scripts.r1_shared_dispatch import authorized_cotenants
from scripts import r1_shared_dispatch as m
from unittest.mock import patch


def test_registered_partner_only():
    partner = {'c': {'gpu': 7, 'run': 'r2c', 'worker_module': 'custody'}}
    entry = {'r1_items_applied': True}
    assert authorized_cotenants(7, entry, partner, [('custody', 'r2c')])
    assert not authorized_cotenants(7, entry, partner, [('custody', 'other')])
    assert not authorized_cotenants(7, entry, partner, [('unknown', 'r2c')])
    assert not authorized_cotenants(7, entry, partner, [('custody', 'r2c'), ('other', 'job')])
    assert not authorized_cotenants(7, entry, partner, [])


def test_unrelated_slots_not_granted_shared_dispatch():
    partner = {'c': {'gpu': 7, 'run': 'r2c', 'worker_module': 'custody'}}
    for slot in (0, 4, 5):
        assert not authorized_cotenants(slot, {'r1_items_applied': True}, partner, [('custody', 'r2c')])
    assert not authorized_cotenants(7, {}, partner, [('custody', 'r2c')])


def test_real_worker_argument_uses_run_not_protocol_suffix(tmp_path, monkeypatch):
    path=tmp_path/'R2_SLOTS.json'
    path.write_text('{"c":{"gpu":7,"run":"r2c","worker_module":"custody"}}')
    original=m.Path
    class FakePath:
        def __new__(cls, *args):
            if args==('/proc','123','cmdline'):
                return type('Cmdline',(),{'read_bytes':lambda _:b'python\0-m\0custody\0runs/bun_native_v2_20261030/protocols/r2c_performance02.json\0r2c\0seed\0'})()
            if args==('/proc','123','cwd'):
                return type('Cwd',(),{'resolve':lambda _:original(m.__file__).resolve().parents[1]})()
            return original(*args)
    monkeypatch.setattr(m,'Path',FakePath)
    with patch.object(m.subprocess,'check_output',side_effect=['7, uuid\n','uuid, 123\n']):
        assert m.shared_free_slots({'7':{'r1_items_applied':True}},path)=={7}
