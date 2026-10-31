"""Same-version R1 continuation; never repeat the original rule migration."""
from pathlib import Path
from types import ModuleType


def worker_source(preserved=False):
    if preserved:
        from scripts.bomb_threat_r1_preserved_train import worker_source as original
    else:
        from scripts.bomb_threat_r1_progress_train import worker_source as original
    s = original()
    if not preserved:
        begin = s.index('        if not cps:\n            exported=')
        end = s.index("        params=payload['params']", begin)
        s = s[:begin] + "        payload=pickle.load((cp/'state.pkl').open('rb'))\n" + s[end:]
    begin = s.index('    if not cps:\n        ')
    end = s.index('    # Initialize cached safety', begin)
    s = s[:begin] + s[end:]
    anchor = '    if args.verify_restore:\n'
    handoff = '''    if not cps:
        expected=json.loads((source.parent.parent/f'NEXT_STEP_u{start:06d}.json').read_text())
        assert probe(P.one(par),st,ax,keys,start)==expected['digest'],'same-version donor next-step mismatch'
        assert digest(P.one(adm))==expected['Adam_digest']
        assert digest((st,ax,keys))==expected['full_environment_digest']
        assert digest(ring)==expected['ring_digest']
        atomic_json(out/'SOURCE_HANDOFF.json',dict(status='PASS',source=protocol['continuation_source'],
            same_version_next_step_exact=True,Actor_Critic_Adam_env_aux_RNG_ring_exact=True,
            RNG_rekeyed=False,reward_ledger_reset=False,critic_warmup_updates=0,
            formal_updates_discarded=0,unix=time.time()))
'''
    assert s.count(anchor) == 1
    return s.replace(anchor, handoff + anchor)


def main():
    import json, os, sys
    protocol = json.loads(Path(sys.argv[1]).read_text())
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],
        JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bomb_threat_r1_items_runtime import install
    install()
    module = ModuleType('r1_same_version_worker')
    module.__file__ = str(Path(__file__))
    exec(compile(worker_source(protocol.get('r1_preserved_legacy', False)), __file__, 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    main()
