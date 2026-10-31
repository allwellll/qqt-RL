"""Frozen A3SR dynamics with explicit full-state source parity and optional PBT RNG."""
from pathlib import Path
from types import ModuleType
from scripts.bomb_threat_a3sr_train import worker_source as original_source


def worker_source():
    source=original_source()
    old="M.collector(spec['rollout_ticks'],n,seed,.10"
    assert old in source
    source=source.replace(old,"M.collector(spec['rollout_ticks'],n,protocol['round14_collector_seed'],.10")
    anchor='    if args.verify_restore:\n'
    hook='''    if source is not None and not cps:
        token=json.loads((source.parent.parent/f'NEXT_STEP_u{start:06d}.json').read_text())
        assert probe(P.one(par),st,ax,keys,start)==token['digest'], 'source next-step parity mismatch'
        assert digest(P.one(adm))==token['Adam_digest']
        assert digest((st,ax,keys))==token['full_environment_digest']
        assert digest(ring)==token['ring_digest']
        preserved=digest((params,adam,st,ax,ring,ring_cursor,fixed,warm_rows))
        old_rng=digest(keys)
        if protocol.get('round14_rekey'):
            keys=jax.random.split(jax.random.PRNGKey(seed),n)[None]
            assert digest(keys)!=old_rng
        assert digest((params,adam,st,ax,ring,ring_cursor,fixed,warm_rows))==preserved
        atomic_json(out/'SOURCE_HANDOFF.json',dict(status='PASS',source=protocol['continuation_source'],
            donor_next_step_exact_before_rekey=True,Actor_Critic_Adam_env_aux_ring_exact=True,
            RNG_rekeyed=bool(protocol.get('round14_rekey')),source_rng_digest=old_rng,
            target_rng_digest=digest(keys),critic_warmup_updates=0,formal_updates_discarded=0,
            lineage=protocol['pbt_lineage'],unix=time.time()))
'''
    assert source.count(anchor)==1
    return source.replace(anchor,hook+anchor)


def main():
    module=ModuleType('a3sr_resume_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__)
    module.main()


if __name__=='__main__':main()
