"""Full-state A3 PBT; donor parity precedes independent target RNG rekey."""
from pathlib import Path
from types import ModuleType


def worker_source():
    source=Path('scripts/bomb_threat_iter2_diagnostics_train.py').read_text()
    source=source.replace("M.collector(spec['rollout_ticks'],n,seed,.10","M.collector(spec['rollout_ticks'],n,protocol['pbt2350_collector_seed'],.10")
    anchor='    if args.verify_restore:\n'
    hook='''    if protocol.get('pbt2350_rekey') and not cps:
        assert source is not None and len(warm_rows)>=20
        token=json.loads((source.parent.parent/f'NEXT_STEP_u{start:06d}.json').read_text())
        assert probe(P.one(par),st,ax,keys,start)==token['digest'], 'donor next-step mismatch before rekey'
        assert digest(P.one(adm))==token['Adam_digest']
        assert digest((st,ax,keys))==token['full_environment_digest']
        assert digest(ring)==token['ring_digest']
        preserved=digest((params,adam,st,ax,ring,ring_cursor,fixed,warm_rows))
        old_rng=digest(keys)
        keys=jax.random.split(jax.random.PRNGKey(seed),n)[None]
        assert digest(keys)!=old_rng
        assert digest((params,adam,st,ax,ring,ring_cursor,fixed,warm_rows))==preserved
        atomic_json(out/'PBT_HANDOFF.json',dict(status='PASS',donor=protocol['continuation_source'],
            donor_next_step_exact_before_rekey=True,Actor_Critic_Adam_env_aux_ring_exact=True,
            preserved_digest=preserved,source_rng_digest=old_rng,target_rng_digest=digest(keys),
            target_seed=seed,critic_warmup_updates=0,formal_updates_discarded=0,
            lineage=protocol['pbt_lineage'],unix=time.time()))
'''
    assert source.count(anchor)==1
    return source.replace(anchor,hook+anchor)


def main():
    module=ModuleType('pbt2350_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__)
    module.main()


if __name__=='__main__':main()
