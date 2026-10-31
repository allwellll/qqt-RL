"""Bounded R1 worker: full old-source parity, explicit ledger migration."""
from pathlib import Path
from types import ModuleType
from scripts.bomb_threat_a3sr_resume_train import worker_source as old_source


def worker_source():
    s=old_source()
    s=s.replace('bomb_threat_a3sr_collector as M','bomb_threat_r1_collector as M')
    s=s.replace("BASE/'iter2/a3sr/cache'","BASE/'iter2/r1_attack_20261030/cache'")
    s=s.replace('seed=args.seed;policy=',"seed=protocol['r1_seed'];policy=")
    anchor='    params=jax.tree.map(j.asarray,params);adam=jax.tree.map(j.asarray,adam)'
    init='''    M.configure(protocol['r1_attack'])
    bm=json.loads((ROOT/protocol['r1_bank_manifest']).read_text())
    assert sha(ROOT/protocol['r1_bank_manifest'])==protocol['r1_bank_manifest_sha256']
    bs,ba=M.banks(bs,ba,bm)
'''
    assert s.count(anchor)==1;s=s.replace(anchor,init+anchor)
    begin=s.index('    if source is not None and not cps:\n')
    end=s.index('    if args.verify_restore:\n',begin)
    hook='''    if source is not None and not cps:
        # Check original donor dynamics/credit, BEFORE new reward metadata.
        from scripts import bomb_threat_a3sr_collector as legacy
        token=json.loads((source.parent.parent/f'NEXT_STEP_u{start:06d}.json').read_text())
        oldchoose=jax.jit(lambda p,s,a,k,r:legacy.choose(p,r,fixed,s,a,k,probe_groups))
        oldstep=jax.jit(lambda s,a,ac,k:legacy.transition(s,a,ac,k,0.))
        ss,aa,kk=jax.tree.map(lambda x:j.asarray(x[0,:1]),(st,ax,keys))
        nk,ke,ac,_=oldchoose(P.one(par),ss,aa,kk,ring[start%4])
        assert digest((nk,ke,ac,oldstep(ss,aa,ac,ke)))==token['digest'], 'donor next-step mismatch'
        assert digest(P.one(adm))==token['Adam_digest']
        assert digest((st,ax,keys))==token['full_environment_digest']
        assert digest(ring)==token['ring_digest']
        preserved=digest((params,adam,st,ring,ring_cursor,fixed,warm_rows))
        old_rng=digest(keys)
        keys=jax.random.split(jax.random.PRNGKey(seed),n)[None]
        assert digest(keys)!=old_rng
        # Source native env/aux remain physical and complete. Only the newly
        # versioned shaping ledger and opponent allocation migrate here.
        kinds=fixed_allocation(n,protocol['round14_collector_seed'],protocol['A1_weights'])
        meta=dict(ax['combat_mix'],kind=j.asarray(kinds,j.int32)[None])
        if protocol['r1_attack']:
            from scripts.bomb_threat_r1_reward import state as credit_state
            meta['bomb_threat']=shard(credit_state(n),1)
        ax=dict(ax,combat_mix=meta)
        # The one-row restore probe must use its post-migration opponent kind.
        new_kind=int(np.asarray(ax['combat_mix']['kind'])[0,0])
        probe_groups.clear()
        probe_groups.update({name:np.array([0]) if new_kind==i else np.array([],int) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]})
        probe_groups['bot']=np.array([0]) if new_kind<3 else np.array([],int)
        assert digest((params,adam,st,ring,ring_cursor,fixed,warm_rows))==preserved
        atomic_json(out/'SOURCE_HANDOFF.json',dict(status='PASS',source=protocol['continuation_source'],
            donor_next_step_exact_before_rekey=True,Actor_Critic_Adam_env_ring_exact=True,
            native_aux_preserved=True,new_reward_ledger_reset=protocol['r1_attack'],opponent_kind_migrated=True,
            RNG_rekeyed=True,source_rng_digest=old_rng,target_rng_digest=digest(keys),
            critic_warmup_updates=0,formal_updates_discarded=0,lineage=protocol['pbt_lineage'],unix=time.time()))
'''
    s=s[:begin]+hook+s[end:]
    # Admission is technical, no unapproved throughput gate on the new reward.
    s=s.replace("        assert batch_size/row['cycle_seconds']>=18000,'A3SR smoke below18000 env/s'", "        assert batch_size/row['cycle_seconds']>0,'no effective environment throughput'")
    anchor="        assert row['exposure']['spawn_eligible']>0,'no eligible real resets in smoke'"
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+"\n        assert min(row['exposure']['bank_seat_ticks'])>0,'no real Bank exposure on both physical seats'\n        assert min(row['exposure']['bank_seat_resets'])>0,'no genuine Bank reset on both physical seats'")
    # Persist anti-hack coefficient changes only at atomic cadence boundaries;
    # no worker signals and no reset of native or optimizer state.
    anchor='        before=P.one(par);began=time.monotonic()'
    hook='''        if protocol['r1_attack'] and (index-1)%40==0:
            request_path=ROOT/protocol['r1_coefficient_request']
            request=json.loads(request_path.read_text()) if request_path.exists() else {}
            factor=float(request.get('factor',1.))
            assert factor in (1.,.5)
            if request.get('protocol_id')=='A3SR-ATTACK-R1-20261030':
                meta=dict(ax['combat_mix']);ledger=dict(meta['bomb_threat'])
                ledger['r1_scale']=j.full_like(ledger['r1_scale'],factor)
                ax=dict(ax,combat_mix=dict(meta,bomb_threat=ledger))
                atomic_json(out/'COEFFICIENT_APPLIED.json',dict(factor=factor,update_boundary=index-1,
                    request_sha256=sha(request_path),model_optimizer_RNG_preserved=True,unix=time.time()))
'''
    assert s.count(anchor)==1;s=s.replace(anchor,hook+anchor)
    return s


def main():
    module=ModuleType('r1_training_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__);module.main()

if __name__=='__main__':main()
