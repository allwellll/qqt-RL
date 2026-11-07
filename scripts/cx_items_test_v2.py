"""Bounded CPU admission: actual web engine parity and native integration."""
import json,os,subprocess,time,sys
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.cx_items_build_v2 import ROOT,build


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    modules=build()
    os.environ['JAXBOMB_BUN_LEVELS']=str((ROOT/'canonical/web/assets/maps/levels.json').resolve())
    subprocess.run(['node','web/test_item_rules.js'],cwd=ROOT/'canonical',check=True,timeout=120)
    from scripts.cx_items_install_v2 import Finder
    sys.meta_path.insert(0,Finder(modules))
    from scripts.combat_specialist_g_league_train import setup
    setup()
    import jax,jax.numpy as j,numpy as np
    from scripts import cx_items_rules_v2 as R
    from jax_bomb import je07_bun_env as env,je07_native_items as old,levels
    from scripts.terminal_checkpoint_restore import digest
    assert jax.default_backend()=='cpu'
    started=time.monotonic();folder=ROOT/'tests';folder.mkdir(exist_ok=True)
    st=env._fresh(jax.random.PRNGKey(2026771200),jax.nn.one_hot(0,16))
    z=j.zeros((13,15),bool);zi=j.zeros((13,15),j.int8)
    st=st._replace(core=st.core._replace(wall=z,brick=z,pushable=z,crate=zi,rec_crate=z,
        fuse=j.zeros((13,15),j.int16),blast_linger=zi,brick_linger=zi,
        pos=j.array([[3.95,3.95],[10.5,12.5]]),alive=j.array([True,True])),
        tactical_crate=zi,field_item=zi,bun_loose=j.zeros((13,15,2),j.int8),native_items=old.fresh())
    rng=np.random.default_rng(2026771201);cases=[];expected=[]
    def case(state,kind,draws,**kw):
        core=state.core;native=state.native_items
        fields=dict(wall=core.wall,brick=core.brick,pushable=core.pushable,cover=native.cover,
            fieldItem=state.field_item,fuse=core.fuse,blastLinger=core.blast_linger,
            brickLinger=core.brick_linger,bunLoose=state.bun_loose)
        value=dict(kind=kind,seed=7,pos=core.pos,alive=core.alive,crate=core.crate,tactical=state.tactical_crate,
            counts=native.counts,fields=fields,falls=np.flatnonzero(np.asarray(native.fall_type)>=0),draws=draws,**kw)
        cases.append(jax.tree.map(lambda x:np.asarray(x).tolist() if hasattr(x,'shape') else x,value))
    release=jax.jit(lambda s,k,d:R.release_deaths(s.core,s.tactical_crate,s.field_item,s.native_items,k,s.bun_loose,d))
    # All legal blockers and fractional player AABBs; then shortage and exact retry.
    blocked=st._replace(core=st.core._replace(wall=z.at[0,0].set(True),brick=z.at[0,1].set(True),
        pushable=z.at[0,2].set(True),fuse=st.fuse.at[0,3].set(2),blast_linger=zi.at[0,4].set(1),brick_linger=zi.at[0,5].set(2)),
        field_item=zi.at[0,6].set(1),bun_loose=st.bun_loose.at[0,7,0].set(1),
        native_items=old.fresh(z.at[0,8].set(True))._replace(fall_type=old.fresh().fall_type.at[9].set(0)))
    for state in (st,blocked,st._replace(core=st.core._replace(wall=j.ones_like(z))),
                  st._replace(core=st.core._replace(wall=j.ones_like(z).at[12,14].set(False)))):
        drops=j.array([[0,0,3,1,1],[1,1,2,1,1],[2,0,7,1,1],[3,0,4,1,1],[4,0,6,1,1]],j.int16)
        state=state._replace(native_items=state.native_items._replace(pending=state.native_items.pending.at[:5].set(drops),pending_size=j.int32(5)))
        draws=j.asarray(rng.random(1024),j.float32);core,ta,ns=release(state,jax.random.PRNGKey(1),draws)
        case(state,'death',draws,drops=drops)
        expected.append((R.available(state.core,state.tactical_crate,state.field_item,state.native_items,state.bun_loose),core.crate,ta,ns))
        assert int(ns.counts.sum()+ns.pending[:int(ns.pending_size),2].sum())==22
        assert int(ns.overflow)==0
        shift=state._replace(core=state.core._replace(pos=state.core.pos),native_items=state.native_items._replace(pending=state.native_items.pending.at[:5,3:].set(12)))
        sc,sta,sns=release(shift,jax.random.PRNGKey(1),draws)
        assert digest((sc,sta,sns.counts))==digest((core,ta,ns.counts))
    # Exact supply/type/super distribution under a common random tape.
    supply=jax.jit(lambda s,k,rate,sup,tf,en,ds:R.initial_supply(s,k,rate,sup,tf,en,ds))
    for rate,en,tf,brick in ((.582011,True,.3,True),(0.,True,.3,False),(.582011,False,.3,True),(0.,True,0.,False)):
        state=st
        if brick:
            mask=j.zeros_like(z).at[:6,:].set(True).at[3:5,3:5].set(False)
            state=st._replace(core=st.core._replace(brick=mask))
        draws=j.asarray(rng.random((195,4)),j.float32)
        after,summary=supply(state,jax.random.PRNGKey(2),rate,.25,tf,en,draws)
        case(state,'supply',draws,rate=rate,enabled=en,super_fraction=.25,tactical_fraction=tf)
        expected.append((R.available(state.core,state.tactical_crate,state.field_item,state.native_items,state.bun_loose),after.core.crate,after.tactical_crate,after.native_items,summary))
    atomic_json(folder/'PARITY_INPUT.json',dict(engine=str((ROOT/'canonical/web/sim-bun-native-v3-items.js').resolve()),cases=cases))
    subprocess.run(['node','model_evaluation.js',_js_config(folder)],check=True,timeout=120)
    outputs=json.loads((folder/'PARITY_OUTPUT.json').read_text())
    for index,(actual,exp) in enumerate(zip(outputs,expected)):
        for label,a,b in [('legal',actual['legal'],exp[0]),('crate',actual['crate'],exp[1]),('tactical',actual['tactical'],exp[2])]:
            assert np.array_equal(np.asarray(a).reshape(-1),np.asarray(b).reshape(-1)),(index,label)
        pending=exp[3].pending[:int(exp[3].pending_size),:3]
        assert np.array_equal(np.asarray(actual['pending']).reshape(-1,3),np.asarray(pending)),index
        # JS stores 1 for implicit new growth/banana and 3 for implicit glue;
        # native count0 encodes exactly those defaults (not a quantity loss).
        native_counts=np.asarray(exp[3].counts)
        occupied=np.asarray(exp[1])>0; tactical=np.asarray(exp[2])
        quantity=np.where(native_counts>0,native_counts,np.where(tactical==2,3,np.where(occupied|(tactical>0),1,0)))
        js_counts=np.asarray(actual['counts']).reshape(13,15)
        js_tactical=np.asarray(actual['tactical']).reshape(13,15)
        js_effective=np.where(js_counts>0,js_counts,np.where(js_tactical==2,3,np.where((np.asarray(actual['crate']).reshape(13,15)>0)|(js_tactical>0),1,0)))
        assert np.array_equal(quantity,js_effective),(index,'counts')
        if len(exp)==5:
            fields=('baselineInitial','baselineBricks','brickRate','expectedBaseline','requestedExtra','added','unplaced','initialActual','expectedTotal')
            assert np.allclose([actual['supply'][f] for f in fields],exp[4],rtol=1e-6,atol=1e-5)
    # Full-map uniformity, no replacement, same key reproducibility.
    free=R.available(st.core,st.tactical_crate,st.field_item,st.native_items,st.bun_loose)
    n=int(free.sum());u=(j.arange(n)+.5)/n
    select=jax.jit(jax.vmap(lambda u:R.choose(free,u)))
    assert np.array_equal(np.sort(select(u)),np.flatnonzero(free))
    sample=np.asarray(select(jax.random.uniform(jax.random.PRNGKey(2026771202),(131072,))))
    hist=np.bincount(sample,minlength=195)[np.asarray(free).reshape(-1)]
    chi=float(((hist-len(sample)/n)**2/(len(sample)/n)).sum());assert chi<n-1+8*np.sqrt(2*(n-1))
    # Actual native death producer, slots and cap behavior retained.
    producer=st._replace(core=st.core._replace(bombs_cap=j.array([5.,2.]),blast_cap=j.array([4.,2.]),
        spd_g=j.array([1.3+2*env.base.GROWTH_SPEED_STEP,1.3])))
    ns,co,sl,ct=old.death_inventory(producer.native_items,producer.core,j.array([True,False]),
        j.zeros((2,7),j.int8).at[0,:2].set(j.array([1,2],j.int8)),j.zeros((2,7),j.int16).at[0,:2].set(j.array([4,6],j.int16)),
        j.array([2.,2.,1.3]),env.base.GROWTH_SPEED_STEP)
    assert ns.pending[:int(ns.pending_size),2].sum()==17 and not np.asarray(sl).any() and not np.asarray(ct).any()
    assert np.allclose(np.asarray([co.bombs_cap[0],co.blast_cap[0],co.spd_g[0]]),[2,2,1.3])
    dead=st._replace(core=st.core._replace(alive=j.array([False,True])),native_items=ns)
    ds=j.asarray(rng.random(1024),j.float32)
    moved=dead._replace(core=dead.core._replace(pos=dead.pos.at[0].set(j.array([11.5,.5]))))
    assert digest(release(dead,jax.random.PRNGKey(6),ds)[1:])==digest(release(moved,jax.random.PRNGKey(6),ds)[1:])
    # Newly burning or player-occupied airdrop landings retain the entire
    # typed stack in the graveyard; a safe own reservation permits landing.
    ns=old.fresh()._replace(fall_type=old.fresh().fall_type.at[80].set(3),
        fall_count=old.fresh().fall_count.at[80].set(6))
    air=jax.jit(lambda c,n:env.native_items.airdrop(c,zi,zi,n,jax.random.PRNGKey(7),st.bun_loose))
    safe=st.core._replace(t=j.int32(3))
    landed=air(safe,ns);assert landed[1][5,5]==1 and landed[2].counts[5,5]==6
    fire=air(safe._replace(blast_linger=zi.at[5,5].set(1)),ns)
    assert fire[1][5,5]==0 and fire[2].grave_size==1 and fire[2].grave[0,2]==6
    crowded=air(safe._replace(pos=safe.pos.at[0].set(j.array([5.5,5.5]))),ns)
    assert crowded[1][5,5]==0 and crowded[2].grave[0,2]==6
    caps=j.array([10.,8.,2.4]);base=j.array([2.,2.,1.3])
    for typ in range(3):
        co=st.core._replace(pos=st.pos.at[0].set(j.array([5.5,5.5])),
            bombs_cap=j.array([10.,2.]),blast_cap=j.array([8.,2.]),spd_g=j.array([2.4,1.3]),
            crate=zi.at[5,5].set(typ+1))
        pickup=jax.jit(lambda c:old.pickup(c,zi,j.zeros(2,j.int16),j.zeros((2,7),j.int8),j.zeros((2,7),j.int16),old.fresh(),jax.random.PRNGKey(8),caps,env.base.GROWTH_SPEED_STEP))
        got=pickup(co)
        assert np.allclose(np.asarray([got[0].bombs_cap[0],got[0].blast_cap[0],got[0].spd_g[0]]),caps)
        assert int(got[4].grave_size)==1
    # Actual natural map and actual private DUEL catalog, not synthetic claims.
    natural=env._fresh(jax.random.PRNGKey(3),jax.nn.one_hot(0,16));table=levels.active();idx=env._BUN_INDEX
    newer,nat=supply(natural,jax.random.PRNGKey(4),table.rate[idx],table.super_f[idx],.3,True,j.asarray(rng.random((195,4)),j.float32))
    assert int(nat[5])==24 and int(nat[7])==24 and np.isclose(float(nat[3]),47.142891)
    from scripts import cx_duel_mixed_v3 as D
    bat=jax.tree.map(lambda x:j.stack([x]*4),natural)
    duel,ids=jax.jit(lambda ks:D.scenario_state(bat,ks,j.array([0,1,0,1]),j.zeros(4,j.int32)))(jax.random.split(jax.random.PRNGKey(5),4))
    counts=((duel.core.crate>0)|(duel.tactical_crate>0)).sum((-2,-1))
    assert np.all(np.asarray(counts)==15),counts
    # Construct final collector and check actual ABI before any GPU switch.
    backend=D.bun_backend();assert 'items_reset_ground' in backend._league_source
    sources={str(p):sha(p) for p in (ROOT/'sources').glob('*.py')}
    sources.update({str(p):sha(p) for p in Path('scripts').glob('cx_items_*v2.*')})
    sources.update({str(p):sha(p) for p in (ROOT/'canonical').rglob('*') if p.is_file()})
    contract=ROOT/'CANONICAL_CONTRACT.json'
    accepted=dict(status='PASS_ACTUAL_CANONICAL_WEB_PARITY',contract_ready=True,rule_version=R.VERSION,
        canonical_contract_sha256=sha(contract),canonical_engine_sha256=sha(ROOT/'canonical/web/sim-bun-native-v3-items.js'),
        source_sha256=sources,private_modules=modules,quantity_policy='preserve_typed_stacks_queue_if_no_cell',
        death_drop_policy='uniform_full_legal_map_without_replacement',supply_transform='initial_global_topup_original_brick_probability',
        supply_summary_fields=['initial','bricks','brick_probability','baseline_expected','requested','added','unplaced','initial_actual','expected_total'],
        natural_supply=np.asarray(nat).tolist(),duel_initial_actual=np.asarray(counts).tolist(),
        actual_canonical_cases=len(cases),global_uniform_samples=len(sample),chi_square=chi,
        checkpoint_schema_unchanged=True,unchanged_caps=[10,8,2.4],
        parity='exact common uniform tape for placement/type/super/quantity and legal masks; independent saved JAX PRNG distribution',
        canonical_full_npm_status=json.loads(contract.read_text())['tests']['full_status'],elapsed_seconds=time.monotonic()-started)
    atomic_json(ROOT/'ACCEPTED_RULE.json',accepted)
    print(json.dumps({k:v for k,v in accepted.items() if k not in ('source_sha256','private_modules')}),flush=True)


def _js_config(folder):
    from scripts.model_launch import manifest
    return manifest(dict(script='scripts/cx_items_parity_v2.js',args=[str(folder/'PARITY_INPUT.json'),str(folder/'PARITY_OUTPUT.json')]))


if __name__=='__main__':main()
