"""Freeze opt-in rule/worker copies. Never edit a source used by old runs."""
import json,shutil
from pathlib import Path
from scripts.population_run_state import atomic_json,sha

ROOT=Path('runs/items_alignment_20261107/v2')
READY=Path('../cx_main_best_path_20261106/.hermes/ITEM_RULE_READY.json')


def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def build():
    contract=json.loads(READY.read_text())
    assert contract['version']=='bun-native-v3-items' and contract['status'] in ('engine-ready-key-tests-pass','ready-validated-awaiting-publication','ready','READY')
    webroot=(READY.parent/contract['repository_relative_to_manifest']).resolve()
    canonical=ROOT/'canonical'
    if (ROOT/'CANONICAL_CONTRACT.json').exists():
        previous=json.loads((ROOT/'CANONICAL_CONTRACT.json').read_text())
        if previous['paths_sha256']!=contract['paths_sha256']:
            assert not (ROOT/'ACCEPTED_RULE.json').exists(),'accepted canonical changed; use another version'
            import time
            canonical.rename(ROOT/('canonical_superseded_'+str(time.time_ns())))
    canonical.mkdir(parents=True,exist_ok=True)
    for f,h in contract['paths_sha256'].items():
        assert sha(webroot/f)==h,f
        target=canonical/f;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==h
        else:shutil.copyfile(webroot/f,target)
    # Old engines and their maps remain needed by canonical tests and snapshot.
    for f in ('web/sim-bun-native-v2-face.js','web/sim-bun-native-v2.js'):
        target=canonical/f
        if not target.exists():shutil.copyfile(webroot/f,target)
    for src in (webroot/'web').glob('*.js'):
        target=canonical/'web'/src.name
        if not target.exists():shutil.copyfile(src,target)
    atomic_json(ROOT/'CANONICAL_CONTRACT.json',contract)
    modules={}
    env=Path('jax_bomb/je07_bun_env_r1_items.py').read_text()
    env=once(env,'from . import je07_native_items as native_items','from scripts import cx_items_native_v2 as native_items')
    env=once(env,'native_items.release_deaths(core,tactical_crate,field_item,native_state)',
        'native_items.release_deaths(core,tactical_crate,field_item,native_state,jax.random.fold_in(key_drop,73),bun_loose)')
    env=once(env,'native_items.airdrop(core,tactical_crate,field_item,native_state,jax.random.fold_in(key_drop,72))',
        'native_items.airdrop(core,tactical_crate,field_item,native_state,jax.random.fold_in(key_drop,72),bun_loose)')
    env=once(env,'        state.core, base_actions, key_base, auto_reset=False, return_info=True,',
        '        state.core._replace(gametype=jnp.where(state.scene==9,9,state.core.gametype).astype(jnp.int8)), base_actions, key_base, auto_reset=False, return_info=True,')
    modules['jax_bomb.je07_bun_env_r1_items']=env
    core=Path('jax_bomb/je05_core_env_r1_items.py').read_text()
    core=once(core,'        rate_b = levels.active().rate[jnp.maximum(level_id, 0)]',
        '        rate_b = jnp.where(_gametype==9,0.,levels.active().rate[jnp.maximum(level_id, 0)])')
    modules['jax_bomb.je05_core_env_r1_items']=core
    regions=Path('scripts/combat_regions_mixed_v2.py').read_text()
    regions=once(regions,'    return st, dict(ax, a3sr=meta)',
        "    from scripts.cx_items_rules_v2 import supply_batch\n    st,_item_supply=supply_batch(st,keys,eligible=ax['provenance_source']<0)\n    return st, dict(ax, a3sr=meta)")
    anchor="    code = code.replace(anchor, anchor + '''\n"
    regions=once(regions,anchor,anchor+"""            diagnostics.update(items_reset_ground=(((fst.core.crate>0)|(fst.tactical_crate>0)).sum((-2,-1))*done).sum(),
                items_reset_natural=(done&(fa['provenance_source']<0)).sum(),
                items_reset_bank=(done&(fa['provenance_source']>=0)).sum(),
                items_reset_duel=(done&(fst.scene==9)).sum())
""")
    modules['scripts.combat_regions_mixed_v2']=regions
    duel=Path('scripts/cx_duel_mixed_v3.py').read_text()
    duel=once(duel,"crate=arr('crate',st.core.crate.dtype),","crate=j.zeros_like(st.core.crate),")
    duel=once(duel,'    return st,ids',
        '    from scripts.cx_items_rules_v2 import supply_batch\n    st,_item_supply=supply_batch(st,keys,duel=True)\n    return st,ids')
    duel=once(duel,'    source=original._league_source\n', '''    source=original._league_source
    _anchor='            diagnostics.update(policy_diag)'
    assert source.count(_anchor)==1
    source=source.replace(_anchor,_anchor+"\\n            diagnostics.update(items_reset_ground=(((fst.core.crate>0)|(fst.tactical_crate>0)).sum((-2,-1))*done).sum(),items_reset_natural=(done&(fa['provenance_source']<0)).sum(),items_reset_bank=(done&(fa['provenance_source']>=0)).sum(),items_reset_duel=(done&(fst.scene==9)).sum())")
''')
    modules['scripts.cx_duel_mixed_v3']=duel
    base=Path('scripts/cc_league2_train.py').read_text()
    base=once(base,"    if 'league_bank' in payload:\n", "    if 'league_bank' in payload and not protocol.get('items_full_state_migration',False):\n")
    base=once(base,"str(BASE / 'league' / sys.argv[2])","str(Path(protocol['namespace']) / 'league' / sys.argv[2])")
    base=once(base,"(BASE / 'league' / sys.argv[2] / 'league_snapshots/G.json').exists()",
        "(Path(os.environ['CC_SNAPSHOT_ROOT']) / 'league_snapshots/G.json').exists()")
    base=once(base,'    replacements = {',"""    replacements = {
        '    M = setup()': '    M = setup()\\n    from scripts.cx_items_receipts_v2 import verify_live\\n    verify_live(out,protocol)',""")
    # Even on first load, require the recorded parent seal. No fresh fork,
    # auxiliary reset, rekey, mode remap, or opponent generation replacement.
    anchor="    def operation(p, a, st, ax, keys, uk, bank, gen):'''"
    base=once(base,anchor,"""    if not cps and protocol.get('items_full_state_migration',False):
        assert payload['protocol_sha256']==protocol['items_transition']['parent_protocol_sha256']
        from scripts.cx_items_receipts_v2 import migration_exact
        migration_exact(out,payload,params,adam,states,aux,keys,update_key,bank,generation,manifests)

"""+anchor)
    base=once(base,"exec(compile(source, __file__, 'exec'), scope)",
        "from scripts.cx_items_build_v2 import finish_source\n    exec(compile(finish_source(source), __file__, 'exec'), scope)")
    modules['scripts.cc_league2_train']=base
    for name in ('scripts.cx_duel_train_v3','scripts.cx_ftn2_train_v1'):
        text=Path(name.replace('.','/')+'.py').read_text()
        text=once(text,"Path('scripts/cc_league2_train.py').read_text()","Path(os.environ['CX_ITEMS_BASE_SOURCE']).read_text()")
        if name.endswith('cx_ftn2_train_v1'):
            text=once(text,'if not cps: aux=variant.fork_aux(aux,out)',
                'if not cps and not protocol.get("items_full_state_migration",False): aux=variant.fork_aux(aux,out)')
        modules[name]=text
    sp=Path('scripts/cc_league2_train_sp.py').read_text()
    sp=once(sp,"f'runs/cc_train_20261105/training/league/{sys.argv[2]}'",
        "str(Path(json.loads(Path(sys.argv[1]).read_text())['namespace'])/'league'/sys.argv[2])")
    modules['scripts.cc_league2_train_sp']=sp
    sync=Path('scripts/cx_duel_sync_train_v4.py').read_text()
    sync=sync.replace("str(root.parent/'league/L2_DUEL_SYNC2')","str(root/'league/L2_DUEL_SYNC2')")
    sync=once(sync,'    if cps:\n',"    if cps or p.get('items_full_state_migration',False):\n")
    sync=once(sync,"    L.receipt(out,dict(spec,sil_coef=.5),p)","""    if not cps and p.get('items_full_state_migration',False):
        assert payload['protocol_sha256']==p['items_transition']['parent_protocol_sha256']
        from scripts.cx_items_receipts_v2 import migration_sync
        migration_sync(out,payload,params,adam,states,aux,keys,update_keys,bank,generation,manifests)
    from scripts.cx_items_receipts_v2 import verify_live
    verify_live(out,p)
    L.receipt(out,dict(spec,sil_coef=.5),p)""")
    sync=once(sync,"target={'smoke2':19304,'to8':19310,'formal':p['chunk_start_update']+p['chunk_updates']}[args.phase]",
        "target=p['chunk_start_update']+{'smoke2':2,'to8':8,'formal':p['chunk_updates']}[args.phase]")
    sync=once(sync,"(out/'FRESH_u019304.json').exists()","(out/f\"FRESH_u{p['chunk_start_update']+2:06d}.json\").exists()")
    # Parent namespace is the physical-lock directory for all arms, never a
    # private duplicate lock. Stop gates remain in old training BASE too.
    sync=once(sync,"root.parent/f'GPU{g}.lock'","Path('runs/cc_train_20261105/training')/f'GPU{g}.lock'")
    modules['scripts.cx_duel_sync_train_v4']=sync
    folder=ROOT/'sources';folder.mkdir(exist_ok=True)
    mapping={}
    for name,text in modules.items():
        compile(text,name,'exec');path=folder/(name.replace('.','_')+'.py')
        if path.exists() and path.read_text()!=text:
            assert not (ROOT/'ACCEPTED_RULE.json').exists(),'immutable source '+str(path)
            path.rename(path.with_suffix('.superseded_'+str(__import__('time').time_ns())+'.py'))
            path.write_text(text)
        elif path.exists():assert path.read_text()==text
        else:path.write_text(text)
        mapping[name]=str(path)
    atomic_json(ROOT/'MODULE_MAP.json',mapping)
    return mapping


def finish_source(source):
    from scripts.cx_ftn2_train_v1 import transform_worker
    # DUEL/FTN already transform this worker; other arms need identical bounded
    # phases and next-update restoration tokens, without changing rewards.
    if "parser.add_argument('--until'" not in source:
        source=transform_worker(source)
    return source.replace("BASE/'eval_requests'/","Path(protocol['namespace'])/'eval_requests'/").replace("ruleset='bun-native-v2'","ruleset='bun-native-v3-items'")


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
