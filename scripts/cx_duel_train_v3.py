"""Independent private mixed-task trainer; upstream seals and GPU locks retained."""
import json,sys
from pathlib import Path


def replace_once(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


def transform_worker(source):
    from scripts.cx_ftn2_train_v1 import transform_worker as bounded
    source=bounded(source)
    source=replace_once(source,'    M = setup()',
        '    M = setup()\n    from scripts.cx_duel_lib_v3 import receipt as duel_receipt\n    duel_receipt(out,spec,protocol)')
    source=replace_once(source,"BASE/'eval_requests'/f'G_u{index:06d}.json'","BASE/'eval_requests'/f'{args.arm}_u{index:06d}.json'")
    source=replace_once(source,'from scripts.cc_sil_updater_v1 import updater as sil_updater',
        'from scripts.cx_duel_updater_v1 import updater as sil_updater')
    # Actual32 minibatches: keep counts, raw signed GAE, normalized signed GAE.
    source=replace_once(source,'categorical=True,dagger=0,combat_kl=0,unix=time.time())',
        'categorical=True,dagger=0,combat_kl=0,minibatch_signed_credit=np.asarray(loss)[:,12:].tolist(),unix=time.time())')
    return source


def main():
    p=json.loads(Path(sys.argv[1]).read_text());assert sys.argv[2] in p['arms']
    assert p['duel']['mix'][1] in (.25,.5,.75) and p['duel']['mask'] is False
    import os
    os.environ['DUEL_FRACTION']=str(p['duel']['mix'][1])
    source=Path('scripts/cc_league2_train.py').read_text()
    source=source.replace('combat_regions_mixed_v2 as mixed','cx_duel_mixed_v3 as mixed')
    source=source.replace('cc_league2_lib as snapshots','cx_duel_lib_v3 as snapshots')
    source=source.replace('cc_league2_lib as league2','cx_duel_lib_v3 as league2')
    anchor="    scope = {'__name__': 'cc_league2_worker', '__file__': __file__}"
    source=replace_once(source,anchor,'    source=transform_worker(source)\n'+anchor)
    scope={'__name__':'duel_private_worker','__file__':__file__,'transform_worker':transform_worker}
    exec(compile(source,__file__,'exec'),scope)
    anchor='            selected, source_manifest = next_generation(start)'
    scope['FORK']=replace_once(scope['FORK'],anchor,
        '            keys=jax.vmap(lambda k:jax.random.fold_in(k,args.seed))(keys)\n'
        '            update_key=jax.random.fold_in(update_key,args.seed)\n'
        '            states,aux,keys=league2.fork(mixed,keys,bs,ba,spawns,args.seed)\n'+anchor)
    scope['main']()


if __name__=='__main__':main()
