"""Independent G league worker, full state and immutable episode opponents."""
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

BASE = Path('runs/combat_specialist_league_20261101')


def completion_receipt(receipt, protocol):
    """Finalize already committed work without duplicating timestamp keywords."""
    return {**receipt, 'completion_status': 'COMPLETE',
            'updates_in_this_phase': int(protocol['chunk_updates']),
            'phase1_bootstrap_updates': 2 if protocol['chunk_start_update'] == 4644 else 0,
            'unix': time.time()}


def setup():
    from scripts import r4_k4_runtime as runtime
    runtime.prepare()
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    from scripts.bun_spawn_side import install as ownside, install_collector
    ownside()
    from scripts import combat_item_engine as T, r3_items_collector as M
    from scripts import ppo_scale_longrun_train as P
    T.configure(); P.C.install_provenance(); M.configure(True,True); install_collector(M)
    runtime.native_safety()
    from scripts.bomb_threat_r1_progress_reward import native_safety
    native_safety()
    return M


def atomic_pickle(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=5); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('protocol'); parser.add_argument('arm'); parser.add_argument('seed', type=int)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--verify-restore', action='store_true')
    args = parser.parse_args(); protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text()); spec = {**protocol['common'], **protocol['arms'][args.arm]}
    out = Path(protocol['namespace'])/args.arm; out.mkdir(parents=True, exist_ok=True)
    guard = lock(out/'worker.lock')
    for path, expected in protocol['source_sha256'].items(): assert sha(path) == expected, path
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'], JAX_PLATFORMS='cuda',
        JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='true',
        JAX_COMPILATION_CACHE_DIR=str(BASE/'cache/G-league-v1'), JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0')
    import jax
    import jax.numpy as j
    import numpy as np
    import optax
    from scripts import combat_specialist_g_mixed as mixed, r4_k4_observation as mode
    from scripts.r4_k4_compilation import CompiledOperator
    from scripts.terminal_checkpoint_restore import digest
    from scripts import combat_specialist_snapshots as snapshots
    M = setup()
    assert jax.default_backend() == 'gpu' and jax.local_device_count() == 1
    assert spec['dagger'] == spec['combat_kl'] == 0 and spec['epochs'] == 1
    bm = Path(protocol['r1_bank_manifest']); assert sha(bm) == protocol['r1_bank_manifest_sha256']
    bs, ba = M.banks(None, None, json.loads(bm.read_text()))
    sp = Path(protocol['spawn_manifest']); assert sha(sp) == protocol['spawn_manifest_sha256']
    spawns = j.asarray(json.loads(sp.read_text())['spawn_cells'])
    n = spec['envs_per_device']; ticks = spec['rollout_ticks']; steps = n*ticks
    optimizer = optax.adam(spec['lr'], eps=1e-8)
    update = CompiledOperator(mixed.updater(spec['global_minibatch'], optimizer))
    collect = CompiledOperator(jax.jit(mixed.collector(ticks, n, args.seed, spawns), keep_unused=True))
    cps = sorted(out.glob('u*/COMMITTED.json'))
    cp = cps[-1].parent if cps else Path(protocol['continuation_source']['checkpoint'])
    cm = json.loads((cp/'COMMITTED.json').read_text())
    assert sha(cp/'state.pkl') == cm['state_sha256'] and sha(cp/'policy.pt') == cm['policy_sha256']
    with (cp/'state.pkl').open('rb') as stream: payload = pickle.load(stream)
    if 'league_bank' in payload:
        if cps:
            assert payload['protocol_sha256'] in [sha(protocol_path)]+protocol.get('compatible_checkpoint_protocol_sha256',[])
        params, adam, states, aux, keys, bank = jax.tree.map(j.asarray,
            (payload['params'], payload['adam'], payload['states'], payload['aux'], payload['keys'], payload['league_bank']))
        start = payload['update']; update_key = j.asarray(payload['optimizer_key'])
        generation = int(payload['league_generation']); manifests = payload['league_manifests']
    else:
        params, adam = mode.migrate(jax.tree.map(j.asarray, (payload['params'], payload['adam'])))
        start = int(protocol['chunk_start_update']); update_key = jax.random.PRNGKey(args.seed ^ 0x1EA6)
        keys = jax.random.split(jax.random.PRNGKey(args.seed), n)
        states, aux, keys = jax.jit(lambda ks:mixed.initialize(ks, bs, ba, spawns, args.seed))(keys)
        generation = 0; manifests = {}; bank = {}
    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])

    def next_generation(index):
        actors = {}; provenance = {}
        for family in ('G', 'C'):
            manifest = snapshots.load(family); provenance[family] = manifest
            ring = manifest['ring'][int(np.random.default_rng(args.seed ^ (index//40)).integers(len(manifest['ring'])))]
            for name, item in [('current', manifest['current']), ('ring', ring), ('fixed', manifest['fixed'][0])]:
                actors[family+name] = mode.migrate(jax.tree.map(j.asarray, snapshots.actor(item)))
        return actors, provenance

    if not cps and 'league_bank' not in payload:
        selected, source_manifest = next_generation(start)
        bank = jax.tree.map(lambda a:j.stack([a, a]), selected)
        manifests = {'0':source_manifest, '1':source_manifest}
        atomic_json(out/'INIT.json', dict(status='PASS', parent=protocol['continuation_source'],
            Actor_Critic_Adam_old_rows_retained=True, new_mode_row_zero=True,
            global_dim=38, natural_episode_restart=True, discarded_optimizer_updates=0,
            new_independent_seed=args.seed, phase1_ability_count=3, unix=time.time()))

    def operation(p, a, st, ax, keys, uk, bank, gen):
        (st, ax, keys), batch, exposure = collect(p, st, ax, keys, bs, ba, bank, j.asarray(gen,j.int32))
        uk, k = jax.random.split(uk)
        (p, a), loss = update(p, a, batch, k)
        return p, a, st, ax, keys, uk, loss, exposure

    def save(index, row):
        cp = out/f'u{index:06d}'; cp.mkdir(parents=True, exist_ok=True)
        value = jax.device_get(dict(params=params, adam=adam, states=states, aux=aux, keys=keys,
            optimizer_key=update_key, update=index, environment_steps=env_steps, optimizer_steps=opt_steps,
            league_bank=bank, league_generation=generation, league_manifests=manifests,
            protocol_sha256=sha(protocol_path), phase1_ability_count=3, seed=args.seed))
        atomic_pickle(cp/'state.pkl', value); atomic_pickle(cp/'policy.pt', value['params']['actor'])
        atomic_pickle(cp/'policy_bun_mode0.pt', mode.bun_mode_zero_actor(value['params']['actor']))
        receipt = dict(status='PASS', passed=True, checkpoint=str(cp), update=index,
            state_sha256=sha(cp/'state.pkl'), policy_sha256=sha(cp/'policy.pt'),
            bun_mode_zero_policy_sha256=sha(cp/'policy_bun_mode0.pt'),
            environment_steps=env_steps, optimizer_steps=opt_steps, metrics=row, unix=time.time())
        atomic_json(cp/'COMMITTED.json', receipt)
        if (out/'RESTORE.json').exists(): snapshots.publish('G', str(cp))
        added = index-int(protocol['campaign_start_update'])
        if added%80 == 0 or index == int(protocol['chunk_start_update'])+int(protocol['chunk_updates']):
            atomic_json(BASE/'eval_requests'/f'G_u{index:06d}.json',dict(**receipt,
                ruleset='bun-native-v2', policy_bun_mode0=str(cp/'policy_bun_mode0.pt'),
                modes=['argmax','categorical'], seats=[0,1], dev=True,
                full_holdout=added%400==0 or added==1280, teacher=False, async_only=True))
        return cp, receipt

    def finite(tree):
        return all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(tree)
                   if np.issubdtype(np.asarray(a).dtype,np.number))

    def stage(name,index):
        atomic_json(out/'CYCLE_STAGE.json',dict(stage=name,update=index,unix=time.time()))

    if args.verify_restore:
        stage('RESTORE_FULL_NEXT_UPDATE',start)
        expected=json.loads((cp/'NEXT_UPDATE.json').read_text())
        result=operation(params,adam,states,aux,keys,update_key,bank,generation);jax.block_until_ready(result)
        assert digest(result)==expected['digest'], 'full next collector/GAE/Adam recovery mismatch'
        atomic_json(out/'RESTORE.json',dict(status='PASS',passed=True,fresh_process=True,
            checkpoint=str(cp),full_Actor_Critic_Adam_env_aux_RNG_league_exact=True,
            next_operator_digest=expected['digest'],diagnostic_steps_not_counted=steps,unix=time.time()))
        snapshots.publish('G', str(cp))
        return

    total=int(protocol['chunk_start_update'])+int(protocol['chunk_updates'])
    target=min(total,int(protocol['chunk_start_update'])+2) if args.smoke else total
    if not args.smoke: assert (out/'SMOKE.json').exists() and (out/'RESTORE.json').exists()
    rows=[]; last=cm.get('metrics',{}); receipt=cm
    for index in range(start+1,target+1):
        if any((base/name).exists() for base in (BASE,out) for name in ('USER_STOP','USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json')):
            cp,receipt=save(index-1,dict(user_paused=True)); return
        added=index-int(protocol['chunk_start_update'])
        counted=index-int(protocol['campaign_start_update'])
        if added>1 and (counted-1)%40==0:
            dest=1-generation
            # A 40x128tick period exceeds a complete1200tick game. Do not
            # overwrite any snapshot still attached to a real unfinished game.
            active=np.asarray(aux['combat_mix']['opponent_generation'])
            assert not np.any(active==dest),'old snapshot generation still in use'
            selected,provenance=next_generation(index)
            bank=jax.tree.map(lambda old,new:old.at[dest].set(new),bank,selected)
            manifests[str(dest)]=provenance;generation=dest
        before=jax.device_get(params);began=time.monotonic();stage('COLLECTOR',index)
        (states,aux,keys),batch,exposure=collect(params,states,aux,keys,bs,ba,bank,j.asarray(generation,j.int32))
        jax.block_until_ready(batch);rollout=time.monotonic()-began;stage('OPTIMIZER',index)
        update_key,k=jax.random.split(update_key)
        (params,adam),loss=update(params,adam,batch,k);jax.block_until_ready(params)
        assert finite((params,adam,loss)),'nonfinite league PPO'
        delta={branch:float(optax.global_norm(jax.tree.map(lambda a,b:a-b,params[branch],before[branch]))) for branch in ('actor','critic')}
        ability=float(optax.global_norm(jax.tree.map(lambda a,b:a-b,
            params['actor']['heads']['wb'],before['actor']['heads']['wb'])))
        assert min(delta.values())>0 and ability>0,'zero real PPO update'
        env_steps+=steps;opt_steps+=steps//spec['global_minibatch'];elapsed=time.monotonic()-began
        last=dict(passed=True,finite=True,new_update=index,additional_updates=added,
            update_l2=delta,ability_update_l2=ability,ability_gradient_norm=float(np.asarray(loss).mean(0)[7]),
            actual_environment_steps_this_update=steps,actual_optimizer_steps_this_update=steps//spec['global_minibatch'],
            environment_steps=env_steps,optimizer_steps=opt_steps,cycle_seconds=elapsed,
            rollout_seconds=rollout,env_per_second=steps/elapsed,loss=np.asarray(loss).mean(0).tolist(),
            exposure=jax.tree.map(lambda a:np.asarray(a).tolist(),exposure),
            opponent_names=list(mixed.NAMES),league_generation=generation,
            league_manifests=manifests,categorical=True,dagger=0,combat_kl=0,unix=time.time())
        with (out/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(last)+'\n');stream.flush();os.fsync(stream.fileno())
        rows.append(last);atomic_json(out/'PROGRESS.json',dict(iterations=index,last=last))
        if args.smoke or counted%40==0 or index==target:
            stage('ATOMIC_SAVE',index);cp,receipt=save(index,last)
    if args.smoke:
        stage('FULL_RESTORE_TOKEN',target)
        result=operation(params,adam,states,aux,keys,update_key,bank,generation);jax.block_until_ready(result)
        assert finite(result)
        atomic_json(cp/'NEXT_UPDATE.json',dict(status='PASS',digest=digest(result),diagnostic_steps_not_counted=steps))
        atomic_json(out/'SMOKE.json',dict(status='PASS',passed=True,rows=rows,checkpoint=str(cp),
            counted_environment_steps=2*steps,unix=time.time()))
    else:
        atomic_json(out/'COMPLETE.json', completion_receipt(receipt, protocol))
    stage('EXIT',target)


if __name__=='__main__':
    main()
