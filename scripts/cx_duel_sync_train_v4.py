"""Two physical GPUs, one synchronized policy/Adam, constant global PPO batch."""
import argparse,fcntl,json,os,pickle,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('protocol');parser.add_argument('phase',choices=['smoke2','to8','formal','fresh'])
    args=parser.parse_args();protocol_path=Path(args.protocol);p=json.loads(protocol_path.read_text())
    root=Path(p['namespace']);out=root/'L2_DUEL_SYNC2';out.mkdir(parents=True,exist_ok=True)
    grant=json.loads(Path('.hermes/GPU_USER_RESERVATION.json').read_text())
    physical=p['physical_gpus'];assert physical==[0,1] and all(g in grant['allowed_gpus'] for g in physical)
    guards=[]
    for path in [root/'worker.lock']+[root.parent/f'GPU{g}.lock' for g in physical]:
        f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.append(f)
    for path,h in p['source_sha256'].items():assert sha(path)==h,path
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str,physical)),JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',DUEL_FRACTION='.5',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true --xla_disable_hlo_passes=multi_output_fusion',
        JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='true',JAX_COMPILATION_CACHE_DIR=str(root/'cache'),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0')
    from scripts.model_cuda_overlay import install
    overlay=install()
    import jax
    import jax.numpy as j
    import numpy as np
    import optax
    from scripts.combat_specialist_g_league_train import setup,atomic_pickle
    from scripts.terminal_checkpoint_restore import digest
    setup()
    os.environ['CC_SNAPSHOT_ROOT']=str(root.parent/'league/L2_DUEL_SYNC2')
    os.environ['CC_LEAGUE2_ARM']='L2_DUEL_SYNC2'
    os.environ['CC_LEAGUE2_PEERS']=json.dumps(p['arms']['L2_DUEL_SYNC2']['league2']['peers'])
    from scripts import cx_duel_mixed_v3 as M,cx_duel_lib_v3 as L
    from scripts.cx_duel_sync_update_v1 import build,verify_collectives
    from scripts.r4_k4_observation import bun_mode_zero_actor,migrate
    assert len(jax.devices())==2
    atomic_json(root/'COLLECTIVE_TEST.json',verify_collectives())
    os.environ['CC_SNAPSHOT_ROOT']=str(root.parent/'league/L2_DUEL_SYNC2')
    os.environ['CC_LEAGUE2_ARM']='L2_DUEL_SYNC2'
    os.environ['CC_LEAGUE2_PEERS']=json.dumps(p['arms']['L2_DUEL_SYNC2']['league2']['peers'])
    replicas=2;local_n=1024;ticks=128;steps=replicas*local_n*ticks;minibatch=4096
    spec=p['arms']['L2_DUEL_SYNC2']
    from scripts import r3_items_collector as C
    bs,ba=C.banks(None,None,json.loads(Path(p['r1_bank_manifest']).read_text()))
    spawns=j.asarray(json.loads(Path(p['spawn_manifest']).read_text())['spawn_cells'])
    cps=sorted(out.glob('u*/COMMITTED.json'))
    cp=cps[-1].parent if cps else Path(p['continuation_source']['checkpoint'])
    cm=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'state.pkl')==cm['state_sha256']
    with (cp/'state.pkl').open('rb') as f:payload=pickle.load(f)
    replicate=lambda tree:jax.tree.map(lambda x:np.stack([np.asarray(x)]*replicas),tree)
    params=replicate(payload['params']);adam=replicate(payload['adam'])
    index=int(payload['update']);environment_steps=int(payload['environment_steps']);optimizer_steps=int(payload['optimizer_steps'])
    def selected_bank(at):
        actors={};manifests={}
        for family in ('G','C'):
            manifest=L.load(family,at);manifests[family]=manifest
            ring=manifest['ring'][(at//40)%len(manifest['ring'])]
            for suffix,item in [('current',manifest['current']),('ring',ring),('fixed',manifest['fixed'][0])]:
                actors[family+suffix]=migrate(jax.tree.map(j.asarray,L.actor(item)))
        return actors,manifests
    if cps:
        assert payload['protocol_sha256'] in [sha(protocol_path)]+p.get('compatible_checkpoint_protocol_sha256',[])
        assert payload['layout']['replicas']==replicas and payload['layout']['per_device_envs']==local_n
        states,aux,keys,update_keys,bank=jax.tree.map(j.asarray,(payload['states'],payload['aux'],payload['keys'],payload['optimizer_keys'],payload['league_bank']))
        generation=int(payload['league_generation']);manifests=payload['league_manifests']
    else:
        seed=p['seed'];ks=jax.random.split(jax.random.PRNGKey(seed),replicas*local_n).reshape(replicas,local_n,2)
        states,aux,keys=jax.pmap(lambda keys:L.fork(M,keys,bs,ba,spawns,seed))(ks)
        update_keys=jax.random.split(jax.random.PRNGKey(seed^0xD0E1),replicas)
        selected,manifest=selected_bank(index)
        bank=replicate(jax.tree.map(lambda x:j.stack([x,x]),selected));generation=0
        manifests={'0':manifest,'1':manifest}
    L.receipt(out,dict(spec,sil_coef=.5),p)
    atomic_json(out/'SYNC_RECEIPT.json',dict(status='LOADED_SYNC2_NATIVE_DUEL',physical_gpus=physical,
        global_envs=2048,per_device_envs=1024,global_minibatch=8192,per_device_minibatch=4096,
        adam_steps_per_update=32,gradient_aggregation='lax.pmean before each Adam',
        normalization='global mode count/sum/centered-variance psum per actual8192 minibatch',
        random_streams='unique2048 per-env keys and2 optimizer keys; checkpointed; no fresh rekey',
        overlay=overlay,protocol_sha256=sha(protocol_path),independent_runs=False))
    collect=jax.pmap(M.collector(ticks,local_n,p['seed'],spawns),axis_name='replicas',
        in_axes=(0,0,0,0,None,None,0,0))
    update=build(minibatch,optax.adam(spec['lr'],eps=1e-8),.5)
    def operation(params,adam,states,aux,keys,update_keys,bank,generation):
        (states,aux,keys),batch,exposure=collect(params,states,aux,keys,bs,ba,bank,j.full(replicas,generation,j.int32))
        split=jax.vmap(jax.random.split)(update_keys);update_keys=split[:,0];uk=split[:,1]
        (params,adam),loss=update(params,adam,batch,uk)
        return params,adam,states,aux,keys,update_keys,loss,exposure
    def stage(name):atomic_json(out/'CYCLE_STAGE.json',dict(stage=name,update=index,unix=time.time()))
    def save(row):
        cp=out/f'u{index:06d}';cp.mkdir(exist_ok=True)
        host=jax.device_get(dict(params=jax.tree.map(lambda x:x[0],params),adam=jax.tree.map(lambda x:x[0],adam),
            states=states,aux=aux,keys=keys,optimizer_keys=update_keys,league_bank=bank,
            league_generation=generation,league_manifests=manifests,update=index,
            environment_steps=environment_steps,optimizer_steps=optimizer_steps,protocol_sha256=sha(protocol_path),
            seed=p['seed'],layout=dict(replicas=2,per_device_envs=1024,state_device_axis=True)))
        atomic_pickle(cp/'state.pkl',host);atomic_pickle(cp/'policy.pt',host['params']['actor'])
        atomic_pickle(cp/'policy_bun_mode0.pt',bun_mode_zero_actor(host['params']['actor']))
        atomic_json(cp/'COMMITTED.json',dict(status='PASS',passed=True,checkpoint=str(cp),update=index,
            state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),
            bun_mode_zero_policy_sha256=sha(cp/'policy_bun_mode0.pt'),environment_steps=environment_steps,
            optimizer_steps=optimizer_steps,physical_gpus=physical,layout=host['layout'],metrics=row,unix=time.time()))
        L.publish('G',str(cp))
        return cp
    if args.phase=='fresh':
        stage('FRESH_FULL_SYNCHRONIZED_UPDATE')
        result=operation(params,adam,states,aux,keys,update_keys,bank,generation);jax.block_until_ready(result)
        expected=json.loads((cp/'NEXT_UPDATE.json').read_text())
        actual=digest(result);assert actual==expected['digest'],'full synchronized restore mismatch'
        atomic_json(out/f'FRESH_u{index:06d}.json',dict(status='PASS_FULL_PARAMS_ADAM_SHARDED_ENV_AUX_RNG_LEAGUE',
            bitwise_exact=True,maxabs=0,checkpoint=str(cp),digest=actual,diagnostic_steps_not_retained=steps))
        return
    target={'smoke2':19304,'to8':19310,'formal':p['chunk_start_update']+p['chunk_updates']}[args.phase]
    if args.phase!='smoke2':assert (out/'FRESH_u019304.json').exists()
    compile_seconds=None
    last={}
    for index in range(index+1,target+1):
        if any((d/name).exists() for d in (root.parent,out) for name in ('USER_STOP','USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json')):
            index-=1;save(dict(saved_user_stop=True));return
        counted=index-19302
        if counted>1 and (counted-1)%40==0:
            dest=1-generation
            assert not np.any(np.asarray(aux['combat_mix']['opponent_generation'])==dest)
            selected,manifest=selected_bank(index)
            bank=jax.tree.map(lambda old,new:old.at[:,dest].set(j.broadcast_to(new,old[:,dest].shape)),bank,selected)
            generation=dest;manifests[str(dest)]=manifest
        before=jax.device_get(params);began=time.monotonic();stage('SYNCHRONIZED_COLLECTOR')
        (states,aux,keys),batch,exposure=collect(params,states,aux,keys,bs,ba,bank,j.full(replicas,generation,j.int32))
        jax.block_until_ready(batch);rollout=time.monotonic()-began;stage('PMEAN_ADAM')
        split=jax.vmap(jax.random.split)(update_keys);update_keys=split[:,0]
        (params,adam),loss=update(params,adam,batch,split[:,1]);jax.block_until_ready(params)
        host=jax.device_get(params)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves((host,jax.device_get(adam))))
        assert all(np.array_equal(x[0],x[1]) for x in jax.tree.leaves(host)),'replica policy divergence'
        delta={branch:float(optax.global_norm(jax.tree.map(lambda a,b:a[0]-b[0],host[branch],before[branch]))) for branch in ('actor','critic')}
        assert min(delta.values())>0
        environment_steps+=steps;optimizer_steps+=32
        elapsed=time.monotonic()-began
        if compile_seconds is None:compile_seconds=elapsed
        exp=jax.tree.map(lambda x:np.asarray(x).sum(0).tolist(),exposure['bun'])
        assert exp['duel_mode_ticks']==[steps/2,steps/2]
        assert exp['duel_package_events']==0 and exp['duel_initial_bank_rows']==0
        assert abs(sum(exp['duel_reward_parts'])-(16*exp['duel_learner_kills']-16*exp['duel_learner_deaths']-8*exp['duel_self_deaths']))<.01
        last=dict(passed=True,finite=True,new_update=index,update_l2=delta,replicas_bitwise_exact=True,
            actual_environment_steps_this_update=steps,actual_optimizer_steps_this_update=32,
            environment_steps=environment_steps,optimizer_steps=optimizer_steps,cycle_seconds=elapsed,
            rollout_seconds=rollout,env_per_second=steps/elapsed,first_update_including_compile_seconds=compile_seconds,
            physical_gpus=physical,global_envs=2048,global_minibatch=8192,per_device_envs=1024,
            per_device_minibatch=4096,loss=np.asarray(loss).mean((0,1)).tolist(),
            minibatch_signed_credit=np.asarray(loss)[0,:,10:].tolist(),exposure=dict(bun=exp),unix=time.time())
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(last)+'\n');f.flush();os.fsync(f.fileno())
        atomic_json(out/'PROGRESS.json',dict(iterations=index,last=last))
        if counted<=2 or counted%40==0 or index==target:cp=save(last)
        print(json.dumps(dict(update=index,finite=True,envsteps_s=steps/elapsed,seconds=elapsed,delta=delta)),flush=True)
    stage('FULL_NEXT_UPDATE_TOKEN')
    result=operation(params,adam,states,aux,keys,update_keys,bank,generation);jax.block_until_ready(result)
    atomic_json(cp/'NEXT_UPDATE.json',dict(status='PASS',digest=digest(result),diagnostic_steps_not_counted=steps))
    atomic_json(out/f'PHASE_{args.phase}.json',dict(status='PASS',update=index,checkpoint=str(cp),last=last))

if __name__=='__main__':main()
