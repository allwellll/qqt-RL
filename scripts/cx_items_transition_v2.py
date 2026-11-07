"""One finite, authorized save-boundary migration of the seven live policies.

This is the training transition itself, not a recurring monitoring task. GPU6
must pass admission first. Remaining workers keep training until their turn.
Old checkpoints/protocols/stop markers are retained. Failure resumes the old
recipe from the exact saved state, without changing its sources or protocol.
"""
import fcntl,json,os,shutil,subprocess,sys,time,concurrent.futures
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.model_launch import manifest,command,public_project

BASE=Path('runs/cc_train_20261105/training')
ROOT=Path('runs/items_alignment_20261107/v2')
REPORT=Path('reports/cx_items_training_align_20261107.md')
STOPS=('USER_STOP','USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json')
CPUS={0:'0-7',2:'144-151',3:'152-159',4:'48-55',5:'56-63',6:'64-71',7:'72-79'}


def read(p):return json.loads(Path(p).read_text())
def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except OSError:return False


def note(arm,stage,**kw):
    row=dict(arm=arm,stage=stage,unix=time.time(),**kw)
    atomic_json(ROOT/'transitions'/arm/'STATUS.json',row)
    with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    with REPORT.open('a') as f:f.write('\n迁移事件：`'+json.dumps(row,ensure_ascii=False)+'`\n')
    print(json.dumps(row),flush=True)


def active_config(arm):
    old=BASE/'formal'/arm
    supervisor=read(old/'SUPERVISOR.json')
    # Trust the actual worker argv, never a stale proposed configuration.
    worker=supervisor.get('worker_pid')
    if not worker or not live(worker):
        rows=[]
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args=proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
                if 'model_training.py' not in args:continue
                cfg=read(args[-1])
                if len(cfg.get('args',[]))>1 and cfg['args'][1]==arm:rows.append((int(proc.name),cfg,args))
            except (OSError,ValueError,KeyError):continue
        assert len(rows)==1,(arm,rows)
        worker,cfg,args=rows[0]
    else:
        args=Path(f'/proc/{worker}/cmdline').read_bytes().decode().split('\0')[:-1]
        assert 'model_training.py' in args,args
        cfg=read(args[-1])
    protocol=Path(cfg['args'][0]);p=read(protocol)
    assert arm in p['arms'] and (Path(p['namespace'])/arm).resolve()==old.resolve()
    return dict(arm=arm,old_out=str(old.resolve()),worker=worker,parent=supervisor['pid'],
        original_config=cfg,visible_args=args,protocol=str(protocol),protocol_sha256=sha(protocol),
        physical_gpus=p.get('physical_gpus',[cfg['gpu']]),implementation=cfg['module'])


def prepare_boundary(plan):
    arm=plan['arm'];out=Path(plan['old_out']);folder=ROOT/'transitions'/arm
    folder.mkdir(parents=True,exist_ok=True)
    for d in (BASE,out):
        assert not any((d/n).exists() for n in STOPS),'persistent stop gate: '+str(d)
    assert read(ROOT/'ACCEPTED_RULE.json')['status']=='PASS_ACTUAL_CANONICAL_WEB_PARITY'
    marker=out/'USER_STOP.json'
    atomic_json(folder/'ORIGINAL_LAUNCH.json',plan)
    atomic_json(marker,dict(owner='CX-ITEMS-TRANSITION-V2',reason='authorized physical version save-boundary migration',
        destination=str(folder),unix=time.time()))
    note(arm,'SAVE_BOUNDARY_REQUESTED',old_protocol=plan['protocol'],worker=plan['worker'])
    deadline=time.monotonic()+180
    while live(plan['worker']) and time.monotonic()<deadline:time.sleep(2)
    assert not live(plan['worker']),'owned worker failed to save/exit in180seconds'
    latest=sorted(out.glob('u*/COMMITTED.json'))[-1];cp=latest.parent;cm=read(latest)
    assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
    # The old episode/aux/RNG are retained, so no fabricated new reset here.
    p=read(plan['protocol']);newroot=BASE/f'items_v3_20261107_{arm.lower()}'
    assert not newroot.exists(),'new namespace already exists'
    newout=newroot/arm;newout.mkdir(parents=True)
    oldleague=BASE/'league'/arm
    shutil.copytree(oldleague,newroot/'league'/arm)
    start=cm['update'];newid=f'CX-ITEMS-V3-{arm}-20261107'
    p.update(id=newid,namespace_id=newid,namespace=str(newroot),chunk_start_update=start,
        chunk_updates=1200,league2_chunk_updates=1200,continuation_source=cm,
        compatible_checkpoint_protocol_sha256=[plan['protocol_sha256']],items_full_state_migration=True,
        rule_version='bun-native-v3-items')
    accepted=ROOT/'ACCEPTED_RULE.json'
    p['items_transition']=dict(accepted_rule=str(accepted),accepted_rule_sha256=sha(accepted),
        parent_protocol=plan['protocol'],parent_protocol_sha256=plan['protocol_sha256'],
        implementation_module='scripts.cx_duel_sync_train_v4' if arm=='L2_DUEL_SYNC2' else plan['implementation'],
        original_checkpoint=str(cp),old_ruleset='bun-native-v2',new_ruleset='bun-native-v3-items',
        exact_incoming_state_required=True,supply_at_actual_reset_only=True,
        unchanged_reward=True,duel_item_pickup_reward=0)
    for f in Path('scripts').glob('cx_items_*v2.*'):p['source_sha256'][str(f)]=sha(f)
    p['source_sha256'].update(read(accepted)['source_sha256'])
    protocol=newroot/'PROTOCOL_V1.json';atomic_json(protocol,p)
    budget=dict(status='PREDECLARED',id=newid,protocol=str(protocol),protocol_sha256=sha(protocol),
        old_checkpoint=str(cp),old_protocol=plan['protocol'],start=start,target=start+1200,
        retained_updates=1200,fresh_envsteps=1200*(262144 if arm=='L2_DUEL_SYNC2' else p['common']['envs_per_device']*p['common']['rollout_ticks']),
        successor='same namespace/config next1200 after exact fresh; no single-point noise stop',
        stopping='technical failure, persistent user stop, or two consecutive strong-evaluation joint net-win/delivery declines',
        physical_gpus=plan['physical_gpus'],complete_state_migration=True)
    atomic_json(newroot/'BUDGET.json',budget);atomic_json(folder/'BUDGET.json',budget)
    atomic_json(out/'PHYSICS_ARCHIVE.json',dict(status='SEALED_OLD_RULE',checkpoint=str(cp),
        protocol=plan['protocol'],protocol_sha256=plan['protocol_sha256'],destination=str(newroot),
        old_rule='bun-native-v2',never_reinterpret_old_replay=True,unix=time.time()))
    with Path('docs/training_experiment_ledger.md').open('a') as f:
        f.write(f'\n### {newid}\n\nFull saved state migration from `{cp}`; old bun-native-v2 archived; new bun-native-v3-items, same seed/reward/league/mix, fixed1200 to{start+1200}, same-recipe successor predeclared. Protocol `{protocol}` SHA `{sha(protocol)}`. Actual smoke/fresh/finite admission pending; no item reward added to DUEL.\n')
    note(arm,'SAVED_NEW_PROTOCOL',source_update=start,protocol=str(protocol),physical_gpus=plan['physical_gpus'])
    return protocol,p


def worker_config(plan,protocol,p,phase):
    arm=plan['arm'];gpu=plan['physical_gpus'][0]
    if arm=='L2_DUEL_SYNC2':args=[str(protocol),phase]
    else:
        args=[str(protocol),arm,str(p['seed'])]
        args+= {'smoke2':['--smoke'],'fresh':['--verify-restore'],
            'to8':['--until',str(p['chunk_start_update']+8)],
            'formal':['--until',str(p['chunk_start_update']+p['chunk_updates'])]}[phase]
    return manifest(dict(module='scripts.cx_items_install_v2',gpu=gpu,args=args))


def launch(plan,protocol,p,phase):
    arm=plan['arm'];out=Path(p['namespace'])/arm;cfg=worker_config(plan,protocol,p,phase)
    with (out/f'{phase}.log').open('a') as log:
        child=subprocess.Popen(['taskset','-c',CPUS[plan['physical_gpus'][0]]]+command('training',cfg),
            cwd=public_project(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(out/'SUPERVISOR.json',dict(status='RUNNING',owner='CX-ITEMS-TRANSITION-V2',pid=os.getpid(),
        worker_pid=child.pid,physical_gpus=plan['physical_gpus'],phase=phase,protocol=str(protocol),config=cfg,
        target_update=p['chunk_start_update']+p['chunk_updates'],unix=time.time()))
    note(arm,'ACTUAL_WORKER_STARTED',phase=phase,config=cfg,compile_budget_seconds=900)
    return child


def wait_phase(plan,p,child,phase):
    out=Path(p['namespace'])/plan['arm'];deadline=time.monotonic()+1200
    while child.poll() is None:
        assert time.monotonic()<deadline,(phase,'1200second finite phase limit')
        time.sleep(3)
    assert child.returncode==0,(phase,child.returncode,str(out/f'{phase}.log'))
    note(plan['arm'],'PHASE_PASS',phase=phase)


def fallback(plan,error):
    # A stopped old worker can restart from the intact saved boundary; only
    # archive the transition marker we created, never a user's existing gate.
    out=Path(plan['old_out']);marker=out/'USER_STOP.json'
    note(plan['arm'],'TECHNICAL_REJECTION',error=str(error))
    if marker.exists() and read(marker).get('owner')=='CX-ITEMS-TRANSITION-V2':
        marker.rename(ROOT/'transitions'/plan['arm']/'SAVED_TRANSITION_GATE.json')
    else:raise RuntimeError('fallback cannot remove a user stop gate')
    # Old owners normally exit after their save gate. Do not duplicate an
    # existing worker; scan actual argv again before launching a fallback.
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
            if 'model_training.py' in args and read(args[-1]).get('args',[])[:2]==plan['original_config']['args'][:2]:
                note(plan['arm'],'OLD_RECIPE_ALREADY_RUNNING');return
        except (OSError,ValueError):pass
    cfg=manifest(plan['original_config'])
    with (out/'items_transition_rollback.log').open('a') as log:
        child=subprocess.Popen(['taskset','-c',CPUS[plan['physical_gpus'][0]]]+command('training',cfg),
            cwd=public_project(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    note(plan['arm'],'OLD_RECIPE_ROLLBACK_STARTED',config=cfg,worker=child.pid)


def migrate(arm):
    plan=active_config(arm);folder=ROOT/'transitions'/arm;folder.mkdir(parents=True,exist_ok=True)
    child=None
    try:
        protocol,p=prepare_boundary(plan)
        for phase in ('smoke2','fresh','to8','fresh'):
            child=launch(plan,protocol,p,phase);wait_phase(plan,p,child,phase)
        out=Path(p['namespace'])/arm;start=p['chunk_start_update']
        assert read(out/'FULL_STATE_MIGRATION.json')['bitwise_exact']
        fresh=[]
        for u in (start+2,start+8):
            f=out/(f'FRESH_u{u:06d}.json' if arm=='L2_DUEL_SYNC2' else f'RESTORE_u{u:06d}.json')
            value=read(f);assert value.get('bitwise_exact',value.get('full_Actor_Critic_Adam_env_aux_RNG_league_exact'))
            fresh.append(value)
        rows=[json.loads(l) for l in (out/'updates.jsonl').read_text().splitlines() if l.strip()]
        assert len(rows)==8 and all(r['finite'] and min(r['update_l2'].values())>0 for r in rows)
        receipt=read(out/'ITEM_RULE_RECEIPT.json');assert receipt['rule_version']=='bun-native-v3-items'
        # Every rollout contains real reset supply counters from the final
        # collector; counters are observations, never bonus reward terms.
        assert all('items_reset_ground' in r['exposure']['bun'] for r in rows)
        admission=dict(status='PASS_NEW_PHYSICS_FINITE_EXACT_MIGRATION_FRESH2_8',physical_gpus=plan['physical_gpus'],
            migration=read(out/'FULL_STATE_MIGRATION.json'),fresh=fresh,rule_receipt=receipt,
            first_update=rows[0],latest_update=rows[-1],protocol_sha256=sha(protocol))
        atomic_json(Path(p['namespace'])/'TECHNICAL_ADMISSION.json',admission)
        with Path('docs/training_experiment_ledger.md').open('a') as f:
            f.write(f'\n{p["id"]}: actual eight finite updates; Actor/Critic/Adam/env/aux/RNG/league incoming exact; independent fresh2/8 PASS; actual supply rollout counters crosschecked; fixed1200 formal follows. Admission `{p["namespace"]}/TECHNICAL_ADMISSION.json`.\n')
        child=launch(plan,protocol,p,'formal')
        # Return only after a REAL formal update; occupancy is not success.
        deadline=time.monotonic()+900
        while child.poll() is None and time.monotonic()<deadline:
            rows=[json.loads(l) for l in (out/'updates.jsonl').read_text().splitlines() if l.strip()]
            if rows[-1]['new_update']>start+8:
                row=rows[-1];assert row['finite']
                atomic_json(Path(p['namespace'])/'FIRST_FORMAL_UPDATE.json',dict(status='PASS_REAL_FINITE_UPDATE',row=row))
                registry=ROOT/'active';temp=registry/(arm+'.next')
                temp.symlink_to(out.resolve(),target_is_directory=True);os.replace(temp,registry/arm)
                note(arm,'ADMITTED_REAL_FORMAL_UPDATE',update=row['new_update'],delta=row['update_l2'],
                    envsteps_s=row['env_per_second'],source_update=start,target=start+1200)
                return dict(plan=plan,protocol=str(protocol),namespace=p['namespace'],worker=child.pid)
            time.sleep(3)
        raise RuntimeError('formal worker has no finite update in900seconds')
    except Exception as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        fallback(plan,error)
        return dict(arm=arm,status='ROLLED_BACK',error=str(error))


def main():
    (ROOT/'transitions').mkdir(exist_ok=True)
    guard=(ROOT/'transition.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry=ROOT/'active';registry.mkdir(exist_ok=True)
    for arm in ('L2_DUEL','L2_DUEL25','L2_DUEL75','L2_DUEL_SYNC2','L2_FTN2','L2_ATK','L2_SP'):
        alias=registry/arm
        if not alias.exists():alias.symlink_to((BASE/'formal'/arm).resolve(),target_is_directory=True)
    first=migrate('L2_DUEL')
    if first.get('status')=='ROLLED_BACK':return
    results=[first]
    # Bound cold compilation to two additional policies at once; all other
    # policies retain effective old-rule training until their saved boundary.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(migrate,arm) for arm in ('L2_DUEL25','L2_DUEL75','L2_DUEL_SYNC2','L2_FTN2','L2_ATK','L2_SP')]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result());atomic_json(ROOT/'MIGRATION_RESULTS.json',dict(status='IN_PROGRESS',results=results))
    atomic_json(ROOT/'MIGRATION_RESULTS.json',dict(status='ALL_TRANSITIONS_FINISHED',results=results))
    # Continue the predeclared fixed budgets under this same training owner;
    # no cron, timer, separate owner or additional monitoring process.
    running=[r for r in results if 'worker' in r]
    while running:
        for r in list(running):
            if live(r['worker']):continue
            plan=r['plan'];protocol=Path(r['protocol']);p=read(protocol);out=Path(p['namespace'])/plan['arm']
            if any((d/n).exists() for d in (BASE,out) for n in STOPS):
                note(plan['arm'],'SAVED_USER_OR_TECHNICAL_STOP');running.remove(r);continue
            target=p['chunk_start_update']+p['chunk_updates']
            latest=sorted(out.glob('u*/COMMITTED.json'))[-1]
            if read(latest)['update']!=target:
                note(plan['arm'],'FORMAL_TECHNICAL_FAILURE',latest=read(latest)['update'],target=target)
                fallback(plan,'formal exited before budget target');running.remove(r);continue
            child=launch(plan,protocol,p,'fresh');wait_phase(plan,p,child,'fresh')
            nxt=protocol.with_name('PROTOCOL_BUDGET_'+str(target)+'.json')
            q=json.loads(json.dumps(p));q.update(chunk_start_update=target,continuation_source=read(latest))
            q['compatible_checkpoint_protocol_sha256'].append(sha(protocol))
            atomic_json(nxt,q)
            child=launch(plan,nxt,q,'formal')
            note(plan['arm'],'SAME_NAMESPACE_FIXED1200_SUCCESSOR',start=target,target=target+1200,protocol=str(nxt))
            r.update(worker=child.pid,protocol=str(nxt))
        time.sleep(10)


if __name__=='__main__':main()
