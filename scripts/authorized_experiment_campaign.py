"""Durable, two-lane owner for the explicitly authorized bounded campaign.

Never launches R2/O5 or signals unrelated processes. Completion requires a real
artifact, not a successful launch. All commands and receipts use relative paths.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import traceback
import time

from scripts.population_run_state import atomic_json, lock, sha
from scripts.campaign_eval_resources import limits, cpu_group, regression, scheduling_slot

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/authorized_campaign_20261026'


class AdoptedProcess:
    """Observe an already launched child after owner restart; never relaunch."""
    def __init__(self,identity):self.identity=identity
    def poll(self):
        path=Path('/proc')/str(self.identity)/'stat'
        try:
            state=path.read_text().rsplit(')',1)[1].split()[0]
            if state=='Z':
                try:os.waitpid(self.identity,os.WNOHANG)
                except ChildProcessError:pass
                return 0
            return None
        except OSError:return 0


def matching_process(job):
    matches=[]
    expected=job['args']
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:words=(path/'cmdline').read_bytes().decode().split('\0')
        except OSError:continue
        if words and Path(words[0]).name.startswith('python') and words[1:1+len(expected)]==expected:
            matches.append(int(path.name))
    if len(matches)>1:raise RuntimeError('Duplicate job processes detected')
    return matches[0] if matches else None


def read(path):
    return json.loads(Path(path).read_text())


def marker_passes(path):
    try:return read(path).get('passed') is True
    except (OSError,ValueError):return False


def job_completion_passes(job):
    """Atomic completion with a checkpoint must verify its real payload hashes."""
    try:
        value=read(ROOT/job['complete'])
        if value.get('passed') is not True:return False
        if value.get('checkpoint') and value.get('state_sha256'):
            checkpoint=ROOT/value['checkpoint']
            if sha(checkpoint/'state.pkl')!=value['state_sha256']:return False
            if value.get('policy_sha256') and sha(checkpoint/'policy.pt')!=value['policy_sha256']:return False
        return True
    except (OSError,ValueError):return False


def progress_observation(job,now):
    path=ROOT/job.get('progress',job['complete'])
    try:
        value=read(path)
        latest=path.stat().st_mtime
        read_error=None
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        # Progress files are atomically replaced, but existence and open are not
        # atomic as a pair. A disappearing/incomplete passive metric must never
        # take down the unique GPU owner.
        value={}
        try: latest=path.stat().st_mtime
        except OSError: latest=job.get('started_unix',now)
        read_error=type(exc).__name__
    return dict(job=job['id'],progress_exists=path.exists(),
        completed_iterations=value.get('iterations'),completed_games=value.get('completed_games'),
        progress_age_seconds=max(0.,now-latest),last=value.get('last'),
        evidence_path=str(path.relative_to(ROOT)),progress_read_error=read_error,
        process_presence_is_not_update_evidence=True)


def replace_pending_job(jobs,replacement,*,retry_failed=False,register_protocol=False):
    """A request/claim race must not terminate the owner or mutate a worker."""
    old=next(j for j in jobs if j['id']==replacement['id'])
    if old['status']!='REGISTERED' and not (retry_failed and old['status'] in ('FAILED','BLOCKED_TECHNICAL','INTERRUPTED_REQUIRES_RECONCILIATION')) and not (register_protocol and old['status']=='AWAITING_PROTOCOL'):
        return dict(passed=False,consumed=True,reason='job_already_claimed_or_terminal',
                    observed_status=old['status'],active_job_unchanged=True)
    old.update(replacement)
    return dict(passed=True,consumed=True)


def rebind_authorized_block(jobs,replacement,authorization,expected_sha256):
    """User-authored amendment reopens a terminal protocol block, with receipt."""
    old=next(j for j in jobs if j['id']==replacement['id'])
    if old['status']!='BLOCKED_PROTOCOL':
        return dict(passed=False,reason='not_protocol_blocked',observed_status=old['status'])
    if sha(ROOT/authorization)!=expected_sha256:raise ValueError('Authorization SHA mismatch')
    approved=read(ROOT/authorization)
    if not approved.get('authorized_by_user') or approved.get('id')!='CAMPAIGN-BANK-RATIO-AB-20261026-V1-AMENDMENT01':
        raise ValueError('Expected explicit Bank-ratio resume amendment')
    history=dict(status=old['status'],terminal_evidence=old.get('terminal_evidence'),complete=old['complete'])
    old.update(replacement,prior_protocol_block=history,authorization=authorization)
    return dict(passed=True,old_block_preserved=True,authorization_sha256=expected_sha256)


def accept_delegated_result(jobs,job_id,receipt_path,receipt_sha256):
    """Accept externally completed diagnostics without launching their worker."""
    job=next(j for j in jobs if j['id']==job_id)
    if job['status']!='DELEGATED_SIDE_CX':
        return dict(passed=False,consumed=True,reason='not_delegated',observed_status=job['status'])
    try:
        if sha(ROOT/receipt_path)!=receipt_sha256:raise ValueError('Receipt SHA mismatch')
        receipt=read(ROOT/receipt_path)
        if receipt.get('passed') is not True or receipt.get('task')!=job_id:raise ValueError('Invalid receipt')
        if sha(ROOT/receipt['side_complete_path'])!=receipt['side_complete_sha256']:raise ValueError('Side marker SHA mismatch')
        for item in receipt['input_hashes']:
            if not item['passed'] or sha(ROOT/item['path'])!=item['expected']:raise ValueError('Input SHA mismatch')
        for item in receipt.get('copied_evidence_hashes',[]):
            if not item['passed'] or sha(ROOT/item['copied_path'])!=item['expected']:raise ValueError('Copied artifact SHA mismatch')
    except (OSError,ValueError,KeyError) as exc:
        return dict(passed=False,consumed=True,reason=str(exc),worker_untouched=True)
    job.update(status='COMPLETE',original_complete=job['complete'],complete=receipt_path,
        terminal_evidence=dict(path=receipt_path,sha256=receipt_sha256,side_complete=receipt['side_complete_path']),
        scientific_acceptance=receipt.get('scientific_acceptance'),terminal_unix=time.time(),
        completion_scope='first-round diagnostic execution; limitations preserved; no re-execution')
    return dict(passed=True,consumed=True,job=job_id,no_worker_launched=True)


def eligible(jobs,resource,*,paused):
    if paused:return []
    done={j['id'] for j in jobs if j['status']=='COMPLETE'}
    return [j for j in jobs if j['resource']==resource and j['status']=='REGISTERED'
            and set(j['dependencies'])<=done
            and all((ROOT/path).exists() for path in j.get('ready_files',[]))
            and ready_json_matches(j)]


def gpu_idle_reason(jobs):
    pending=[j for j in jobs if j['resource']=='gpu' and j['status']!='COMPLETE']
    return ('NO_REGISTERED_GPU_SUCCESSOR' if not pending
            else 'GPU_PENDING_DEPENDENCIES_OR_TECHNICAL_WORK')


def ready_json_matches(job):
    """A stale smoke request cannot claim a newly committed formal snapshot."""
    for path, expected in job.get('ready_json', {}).items():
        try: value = read(ROOT/path)
        except (OSError, ValueError): return False
        if any(value.get(key) != item for key, item in expected.items()): return False
    return True


def resources_free(memory,apps):
    return len(memory)==8 and all(float(x)==0 for x in memory) and not apps


def gpu_free_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    rows=[[float(x.strip()) for x in row.split(',')] for row in raw.splitlines()]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name,used_memory',
        '--format=csv,noheader'],text=True).splitlines()
    return dict(free=resources_free([r[1] for r in rows],apps),
        devices=[dict(index=int(r[0]),memory_used_mib=r[1],utilization_percent=r[2]) for r in rows],
        compute_app_count=len(apps),unrelated_tasks_preserved=True)


def bank_owned_slots_only():
    """Resume a dead CPU slot controller without duplicating live GPU workers."""
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    if not apps:return False
    known=[]
    for value in apps:
        try:
            args=(Path('/proc')/str(int(value))/'cmdline').read_bytes().decode().split('\0')
            if args[1:3]!=['-m','scripts.bank_ratio_slot_train'] or '--verify' in args:return False
            if args[3] not in ('control','reduced') or int(args[4]) not in (2026103151,2026103152,2026103153):return False
            known.append((args[3],args[4]))
        except (OSError,ValueError,IndexError):return False
    return len(known)==len(set(known)) and len(known)<=6


def notify(job,event,details):
    value=dict(job=job,event=event,details=details,unix=time.time())
    atomic_json(BASE/'notifications'/f'{time.time_ns()}.json',value)
    try:subprocess.run(['jdme-notify','send',f'实验campaign {job} {event}：{details}'],
        timeout=10,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.TimeoutExpired):pass


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['owner'])
    parser.parse_args();BASE.mkdir(parents=True,exist_ok=True)
    owner_lock=lock(BASE/'owner.lock')
    atomic_json(BASE/'OWNER.json',dict(unique=True,role='campaign_owner',started_unix=time.time(),
        old_queues_resumed=False,resume_command='PYTHONPATH=. .venv/bin/python -m scripts.authorized_experiment_campaign owner'))
    campaign=read(BASE/'CAMPAIGN.json');active={};first_seen=set();last_gpu_check=0;last_samples=0
    gpu_idle_since=None;gpu_idle_notified=False;last_idle_check=0
    for job in campaign['jobs']:
        if job['status']=='RUNNING':
            if job_completion_passes(job):
                job['status']='COMPLETE'
                job['terminal_evidence']=dict(path=job['complete'],sha256=sha(ROOT/job['complete']))
                notify(job['id'],'COMPLETE_RECONCILED',job['complete'])
            else:
                identity=matching_process(job)
                if identity is not None:
                    stream=(BASE/'owner.log').open('a')
                    slot=scheduling_slot(job['resource'],active,job.get('scheduling_slot'))
                    key=job['resource'] if job['resource']=='gpu' else f"{job['resource']}:{slot}"
                    active[key]=(job,AdoptedProcess(identity),stream)
                    job['scheduling_slot']=slot
                    job['adopted_existing_worker']=True
                else:job['status']='INTERRUPTED_REQUIRES_RECONCILIATION'
    first_seen.update(j['id'] for j in campaign['jobs'] if j.get('first_real_work_evidence'))
    while True:
        paused=(BASE/'USER_STOP.json').exists() or (BASE/'USER_PAUSE.json').exists()
        campaign['paused']=paused
        config_path=BASE/'EVAL_PARALLEL_CONFIG.json'
        config=read(config_path) if config_path.exists() else {'limits':dict(gpu=1,cpu=1,analysis=1,evaluation=1,js=1)}
        caps=limits(config)
        allowed=sorted(os.sched_getaffinity(0))
        for key,(running,process,_) in active.items():
            resource=running['resource']
            if resource=='gpu':continue
            group=cpu_group(allowed,resource,running.get('scheduling_slot',0),config.get('cpu_partition_offset',12))
            identity=process.identity if isinstance(process,AdoptedProcess) else process.pid
            if group and running.get('cpu_affinity')!=group:
                try:
                    for thread in (Path('/proc')/str(identity)/'task').iterdir():
                        os.sched_setaffinity(int(thread.name),group)
                    running['cpu_affinity']=group
                except (ProcessLookupError,FileNotFoundError,PermissionError):pass
        from scripts.campaign_eval_resources import throughput_guard, js_handoff_tick
        throughput_guard(campaign,active,config,config_path,ROOT,BASE)
        caps=limits(config)
        js_handoff_tick(campaign,active,BASE)
        for request in sorted((BASE/'requests').glob('*.json')):
            value=read(request)
            if (BASE/'requests_done'/request.name).exists():continue
            if value['action']=='handoff_js_scheduler':
                from scripts.campaign_eval_resources import request_js_handoff
                if request_js_handoff(campaign,value,matching_process):
                    atomic_json(BASE/'requests_done'/request.name,dict(passed=True,phase='wait_for_current_batch_commit',unix=time.time()))
                continue
            if value['action']=='reload_owner':
                atomic_json(BASE/'CAMPAIGN.json',campaign)
                atomic_json(BASE/'requests_done'/request.name,dict(passed=True,unix=time.time()))
                os.execv(sys.executable,[sys.executable,'-m','scripts.authorized_experiment_campaign','owner'])
            if value['action']=='watchdog_selftest_exit':
                # Explicit bounded liveness drill: leave all worker sessions
                # untouched and persist state before the watchdog adopts them.
                atomic_json(BASE/'CAMPAIGN.json',campaign)
                atomic_json(BASE/'requests_done'/request.name,dict(passed=True,unix=time.time(),
                    kind='controlled_owner_exit_only',workers_signalled=False))
                raise SystemExit(73)
            if value['action']=='accept_delegated_result':
                result=accept_delegated_result(campaign['jobs'],value['job_id'],value['receipt'],value['receipt_sha256'])
                atomic_json(BASE/'requests_done'/request.name,dict(**result,request_sha256=sha(request),unix=time.time()))
                if result['passed']:notify(value['job_id'],'COMPLETE_SIDE_ACCEPTED',value['receipt'])
                continue
            if value['action']=='rebind_authorized_block':
                result=rebind_authorized_block(campaign['jobs'],value['job'],value['authorization'],value['authorization_sha256'])
                atomic_json(BASE/'requests_done'/request.name,dict(**result,request_sha256=sha(request),unix=time.time()))
                continue
            if value['action']=='refresh_running_source_seal':
                running=next(j for j in campaign['jobs'] if j['id']==value['job_id'])
                seal_path=ROOT/value['seal']
                if running['status']!='RUNNING' or sha(seal_path)!=value['seal_sha256']:
                    result=dict(passed=False,reason='status_or_seal_mismatch')
                else:
                    seal=read(seal_path)
                    if any(sha(ROOT/p)!=h for p,h in seal['files'].items()):raise ValueError('Invalid refreshed source seal')
                    running['frozen_files']=seal['files']
                    result=dict(passed=True,worker_untouched=True,reason=value['reason'])
                atomic_json(BASE/'requests_done'/request.name,dict(**result,unix=time.time()))
                continue
            if value['action'] in ('replace_job','retry_failed_job','register_protocol'):
                replacement=value['job']
                result=replace_pending_job(campaign['jobs'],replacement,
                    retry_failed=value['action']=='retry_failed_job',
                    register_protocol=value['action']=='register_protocol')
                atomic_json(BASE/'requests_done'/request.name,dict(**result,request_sha256=sha(request),unix=time.time()))
                continue
            elif value['action']=='append_job':campaign['jobs'].append(value['job'])
            atomic_json(BASE/'requests_done'/request.name,dict(passed=True,request_sha256=sha(request),unix=time.time()))
        # Only this owner mutates campaign job state. No launch-only success.
        for lane,(job,process,stream) in list(active.items()):
            observation=progress_observation(job,time.time())
            atomic_json(BASE/'progress_observations'/f'{job["id"]}.json',observation)
            if observation['progress_age_seconds']>=180 and time.time()-job.get('last_progress_age_notification',0)>=300:
                job['last_progress_age_notification']=time.time()
                notify(job['id'],'进度年龄提醒',f"{observation['progress_age_seconds']:.0f}s without committed progress; inspect compile/collector/restore, no automatic stop")
            if job['resource']=='gpu' and time.monotonic()-last_samples>=1:
                try:
                    from scripts.gpu_efficiency_observe import process_snapshot,gpu_snapshot
                    ps=process_snapshot()
                    sample=dict(unix=time.time(),gpus=gpu_snapshot(ps),job=job['id'])
                    sample_dir=BASE/'gpu_samples';sample_dir.mkdir(exist_ok=True)
                    with (sample_dir/f'{job["id"]}.jsonl').open('a') as out:out.write(json.dumps(sample)+'\n')
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    atomic_json(BASE/'GPU_SAMPLE_ERROR.json',dict(job=job['id'],error=type(exc).__name__,unix=time.time()))
                last_samples=time.monotonic()
            progress=ROOT/job.get('progress',job['complete'])
            if job['id'] not in first_seen:
                if (observation.get('completed_iterations') or 0)>0 or (observation.get('completed_games') or 0)>0:
                    job['first_real_work_evidence']=dict(path=str(progress.relative_to(ROOT)),record=observation)
                    first_seen.add(job['id']);notify(job['id'],'首个有效步骤',str(progress.relative_to(ROOT)))
            if campaign.get('js_scheduler_handoff',{}).get('job')==job['id'] and not campaign['js_scheduler_handoff'].get('complete'):continue
            code=process.poll()
            if code is None:continue
            stream.close();job['exit_code']=code;job['terminal_unix']=time.time()
            if code==0 and job_completion_passes(job):
                job['status']='COMPLETE';job['terminal_evidence']=dict(path=job['complete'],sha256=sha(ROOT/job['complete']))
            elif code==78 and job.get('execution_revision')=='real_cpu_producer_v1' and (Path(job['complete']).parent/'BLOCKED.json').exists():
                job['status']='BLOCKED_PROTOCOL'
                job['terminal_evidence']=dict(path=str(Path(job['complete']).parent/'BLOCKED.json'),
                    reason='CPU/GPU scope mismatch in frozen Bank-ratio contract; no training fabricated')
            else:job['status']='FAILED'
            notify(job['id'],job['status'],f'exit={code}; artifact={job["complete"]}')
            del active[lane]
        for lane in ('gpu','cpu','analysis','evaluation','js'):
            if sum(j['resource']==lane for j,_,_ in active.values())>=caps[lane]:continue
            ready=eligible(campaign['jobs'],lane,paused=paused)
            if not ready:continue
            job=ready[0]
            if job_completion_passes(job):
                job['status']='COMPLETE';job['reconciled_existing_completion']=True;continue
            if lane=='gpu':
                if time.monotonic()-last_gpu_check<10:continue
                snapshot=gpu_free_snapshot();last_gpu_check=time.monotonic()
                atomic_json(BASE/'GPU_RESOURCE_STATUS.json',dict(**snapshot,unix=time.time(),next_job=job['id']))
                owned_resume=job['id']=='BANK-RATIO-GPU-SLOTS-V1' and job.get('allow_owned_slot_resume') and bank_owned_slots_only()
                if not snapshot['free'] and not owned_resume:
                    campaign['gpu_state']='WAITING_EXTERNAL_GPU_TASKS';continue
                if job['id'].startswith('GPU-P1-'):
                    from scripts.gpu_efficiency_performance_probe import verify_handoff
                    verify_handoff()
            mismatch=[path for path,expected in job.get('frozen_files',{}).items()
                      if not (ROOT/path).exists() or sha(ROOT/path)!=expected]
            if mismatch:
                job.update(status='BLOCKED_TECHNICAL',blocked_unix=time.time(),
                    blocked_reason='Frozen source mismatch',mismatched_files=mismatch)
                atomic_json(BASE/'technical_blocks'/f'{job["id"]}.json',dict(job=job['id'],
                    reason='Frozen source mismatch; explicit technical rebind required',files=mismatch,unix=time.time()))
                notify(job['id'],'BLOCKED_TECHNICAL','Frozen source mismatch; no worker launched')
                continue
            env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONPATH='.',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
            if lane!='gpu':env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
            env.update(job.get('env',{}))
            if lane!='gpu':
                env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
            slot=scheduling_slot(lane,active)
            key=lane if lane=='gpu' else f'{lane}:{slot}'
            group=cpu_group(allowed,lane,slot,config.get('cpu_partition_offset',12)) if lane!='gpu' else job.get('cpu_affinity',[])
            command=[sys.executable,*job['args']]
            if group:command=['taskset','-c',','.join(map(str,group)),*command]
            log=BASE/'logs'/f'{job["id"]}.log';log.parent.mkdir(parents=True,exist_ok=True)
            stream=log.open('a')
            process=subprocess.Popen(command,cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            job.update(status='RUNNING',started_unix=time.time(),owner='campaign_owner',log=str(log.relative_to(ROOT)))
            if lane=='gpu':campaign['gpu_state']='RUNNING'
            job['scheduling_slot']=slot
            if group:job['cpu_affinity']=group
            active[key]=(job,process,stream)
            notify(job['id'],'STARTED','awaiting actual progress evidence')
        if 'gpu' not in active and campaign.get('gpu_state')=='RUNNING':
            campaign['gpu_state']='IDLE_DEPENDENCIES_OR_TECHNICAL_WORK'
        if 'gpu' in active:
            gpu_idle_since=None;gpu_idle_notified=False
        elif time.monotonic()-last_idle_check>=10:
            last_idle_check=time.monotonic()
            try:
                snapshot=gpu_free_snapshot()
                atomic_json(BASE/'GPU_RESOURCE_STATUS.json',dict(**snapshot,unix=time.time(),next_job=None))
                if snapshot['free']:
                    if gpu_idle_since is None: gpu_idle_since=time.time()
                    idle_seconds=time.time()-gpu_idle_since
                    idle_reason=gpu_idle_reason(campaign['jobs'])
                    campaign['gpu_idle_reason']=idle_reason
                    if idle_reason=='NO_REGISTERED_GPU_SUCCESSOR':
                        campaign['gpu_state']='GPU_BUDGETS_COMPLETE_CPU_ONLY_REMAINDER'
                    if idle_seconds>=300 and (not gpu_idle_notified or
                            (idle_reason!='NO_REGISTERED_GPU_SUCCESSOR' and idle_seconds%300<2)):
                        notify('GPU-CAMPAIGN','GPU空闲超过5分钟',
                            f'{idle_seconds:.0f}s free; reason={idle_reason}; registered_ready={ [j["id"] for j in eligible(campaign["jobs"],"gpu",paused=paused)] }; paused={paused}; no unregistered work launched')
                        gpu_idle_notified=True
                else:
                    gpu_idle_since=None;gpu_idle_notified=False
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                atomic_json(BASE/'GPU_RESOURCE_ERROR.json',dict(error=type(exc).__name__,unix=time.time()))
        campaign['updated_unix']=time.time();atomic_json(BASE/'CAMPAIGN.json',campaign)
        if (BASE/'EVAL_PARALLEL_CONFIG.json').exists():
            try:
                from scripts.campaign_eval_monitor import tick as evaluation_monitor_tick
                evaluation_monitor_tick(campaign,active,ROOT,BASE)
            except Exception as exc:
                atomic_json(BASE/'EVAL_PARALLEL_MONITOR_ERROR.json',dict(
                    error=type(exc).__name__,unix=time.time(),
                    training_and_evaluation_workers_untouched=True))
        atomic_json(BASE/'EXECUTION_STATUS.json',dict(stage='PAUSED' if paused else 'SUPERVISING',
            active={lane:job['id'] for lane,(job,_,_) in active.items()},
            statuses={j['id']:j['status'] for j in campaign['jobs']},
            gpu_state=campaign.get('gpu_state'),unix=time.time(),old_queues_resumed=False,
            pending_evidence={j['id']:[path for path in j.get('ready_files',[]) if not (ROOT/path).exists()]
                              for j in campaign['jobs'] if j['status']=='REGISTERED' and j.get('ready_files')}))
        if all(j['status']=='COMPLETE' for j in campaign['jobs']):
            atomic_json(BASE/'COMPLETE.json',dict(passed=True,jobs=len(campaign['jobs']),unix=time.time()));return
        time.sleep(2)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        try:
            stamp=time.time_ns()
            atomic_json(BASE/'owner_failures'/f'{stamp}.json',dict(
                schema='qqt.authorized-campaign-owner-failure/v1',
                exception=type(exc).__name__,message=str(exc).replace(str(ROOT),'<WORKTREE>')[:500],
                traceback_frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)
                                  for f in traceback.extract_tb(exc.__traceback__)],
                campaign_sha256=sha(BASE/'CAMPAIGN.json') if (BASE/'CAMPAIGN.json').exists() else None,
                status_mtime=(BASE/'EXECUTION_STATUS.json').stat().st_mtime
                    if (BASE/'EXECUTION_STATUS.json').exists() else None,
                unix=time.time()))
        except BaseException:
            pass
        raise
