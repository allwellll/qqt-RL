"""Single durable consumer for the bounded COMBAT-MIX-V1 queue."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/combat_mix_v1'

def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return default

def run_output(name):
    spec=read(BASE/'protocols'/f'{name}.json',{})
    return ROOT/spec['namespace']/name if 'namespace' in spec else BASE/'formal'/name

def workers():
    result={}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            argv=(d/'cmdline').read_bytes().split(b'\0')
            if not any(m.encode() in argv for m in ('scripts.campaign_scale_train','scripts.combat_mix_train','scripts.combat_mix_legacy_train','scripts.combat_mix_legacy_native_train')):continue
            paths=[a.decode() for a in argv if a.startswith(b'runs/combat_mix_v1/protocols/')]
            if not paths:continue
            name=Path(paths[0]).stem
            if (d/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            env=dict(v.split(b'=',1) for v in (d/'environ').read_bytes().split(b'\0') if b'=' in v)
            if name in result:raise RuntimeError('duplicate COMBAT worker: '+name)
            result[name]=int(env[b'JAXBOMB_DEVICE_OVERRIDE'])
        except (OSError,ValueError,KeyError):continue
    return result

def notify(name,message,details):
    marker=BASE/'notifications'/f'{name}.json'
    if marker.exists():return
    atomic_json(marker,{**details,'unix':time.time(),'message':message})
    try:
        proc=subprocess.run(['jdme-notify','send',message],capture_output=True,timeout=20)
        atomic_json(marker.with_suffix('.delivery.json'),dict(exit_code=proc.returncode,unix=time.time()))
    except (OSError,subprocess.TimeoutExpired) as e:
        atomic_json(marker.with_suffix('.delivery.json'),dict(error=type(e).__name__,unix=time.time()))

def progress(name):
    p=read(run_output(name)/'PROGRESS.json',{})
    r=p.get('last',{})
    return dict(update=p.get('iterations'),finite=r.get('finite'),update_l2=r.get('update_l2'),
        environment_steps=r.get('environment_steps'),cycle_seconds=r.get('cycle_seconds'),
        env_per_second=r.get('actual_environment_steps_this_update',0)/r['cycle_seconds'] if r.get('cycle_seconds') else None,
        age_seconds=time.time()-r['unix'] if 'unix' in r else None)

def resources():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()
    rows=[]
    for line in result.stdout.splitlines():
        g,uid,mem,util,power=[x.strip() for x in line.split(',')]
        rows.append(dict(gpu=int(g),memory_mib=float(mem),util_percent=float(util),power_w=float(power),occupied=uid in apps))
    return rows

def paused():
    paths=[BASE/n for n in ('USER_STOP.json','USER_PAUSE.json')]+[ROOT/'runs/authorized_campaign_20261026'/n for n in ('USER_STOP.json','USER_PAUSE.json')]
    return [str(p.relative_to(ROOT)) for p in paths if p.exists()]

def technical_failure(name):
    """Consume persistent failure even when a predecessor adopted the worker."""
    out=run_output(name)
    failed=read(out/'TECHNICAL_FAILURE.json')
    override=read(out/'USER_SMOKE_OVERRIDE.json',{})
    repair=read(out/'TECHNICAL_REPAIR.json',{})
    if failed and repair.get('superseded_failure_sha256')==sha(out/'TECHNICAL_FAILURE.json'):
        failed=None
    elif failed and override.get('passed') and override.get('threshold')==.5 and failed.get('unix',0)<=override.get('unix',0):
        failed=None  # Only the older frozen85 smoke failure is superseded.
    if failed:return failed
    smoke=read(out/'SMOKE.json')
    if smoke is not None and not smoke.get('passed'):
        return dict(status=smoke.get('status','FAILED_SMOKE'),source='persisted_smoke',
                    receipt=str((out/'SMOKE.json').relative_to(BASE)))
    arm=name.rsplit('_s',1)[0]
    hold=read(BASE/'ARM_TECHNICAL_HOLD.json',{}).get(arm)
    if hold and str(hold.get('only_seed',''))!=name.rsplit('_s',1)[1]:
        return dict(**hold,source='arm_technical_hold')
    return None

def worker_reserved_devices(procroot=Path('/proc')):
    """CUDA initialization is not yet visible in nvidia-smi compute apps."""
    reserved=set()
    for d in procroot.iterdir():
        if not d.name.isdigit():continue
        try:
            args=(d/'cmdline').read_bytes().split(b'\0')
            if not any(a.startswith(b'scripts.') and (b'train' in a or b'probe' in a) for a in args):continue
            if (d/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            env=dict(x.split(b'=',1) for x in (d/'environ').read_bytes().split(b'\0') if b'=' in x)
            for g in env.get(b'JAXBOMB_DEVICE_OVERRIDE',env.get(b'CUDA_VISIBLE_DEVICES',b'')).split(b','):
                if g.isdigit():reserved.add(int(g))
        except (OSError,ValueError):continue
    return reserved

def main():
    guard=lock(BASE/'OWNER.lock')
    protocol=read(BASE/'PROTOCOL.json');seeds=protocol['seeds']
    order=protocol.get('queue_order',[f'{arm}_s{seed}' for arm in protocol['arms'] for seed in seeds])
    cpu={str(g):','.join(str(x) for x in range(32+8*g,40+8*g)) for g in range(8)}
    cpu.update({'3':'72-79','4':'80-87','5':'88-95','6':'96-103','7':'104-111'})
    atomic_json(BASE/'OWNER_ACCEPTANCE.json',dict(unique_lock=True,adopts_live_controls=True,bounded_runs=len(order),
        cpu_affinity=cpu,legacy_queues_resumed=False,protocol_sha256=sha(BASE/'PROTOCOL.json'),unix=time.time()))
    children={};complete={};idle={};lastsample=0
    while True:
        # Latest user authorization relaxes only the legacy candidate throughput.
        # Do not mutate the sealed trainer or discard its original smoke evidence.
        legacy=run_output('legacy_mix_s2026103051')
        original=read(legacy/'SMOKE.json')
        if original and not (legacy/'USER_SMOKE_OVERRIDE.json').exists() and (BASE/'USER_PBT_QUEUE_OVERRIDE.json').exists():
            atomic_json(legacy/'SMOKE_FROZEN85.json',original)
            fraction=original.get('fraction',0)
            metrics=original.get('checkpoint',{}).get('metrics',{})
            valid=metrics.get('finite',False) and all(v>0 for v in metrics.get('update_l2',{}).values()) and bool(metrics.get('update_l2'))
            passed=valid and fraction>=.5
            override=dict(passed=passed,threshold=.5,measured_fraction=fraction,authorization='.hermes/iter_pbt_plan_0045.md',original='SMOKE_FROZEN85.json',unix=time.time())
            atomic_json(legacy/'USER_SMOKE_OVERRIDE.json',override)
            if passed:
                atomic_json(legacy/'SMOKE.json',dict(**{k:v for k,v in original.items() if k not in ('passed','status')},passed=True,status='PASS_USER_50_PERCENT_OVERRIDE',user_override=override))
            notify('LEGACY_USER50_RECLASSIFICATION',f'legacy候选实测基线比例{fraction:.3f}，按用户50%吞吐要求'+('通过，fresh恢复后续训。' if passed else '未通过，保留证据并跳过；GPU交给PBT。'),override)
        holds_by_arm=read(BASE/'ARM_TECHNICAL_HOLD.json',{})
        for arm,hold in list(holds_by_arm.items()):
            seed=hold.get('only_seed')
            if seed is None:continue
            out=run_output(f'{arm}_s{seed}')
            smoke=read(out/'SMOKE.json',{});restore=read(out/'RESTORE.json',{})
            parity_ok=not hold.get('requires_native_parity') or read(BASE/'LEGACY_NATIVE_FULL_PARITY.json',{}).get('passed')
            if smoke.get('passed') and restore.get('passed') and parity_ok:
                atomic_json(BASE/f'{arm}_TECHNICAL_HOLD_RELEASED.json',dict(
                    previous_hold=hold,smoke=smoke,restore=restore,unix=time.time()))
                del holds_by_arm[arm]
                atomic_json(BASE/'ARM_TECHNICAL_HOLD.json',holds_by_arm)
                notify(arm+'_TECHNICAL_RECHECK_PASS',f'COMBAT {arm}独占技术复测及fresh恢复通过，释放原冻结三seed队列。',dict(smoke=smoke,restore=restore))
        latest=read(BASE/'PROTOCOL.json')
        if latest.get('queue_order')!=protocol.get('queue_order') or list(latest['arms'])!=list(protocol['arms']):
            protocol=latest
            order=protocol.get('queue_order',[f'{arm}_s{seed}' for arm in protocol['arms'] for seed in seeds])
        live=workers();pending=[];failures={};events=[]
        for name in order:
            out=run_output(name)
            if name in live:
                p=progress(name)
                if p['update'] is not None and p['update']>=162:
                    events.append((name+'_FIRST_UPDATES',f"COMBAT正式有效更新：{name} u{p['update']}，finite={p['finite']}，环境步/秒={p['env_per_second']:.0f}；CPU评估异步。",p))
                continue
            marker=read(out/'COMPLETE.json')
            if marker:
                if name not in complete:
                    cp=ROOT/marker['checkpoint']
                    assert marker['passed'] and marker['update']==320
                    assert sha(cp/'state.pkl')==marker['state_sha256']
                    assert sha(cp/'policy.pt')==marker['policy_sha256']
                    complete[name]=marker
                    atomic_json(BASE/'eval_requests'/f'{name}_u320.json',dict(**marker,run=name,
                        requested_splits=['dev','holdout'],JS_Hunter_games_per_tier=64,native_holdout_games=304,
                        teacher=False,action_modes=['categorical','greedy']))
                    events.append((name+'_COMPLETE',f'COMBAT固定run完成：{name} u320，新增41943040步，原子SHA验收；接续冻结队列。',marker))
                continue
            failed=technical_failure(name)
            if failed:failures[name]=failed
            else:pending.append(name)
        for name,(proc,stream) in list(children.items()):
            rc=proc.poll()
            if rc is None:continue
            stream.close();del children[name]
            if rc and name not in complete:
                data=dict(status='TECHNICAL_FAILURE',exit_code=rc,progress=progress(name),unix=time.time())
                atomic_json(run_output(name)/'TECHNICAL_FAILURE.json',data);failures[name]=data
                if name in pending:pending.remove(name)
                events.append((name+'_FAILED',f'COMBAT技术故障：{name} exit={rc}，已保留完整checkpoint与日志，需修复。',data))
        samples=resources();holds=paused();reserved=worker_reserved_devices()
        ready=read(BASE/'IMPLEMENTATION_READY.json',{})
        oldbase=ROOT/'runs/authorized_campaign_20261026/scale_followup_2155_v1'
        old_done=(oldbase/'RESULT.json').exists()
        old_status=read(oldbase/'STATUS.json',{})
        old_reserved=bool(old_status.get('queued')) and not old_done
        for sample in samples:
            g=sample['gpu']
            if sample['occupied'] or g in live.values() or g in reserved:
                idle.pop(g,None);continue
            idle.setdefault(g,time.time())
            if not pending or holds:
                if time.time()-idle[g]>300 and len(complete)<len(order):
                    events.append((f'GPU{g}_IDLE_{int(idle[g])}',
                        f'GPU{g}空闲超过5分钟：后继受技术检查或用户暂停约束，原因与恢复路径已落盘。',
                        dict(pending=pending,technical_failures=failures,pause_receipts=holds,
                             technical_rebind=read(BASE/'NATIVE_REBIND_STATUS.json',{}))))
                continue
            candidates=[n for n in pending if n.startswith('control_') or ready.get('status') in ('READY_FOR_SMOKE','PASS')]
            if g in (6,7) and old_reserved:candidates=[]
            if candidates:
                name=candidates[0];arm,seed=name.rsplit('_s',1)
                sp=BASE/'protocols'/f'{name}.json'
                if not sp.exists():continue
                module='scripts.campaign_scale_train' if arm=='control' else 'scripts.combat_mix_legacy_train' if arm=='legacy_mix' else 'scripts.combat_mix_train'
                module=read(sp,{}).get('worker_module',module)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),JAXBOMB_DEVICE_OVERRIDE=str(g),
                    JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                stream=(BASE/'logs'/f'{name}.log').open('ab',buffering=0)
                arguments=[str(sp.relative_to(ROOT)),arm,seed]
                stage='formal'
                if arm!='control':
                    out=run_output(name)
                    if not (out/'SMOKE.json').exists():
                        arguments.append('--smoke');stage='two_update_smoke'
                    elif not read(out/'SMOKE.json',{}).get('passed'):
                        continue
                    elif not (out/'RESTORE.json').exists():
                        arguments.append('--verify-restore');stage='fresh_restore'
                if stage=='formal' and read(sp,{}).get('requires_native_full_parity'):
                    if not read(BASE/'LEGACY_NATIVE_FULL_PARITY.json',{}).get('passed'):continue
                proc=subprocess.Popen(['taskset','-c',cpu[str(g)],sys.executable,'-m',module]+arguments,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                children[name]=(proc,stream);pending.remove(name);live[name]=g;idle.pop(g,None)
                events.append((name+'_'+stage+'_START',f'COMBAT冻结队列启动：{name} GPU{g}，stage={stage}，等待真实更新/恢复验收。',dict(gpu=g,run=name,module=module,stage=stage)))
            elif time.time()-idle[g]>300:
                events.append((f'GPU{g}_IDLE_{int(idle[g])}',f'GPU{g}空闲超过5分钟：COMBAT后继尚待接线/技术smoke，不用无关任务占卡。',dict(pending=pending,ready=ready)))
        if time.time()-lastsample>=10:
            with (BASE/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),gpus=samples))+'\n')
            lastsample=time.time()
        status='RUNNING' if live else 'COMPLETE' if len(complete)==len(order) else 'PAUSED' if holds else 'BLOCKED' if failures else 'WAITING_IMPLEMENTATION_OR_RESOURCE'
        atomic_json(BASE/'STATUS.json',dict(status=status,active=[dict(run=n,gpu=g,progress=progress(n)) for n,g in live.items()],
            pending=pending,complete=complete,failures=failures,pause_receipts=holds,updated_unix=time.time(),
            legacy_queues_resumed=False,last_gpu_sample=samples))
        for e in events:notify(*e)
        if status in ('COMPLETE','PAUSED'):
            atomic_json(BASE/f'{status}.json',dict(status=status,complete=complete,unix=time.time()))
            return
        time.sleep(5)

if __name__=='__main__':main()
