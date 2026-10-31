"""Durable native lanes. Dedicated GPUs are explicitly excluded from training.

Shared mode yields only evaluators and recovers its cap after healthy cooldown.
Dedicated mode has a fixed lane ceiling and never applies a training contention
penalty. Neither mode sends signals to any training or evaluation worker.
"""
import json, os, subprocess, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock
from scripts.iter_pbt_owner import notify, read, live_workers
from scripts.iter_pbt_native_gpu_policy import (
    dedicated_gpus, gpu_lock_name, native_priority, reduce_concurrency,
    recover_concurrency, evaluation_slots, returnable_gpus,
)
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/iter_pbt_v1'
B=BASE/'native_gpu'

class Adopted:
    def __init__(self,identity,out):
        self.proc=Path('/proc')/str(identity);self.out=out;self.returncode=None
    def poll(self):
        try:
            if self.proc.exists() and (self.proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return None
        except OSError:pass
        self.returncode=0 if (self.out/'NATIVE_COMPLETE.json').exists() or (self.out/'YIELDED.json').exists() else 1
        return self.returncode

def native_workers(procroot=Path('/proc')):
    """Adopt both formal workers and pre-handover temporary lane wrappers."""
    found={};parents={}
    for d in procroot.iterdir():
        if not d.name.isdigit():continue
        try:
            args=(d/'cmdline').read_bytes().split(b'\0')
            formal=b'scripts.iter_pbt_native_eval' in args
            legacy=any(Path(a.decode()).name=='native_eval_lane.py' for a in args if a)
            if not (formal or legacy):continue
            stat=(d/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]=='Z':continue
            paths=[a.decode() for a in args if a.startswith(b'runs/iter_pbt_v1/eval_requests/')]
            if not paths:continue
            name=Path(paths[0]).stem;identity=int(d.name);parent=int(stat[1])
            if name in found:
                if parent==found[name]['identity']:continue
                if parents[name]!=identity:raise RuntimeError('duplicate native evaluator '+name)
            env=dict(v.split(b'=',1) for v in (d/'environ').read_bytes().split(b'\0') if b'=' in v)
            gpu=int(args[args.index(b'--gpu')+1]) if b'--gpu' in args else int(env.get(b'CUDA_VISIBLE_DEVICES',b'0'))
            lane=int(args[args.index(b'--lane')+1]) if b'--lane' in args else int(env[b'DRAIN_LANE']) if b'DRAIN_LANE' in env else None
            found[name]=dict(identity=identity,gpu=gpu,lane=lane);parents[name]=parent
        except (OSError,ValueError):continue
    return found

def lane_busy(base,gpu,lane):
    try:guard=lock(base/gpu_lock_name(gpu,lane))
    except BlockingIOError:return True
    guard.close();return False

def samples_after(gpu,started):
    rows=[]
    for p in (BASE/'formal').glob(f'slot{gpu:02d}_c*/updates.jsonl'):
        if p.stat().st_mtime<started:continue
        for line in p.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if r.get('unix',0)>started and 0<r.get('cycle_seconds',0)<30:rows.append(r)
    return sorted(rows,key=lambda r:r['unix'])[-3:]

def main():
    guard=lock(B/'OWNER.lock');children={};last_sample=0
    while not any((BASE/n).exists() for n in ('USER_STOP.json','USER_PAUSE.json')):
        config=read(B/'CONFIG.json',{});dedicated=dedicated_gpus(config)
        enabled=config.get('enabled',False) and read(B/'PARITY_GATE.json',{}).get('status')=='PASS'
        ceiling=max(1,config.get('concurrency_ceiling',6 if dedicated else 8))
        cap=min(ceiling,max(1,config.get('max_concurrent',ceiling)))
        # There is no colocated training to throttle in dedicated mode.
        if dedicated:cap=ceiling
        for n,v in native_workers().items():
            if n in children or not (BASE/'native_requests'/f'{n}.json').exists():continue
            out=BASE/'eval'/n/'native';launch=read(out/'NATIVE_LAUNCH.json',{})
            children[n]=dict(proc=Adopted(v['identity'],out),stream=None,started=launch.get('unix',time.time()),**v)
        for n,v in list(children.items()):
            out=BASE/'eval'/n/'native';p=v['proc']
            if p.poll() is not None:
                if v['stream']:v['stream'].close()
                del children[n]
                if p.returncode and not (out/'YIELD_GPU.json').exists():
                    atomic_json(out/'NATIVE_FAILED.json',dict(exit_code=p.returncode,unix=time.time()))
                    notify(n+'_NATIVE_FAILED',f'ITER-PBT原生评估失败：{n}，训练不停止，证据保留。')
                continue
            progress=read(out/'NATIVE_PROGRESS.json',{})
            if time.time()-progress.get('unix',v['started'])>900:
                notify(n+'_NATIVE_STALE',f'ITER-PBT原生评估15分钟无scan进展：{n}；训练继续。')
            if v['gpu'] in dedicated:continue
            rates=[262144/r['cycle_seconds'] for r in samples_after(v['gpu'],v['started'])]
            baseline=config.get('baseline_env_per_second',{}).get(str(v['gpu']),20000)
            low=len(rates)>=3 and reduce_concurrency(2,baseline,rates)==1
            if low and not (out/'YIELD_GPU.json').exists():
                cap=reduce_concurrency(cap,baseline,rates)
                config.update(last_change_unix=time.time(),last_reduction_unix=time.time(),
                              throttled_gpus=sorted(set(config.get('throttled_gpus',[]))|{v['gpu']}))
                atomic_json(out/'YIELD_GPU.json',dict(reason='training throughput below 95%; evaluation-only cooperative yield',baseline_env_per_second=baseline,observed=rates,unix=time.time()))
                notify(n+'_THROUGHPUT_GUARD',f'ITER-PBT共卡评估降并发至{cap}，下限1；仅评估协作让卡，冷却后恢复。',observed=rates)
        if not dedicated:
            # Every affected GPU must supply three *new* healthy warm updates.
            changed=config.get('last_change_unix',config.get('last_reduction_unix',0))
            affected=config.get('throttled_gpus',[])
            healthy=bool(affected)
            for gpu in affected:
                rates=[262144/r['cycle_seconds'] for r in samples_after(gpu,changed)]
                baseline=config.get('baseline_env_per_second',{}).get(str(gpu),20000)
                healthy &= len(rates)>=3 and __import__('statistics').median(rates)>=baseline*.95
            newcap=recover_concurrency(cap,ceiling,1,[1]*3 if healthy else [],time.time(),changed,config.get('recovery_cooldown_seconds',300))
            if newcap>cap:config['last_change_unix']=time.time();cap=newcap
        if config.get('max_concurrent')!=cap:
            config['max_concurrent']=cap;atomic_json(B/'CONFIG.json',config)
        elif config.get('last_change_unix',0)>config.get('configuration_saved_unix',0):
            config['configuration_saved_unix']=time.time();atomic_json(B/'CONFIG.json',config)
        if time.time()-last_sample>5:
            r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            memory={}
            for line in r.stdout.splitlines():
                try:i,m=line.split(',');memory[int(i)]=float(m.strip())
                except ValueError:continue
            samples=[]
            for n,v in children.items():
                used=memory.get(v['identity'])
                samples.append(dict(run=n,gpu=v['gpu'],lane=v['lane'],memory_MiB=used))
                if used is not None and used>config.get('memory_cap_MiB',6144):
                    out=BASE/'eval'/n/'native'
                    if not (out/'YIELD_GPU.json').exists():
                        atomic_json(out/'YIELD_GPU.json',dict(reason='evaluation memory budget; cooperative yield only',memory_MiB=used,unix=time.time()))
                        notify(n+'_MEMORY_GUARD',f'ITER-PBT评估显存超预算：{n}，请求scan边界协作退出，不发进程信号。')
            with (B/'SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),evaluators=samples))+'\n')
            last_sample=time.time()
        pending=[read(q) for q in (BASE/'native_requests').glob('*.json')]
        recovery=set(read(BASE/'NATIVE_EVAL_RECOVERY_QUEUE.json',{}).get('names',[]))
        recovery_pending=any(not (BASE/'eval'/n/'native/NATIVE_COMPLETE.json').exists() for n in recovery)
        borrowed=set(config.get('temporary_borrowed_gpus',[]))
        waiting_recovery=any(not (BASE/'eval'/n/'native/NATIVE_COMPLETE.json').exists() and n not in children for n in recovery)
        returned=returnable_gpus(config,{v['gpu'] for v in children.values()},waiting_recovery)
        if returned and recovery:
            # Persistent bounded borrowing: no dependency on an interactive
            # session to return a card, and no interruption of an active lane.
            dedicated-=returned
            config.update(dedicated_gpus=sorted(dedicated),temporary_borrowed_gpus=sorted(borrowed-returned),
                          concurrency_ceiling=len(dedicated)*config.get('lanes_per_gpu',6),
                          max_concurrent=len(dedicated)*config.get('lanes_per_gpu',6))
            atomic_json(B/'CONFIG.json',config)
            receipt=read(BASE/'NATIVE_RECOVERY_GPU_BORROW.json',{})
            receipt.update(status='RETURNED_TO_TRAINING' if not borrowed-returned else 'PARTIALLY_RETURNED',
                           returned_gpus=sorted(set(receipt.get('returned_gpus',[]))|returned),
                           return_unix=time.time(),returned_cards_have_no_active_lanes=True,updates_lost=0)
            atomic_json(BASE/'NATIVE_RECOVERY_GPU_BORROW.json',receipt)
            cap=config['max_concurrent']
            notify('NATIVE_RECOVERY_BORROW_RETURNED_'+str(sorted(returned)),f'ITER-PBT临时评估卡{sorted(returned)}在途任务已完成，现已归还训练；其余评估继续，未中断更新。',returned_gpus=sorted(returned))
        # A borrowed GPU is reserved immediately by the training owner, but its
        # current chunk keeps running. Never evaluate until that worker exits.
        training_gpus={int(n[4:6]) for n in live_workers('scripts.iter_pbt_train')}
        for r in sorted(pending,key=lambda r:native_priority(r,config.get('first_requests',[]))):
            n=r['name'];out=BASE/'eval'/n/'native'
            if n in children or (out/'NATIVE_COMPLETE.json').exists() or (out/'NATIVE_FAILED.json').exists():continue
            if r.get('selection_eligible') is False:continue
            if (BASE/'eval'/n/'SKIPPED_SUPERSEDED.json').exists() and n not in recovery:continue
            if not enabled or len(children)>=cap:break
            used={(v['gpu'],v['lane']) for v in children.values()}
            slots=evaluation_slots(config,training_gpus,recovery_pending)
            slot=next(((g,i) for g,i in slots if (g,i) not in used and not lane_busy(BASE,g,i)),None)
            if slot is None:break
            gpu,lane=slot;out.mkdir(parents=True,exist_ok=True)
            for stale in ('YIELD_GPU.json','YIELDED.json'):
                p=out/stale
                if p.exists():p.rename(out/f'{p.stem}_{time.time_ns()}.json')
            stream=(B/f'{n}.log').open('ab',buffering=0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            affinities=config.get('lane_cpu_affinity',{}).get(str(gpu),[])
            core=str(affinities[lane]) if lane is not None and lane<len(affinities) else str(184+lane if lane is not None else 184+gpu)
            command=['taskset','-c',core,sys.executable,'-m','scripts.iter_pbt_native_eval',str((BASE/'eval_requests'/f'{n}.json').relative_to(ROOT)),'--gpu',str(gpu)]
            if lane is not None:command+=['--lane',str(lane)]
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            now=time.time();children[n]=dict(proc=p,stream=stream,gpu=gpu,lane=lane,started=now,identity=p.pid)
            atomic_json(out/'NATIVE_LAUNCH.json',dict(gpu=gpu,lane=lane,cpu_affinity=core,unix=now,formal_owner=True))
        completed={}
        for result in (BASE/'eval').glob('*/native/NATIVE_SUMMARY.json'):
            v=read(result,{})
            if v.get('backend')=='gpu':completed[result.parent.parent.name]=dict(elapsed_seconds=v.get('elapsed_seconds'),games=v.get('natural_jax'),scan_seconds=v.get('scan_seconds'),completed_unix=v.get('unix'))
        active={n:dict(gpu=v['gpu'],lane=v['lane'],age_seconds=time.time()-v['started']) for n,v in children.items()}
        atomic_json(B/'REPORT.json',dict(completed_GPU_points=completed,lag=read(BASE/'EVAL_LAG.json',{}),max_concurrent=cap,dedicated_gpus=sorted(dedicated),updated_unix=time.time()))
        atomic_json(B/'STATUS.json',dict(enabled=enabled,max_concurrent=cap,active=active,dedicated_gpus=sorted(dedicated),updated_unix=time.time(),training_signal_authority=False,worker_signals=False))
        time.sleep(5)
if __name__=='__main__':main()
