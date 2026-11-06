"""Durable unique G owner/watchdog; adopts workers and continues bounded chunks."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

BASE=Path('runs/combat_specialist_league_20261101')
MODULE='scripts.combat_specialist_g_league_train'
SELF='scripts.combat_specialist_g_owner'


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def running(module, run=None):
    total=0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            words=(proc/'cmdline').read_bytes().split(b'\0')
            state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state!='Z' and module.encode() in words and (run is None or run.encode() in words):total+=1
        except OSError:pass
    return total


def paused():
    return any((BASE/name).exists() for name in ('USER_STOP','USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))


def notify(key,message,**evidence):
    path=BASE/'notifications'/('G_'+key+'.json')
    if path.exists():return
    atomic_json(path,dict(message=message,**evidence,unix=time.time()))
    result=subprocess.run(['jdme-notify','send',message],capture_output=True)
    atomic_json(path,dict(message=message,notification_returncode=result.returncode,**evidence,unix=time.time()))


def launch(module, args, log, cpu='72-79',gpu='0'):
    argv=['taskset','-c',cpu,sys.executable,'-u','-m',module]+args
    env=dict(os.environ,JAXBOMB_DEVICE_OVERRIDE=gpu,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1')
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('ab') as stream:
        return subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def owner():
    guard=lock(BASE/'G_OWNER.lock')
    current=read(BASE/'G_CURRENT.json',{'protocol':str(BASE/'G_DYNAMIC_PROTOCOL.json'),'run':'G_league','chunk':0})
    child=None;idle_since=time.time()
    while not paused():
        protocol_path=Path(current['protocol']);p=read(protocol_path);run=current['run'];out=Path(p['namespace'])/run
        out.mkdir(parents=True,exist_ok=True)
        active=running(MODULE,run)
        if active>1:
            notify('duplicate_'+run,'G联赛检测到重复worker，停止新派发并保留现场。');return
        progress=read(out/'PROGRESS.json',{});row=progress.get('last',{})
        atomic_json(BASE/'G_EXECUTION_STATUS.json',dict(status='WORKER_ACTIVE' if active else 'BETWEEN_STAGES',
            run=run,protocol=str(protocol_path),workers=active,stage=read(out/'CYCLE_STAGE.json'),
            last_update=progress.get('iterations'),env_per_second=row.get('env_per_second'),unix=time.time()))
        if active:
            idle_since=time.time()
            if row.get('finite') and min(row.get('update_l2',{'actor':0,'critic':0}).values())>0 and row.get('ability_update_l2',0)>0:
                acceptance=BASE/(run+'_FIRST_UPDATE.json')
                if not acceptance.exists():
                    atomic_json(acceptance,dict(status='PASS',row=row,restore=read(out/'RESTORE.json'),unix=time.time()))
                    notify('first_'+run,f"G联赛{run}真实u{row['new_update']}更新：Actor/Critic/ability非零finite，262144新步；冷周期{row['cycle_seconds']:.1f}s，继续恢复验收和固定预算。",receipt=str(acceptance))
            time.sleep(10);continue
        if child is not None:
            code=child.wait();child=None
            if code:
                atomic_json(out/'TECHNICAL_BLOCKED.json',dict(status='BLOCKED',returncode=code,
                    last_committed=[str(x) for x in sorted(out.glob('u*/COMMITTED.json'))[-1:]],log=str(out/'worker.log'),unix=time.time()))
        # A committed full-budget checkpoint is authoritative even if a
        # post-save receipt writer failed and left a stale worker error.
        if (out/'COMPLETE.json').exists():
            result=read(out/'COMPLETE.json')
            assert sha(Path(result['checkpoint'])/'state.pkl')==result['state_sha256']
            notify('complete_'+run,f"G联赛{run}固定预算完成，下一1280更新同配方自动接续；CPU评估异步。",receipt=str(out/'COMPLETE.json'))
            count=current['chunk']+1;next_run=f'G_league_c{count:03d}'
            p['arms']={next_run:p['arms'][run]};p['continuation_source']=result
            p['chunk_start_update']=result['update'];p['chunk_updates']=1280
            next_path=BASE/'protocols'/(next_run+'.json');atomic_json(next_path,p)
            current=dict(protocol=str(next_path),run=next_run,chunk=count)
            atomic_json(BASE/'G_CURRENT.json',current);continue
        if (out/'TECHNICAL_BLOCKED.json').exists():
            notify('blocked_'+run,'G联赛技术阻塞，已保留原子点与worker.log；未重复smoke或占卡伪装训练。',receipt=str(out/'TECHNICAL_BLOCKED.json'))
            atomic_json(BASE/'G_EXECUTION_STATUS.json',dict(status='TECHNICAL_BLOCKED',run=run,unix=time.time()));return
        stage='SMOKE' if not (out/'SMOKE.json').exists() else 'RESTORE' if not (out/'RESTORE.json').exists() else 'FORMAL'
        args=[str(protocol_path),run,str(p['seed'])]+(['--smoke'] if stage=='SMOKE' else ['--verify-restore'] if stage=='RESTORE' else [])
        child=launch(MODULE,args,out/'worker.log')
        atomic_json(out/'LAUNCH.json',dict(status='LAUNCHED_NOT_UPDATE_EVIDENCE',stage=stage,
            gpu=0,cpu_affinity='72-79',module=MODULE,args=args,protocol_sha256=sha(protocol_path),unix=time.time()))
        if time.time()-idle_since>300:notify('idle_'+run,'G联赛阶段切换空闲超过5分钟，检查编译/恢复证据。')
        time.sleep(5)
    atomic_json(BASE/'G_EXECUTION_STATUS.json',dict(status='USER_PAUSED_NO_NEW_LAUNCH',unix=time.time()))


def watchdog():
    guard=lock(BASE/'G_WATCHDOG.lock');child=None
    while not paused():
        status=read(BASE/'G_EXECUTION_STATUS.json',{})
        if status.get('status')=='TECHNICAL_BLOCKED':return
        if not running(SELF,'--owner'):
            if child is not None:
                code=child.wait()
                atomic_json(BASE/'G_OWNER_RESTART_RECEIPT.json',dict(reason='owner exited; surviving worker adopted on restart',returncode=code,unix=time.time()))
            child=launch(SELF,['--owner'],BASE/'G_OWNER.log')
        time.sleep(15)


if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    if '--watchdog' in sys.argv:watchdog()
    else:owner()
