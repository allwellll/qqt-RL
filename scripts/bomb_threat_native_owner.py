"""One GPU2-only auxiliary queue; never consumes legacy training/eval queues."""
import json,os,subprocess,sys,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.combat_mix_owner import read
from scripts.bomb_threat_owner import invalid_ancestry
from scripts.bomb_threat_eval_owner import catchup_order,request_kind
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def active():
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            if b'scripts.bomb_threat_native_eval' in (d/'cmdline').read_bytes().split(b'\0') and (d/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def main():
    if (BASE/'native_fullgame_v2/PROTOCOL.json').exists():
        from scripts.bomb_threat_hunter_eval import run_owner
        return run_owner()
    guard=lock(BASE/'NATIVE_OWNER.lock');child=None
    while not any((BASE/n).exists() for n in ('USER_STOP_NATIVE.json','USER_STOP.json','USER_PAUSE.json')):
        if child is not None and child[0].poll() is not None:
            proc,stream,out=child;stream.close()
            if proc.returncode:atomic_json(out/'FAILURE.json',dict(exit_code=proc.returncode,unix=time.time()))
            else:atomic_json(BASE/'METRICS_REFRESH.json',dict(status='RETRY_PENDING',unix=0))
            child=None
        newest=max((p.stat().st_mtime for p in (BASE/'eval').glob('*/native/NATIVE_COMPLETE.json')),default=0)
        if newest>read(BASE/'METRICS_REFRESH.json',{}).get('unix',0):
            atomic_json(BASE/'METRICS_REFRESH.json',dict(status='RETRY_PENDING',unix=0))
        if active():
            atomic_json(BASE/'NATIVE_STATUS.json',dict(status='RUNNING',dedicated_gpu=2,old_queues_consumed=False,unix=time.time()))
            time.sleep(5);continue
        requests=catchup_order(list((BASE/'eval_requests').glob('*.json')))
        for p in requests:
            req=read(p,{})
            if not req.get('checkpoint') or request_kind(req)=='technical_boundary':continue
            cp=ROOT/req['checkpoint'];run=cp.parent
            if run.name.startswith('bt_') and (invalid_ancestry(run.name) or (run/'EXCLUDED_EVALUATION.json').exists()):continue
            out=BASE/'eval'/p.stem/'native';out.mkdir(parents=True,exist_ok=True)
            if any((out/n).exists() for n in ('NATIVE_COMPLETE.json','FAILURE.json')):continue
            # Physical card is reserved by the training owner. Never evict an
            # unknown compute application or toggle any keepalive here.
            lines=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if lines:break
            log=(BASE/'logs'/f'native_{p.stem}.log').open('ab',buffering=0)
            proc=subprocess.Popen(['taskset','-c','68-71',sys.executable,'-m','scripts.bomb_threat_native_eval',str(p.relative_to(ROOT))],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            child=(proc,log,out)
            atomic_json(out/'LAUNCH.json',dict(gpu=2,request=str(p.relative_to(ROOT)),unix=time.time()))
            break
        atomic_json(BASE/'NATIVE_STATUS.json',dict(status='RUNNING' if active() else 'IDLE',dedicated_gpu=2,old_queues_consumed=False,unix=time.time()))
        time.sleep(5)

if __name__=='__main__':main()
