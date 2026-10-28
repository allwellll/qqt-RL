"""Bounded two-slot CPU JS consumer; never controls GPU training."""
import os,json,subprocess,sys,time
from pathlib import Path
from scripts.population_run_state import lock,atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/combat_mix_v1'

def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def checkpoint_path(arm,seed,u):
 spec=read(BASE/'protocols'/f'{arm}_s{seed}.json',{})
 namespace=spec.get('namespace','runs/combat_mix_v1/formal')
 return f'{namespace}/{arm}_s{seed}/u{u:06d}'

class Adopted:
 def __init__(self,pid,out):self.path=Path('/proc')/str(pid);self.out=out;self.returncode=None
 def poll(self):
  try:running=self.path.exists() and (self.path/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
  except OSError:running=False
  if running:return None
  self.returncode=0 if (self.out/'COMPLETE.json').exists() else 1
  return self.returncode

def existing(name):
 found=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:args=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\x00')]
  except (OSError,UnicodeError):continue
  if 'scripts.combat_mix_js_eval' in args and name in args:
   if (d/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':found.append(int(d.name))
 return found

def main():
 guard=lock(BASE/'eval/JS_OWNER.lock')
 p=read(BASE/'PROTOCOL.json');jobs={}
 for seed in p['seeds']:
  jobs[f'parent_s{seed}_u160']=p['sources'][str(seed)]['checkpoint']
  for arm in p['arms']:
   for u in (240,320):jobs[f'{arm}_s{seed}_u{u}']=checkpoint_path(arm,seed,u)
 children={};completed={};failed={};slots=['148-149','150-151']
 while len(completed)+len(failed)<len(jobs):
  # Additive authorized arms are read live; existing jobs/outputs stay immutable.
  newest=read(BASE/'PROTOCOL.json')
  for seed in newest['seeds']:
   for arm in newest['arms']:
    for u in (240,320):jobs[f'{arm}_s{seed}_u{u}']=checkpoint_path(arm,seed,u)
  for n,cp in jobs.items():
   if n in children or n in completed or n in failed:continue
   found=existing(n)
   assert len(found)<=1,'duplicate JS evaluator'
   if found:
    affinity=','.join(map(str,sorted(os.sched_getaffinity(found[0]))))
    slot=next((s for s in slots if set(range(int(s.split('-')[0]),int(s.split('-')[1])+1))==os.sched_getaffinity(found[0])),affinity)
    stream=(BASE/'eval/logs'/f'JS_{n}.adopt.log').open('ab',buffering=0)
    children[n]=(Adopted(found[0],BASE/'eval/js'/n),stream,slot)
  for n,(proc,stream,slot) in list(children.items()):
   if proc.poll() is None:continue
   rc=proc.returncode;stream.close();del children[n]
   out=BASE/'eval/js'/n
   if rc:
    failed[n]=dict(exit_code=rc,log=f'runs/combat_mix_v1/eval/logs/JS_{n}.log',unix=time.time())
    atomic_json(out/'FAILED.json',failed[n])
   else:
    done=read(out/'COMPLETE.json')
    assert done['passed'] and done['summary_sha256']==sha(out/'SUMMARY.json')
    completed[n]=done
  used=[t[2] for t in children.values()]
  for n,cp in jobs.items():
   if n in children or n in completed or n in failed:continue
   if len(children)>=len(slots):break
   if not (ROOT/cp/'COMMITTED.json').exists():continue
   # Adopt results across controller recovery without rerunning games.
   out=BASE/'eval/js'/n;done=read(out/'COMPLETE.json')
   if done:
    assert done['summary_sha256']==sha(out/'SUMMARY.json');completed[n]=done;continue
   slot=next(x for x in slots if x not in used);used.append(slot)
   stream=(BASE/'eval/logs'/f'JS_{n}.log').open('ab',buffering=0)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
   proc=subprocess.Popen(['taskset','-c',slot,sys.executable,'-m','scripts.combat_mix_js_eval',n,cp],
      cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   children[n]=(proc,stream,slot)
  atomic_json(BASE/'eval/JS_STATUS.json',dict(status='RUNNING' if children else 'WAITING_FOR_CHECKPOINTS',
    active={n:dict(cpu_affinity=t[2],checkpoint=jobs[n]) for n,t in children.items()},
    completed=list(completed),failed=failed,pending=[n for n in jobs if n not in completed and n not in children and n not in failed],
    cpu_only=True,no_training_gate=True,updated_unix=time.time()))
  time.sleep(5)
 atomic_json(BASE/'eval/JS_COMPLETE.json',dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',completed=completed,failed=failed,unix=time.time()))
if __name__=='__main__':main()
