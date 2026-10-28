"""Real production JS Hunter and native JAX complete matches, CPU only."""
import os,json,sys,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.iter_pbt_eval_resources import stage_token
from scripts.population_run_state import atomic_json,lock,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/iter_pbt_v1'
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')

def main():
 request=json.loads(Path(sys.argv[1]).read_text());name=request['name'];out=BASE/'eval'/name
 out.mkdir(parents=True,exist_ok=True);guard=lock(out/'worker.lock')
 cp=ROOT/request['checkpoint'];marker=json.loads((cp/'COMMITTED.json').read_text())
 assert sha(cp/'state.pkl')==marker['state_sha256'] and sha(cp/'policy.pt')==marker['policy_sha256']
 if (out/'COMPLETE.json').exists():
  assert json.loads((out/'COMPLETE.json').read_text())['summary_sha256']==sha(out/'SUMMARY.json');return
 split=request.get('split','dev');seed=20261101 if split=='dev' else 20261001
 shared=BASE/'eval_inputs'/split
 from scripts import je08_bun_authority as A
 A.OUT=shared;model=A.export(cp/'policy.pt',out/'model');hunter={};files={}
 def run_hunter(tier,start):
  path=out/f'hunter_{tier}_{start:03d}.json'
  if not path.exists():
   with stage_token(BASE,'JS',28):
    subprocess.run(['node','scripts/je05_web_hunter.js','--model',str(model.relative_to(ROOT)),'--difficulty',tier,'--spawns',str((shared/'spawns.json').relative_to(ROOT)),'--start',str(start),'--end',str(start+8),'--seed',str(seed),'--checkpoint-sha',marker['policy_sha256'],'--out',str(path.relative_to(ROOT))],check=True,cwd=ROOT)
  v=json.loads(path.read_text());A.validate_hunter(v,marker['policy_sha256'],tier,start)
  return tier,start,path,v
 batches={tier:{} for tier in ('hard','normal','easy')}
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(run_hunter,tier,start) for tier in batches for start in range(0,64,8)]
  for future in as_completed(futures):
   tier,start,path,v=future.result();batches[tier][start]=v['episodes'];files[path.name]=sha(path)
   atomic_json(out/'PROGRESS.json',dict(phase='production_JS_Hunter',completed_games=8*sum(len(v) for v in batches.values()),cpu_only=True,JS_concurrency_limit=28,unix=time.time()))
 for tier,parts in batches.items():
  rows=[e for start in sorted(parts) for e in parts[start]]
  hunter[tier]=A.summarize_hunter(rows)
 # JS remains CPU-only. The isolated native owner initializes GPU in a
 # separate process; no imported CPU JAX runtime is reused on a GPU.
 native=out/'native'
 atomic_json(BASE/'native_requests'/f'{name}.json',request)
 while not (native/'NATIVE_COMPLETE.json').exists():
  if (native/'NATIVE_FAILED.json').exists():raise RuntimeError('native evaluator failed; retained for diagnosis')
  atomic_json(out/'PROGRESS.json',dict(phase='waiting_native_GPU',JS_games=192,cpu_only_JS=True,native_status=json.loads((native/'NATIVE_PROGRESS.json').read_text()) if (native/'NATIVE_PROGRESS.json').exists() else None,unix=time.time()))
  time.sleep(5)
 complete=json.loads((native/'NATIVE_COMPLETE.json').read_text());assert complete['summary_sha256']==sha(native/'NATIVE_SUMMARY.json')
 nv=json.loads((native/'NATIVE_SUMMARY.json').read_text());assert nv['policy_sha256']==marker['policy_sha256']
 natural=nv['natural_jax']
 files.update({str(Path('native')/f):h for f,h in complete['files'].items()})
 from scripts.iter_pbt_recipe import score,eligible
 value=dict(checkpoint=str(cp.relative_to(ROOT)),policy_sha256=marker['policy_sha256'],hunter_js=hunter,natural_jax=natural,selection_score=score(hunter),split=split,seed=seed,teacher=False,engine_separation=True,native_backend=nv['backend'],native_episode_backend_counts=nv.get('episode_backend_counts'),native_elapsed_seconds=nv['elapsed_seconds'])
 atomic_json(out/'SUMMARY.json',value)
 atomic_json(out/'COMPLETE.json',dict(passed=True,summary_sha256=sha(out/'SUMMARY.json'),files=files,unix=time.time()))
 # Parent delivery is measured on exactly this split and never imputed.
 parent=BASE/'eval'/('parent_'+split)/'SUMMARY.json'
 if parent.exists():
  pv=json.loads(parent.read_text())
  atomic_json(out/'SELECTION.json',dict(score=value['selection_score'],eligible=eligible(natural['categorical']['capture_episodes'],pv['natural_jax']['categorical']['capture_episodes']),checkpoint=value['checkpoint'],summary_sha256=sha(out/'SUMMARY.json')))
if __name__=='__main__':main()
