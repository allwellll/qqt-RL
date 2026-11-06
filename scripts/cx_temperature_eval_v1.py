"""Frozen CPU-only temperature evaluation; existing training remains untouched."""
import concurrent.futures,json,os,subprocess,time
from pathlib import Path
from scripts import cc_strong_eval_v1 as E
from scripts.model_launch import manifest,command,public_project
from scripts.population_run_state import atomic_json,sha
OUT=E.OUT/'cx_temperature_20261106_v3'
HOLD=E.OUT/'cx_holdout_20261106_v1'
MODES={'greedy':None,'tau03':.3,'tau05':.5,'tau07':.7,'tau10':1.}
REPORT=Path('reports/cx_temperature_eval_20261106.md')
SOURCES=['scripts/native_eval_simulator_v1.js','web/review-sim/bun-native-v2-20261104.js','scripts/resume_gpu67_eval_v1.js','scripts/cx_temperature_worker_v1.js','scripts/cx_temperature_inference_v1.js','scripts/cx_idle_replay_v1.js','scripts/cx_idle_observer_adapter_v1.js','scripts/cc_strong_metrics_fountain_v1.js','web/sim-bun-native-v2.js','web/five-item-slots-model-v1.js','web/five-item-slots-v1.js']

def freeze():
 OUT.mkdir(parents=True,exist_ok=True);f=OUT/'PROTOCOL.json'
 if f.exists():
  p=json.loads(f.read_text());assert p['source_sha256']=={s:sha(s) for s in SOURCES};return p
 sp=sorted((E.OUT/'points').glob('L2_SP_u*__seats01.json'))[-1];spm=json.loads(sp.read_text())['learner'];models=[E.ensure_export(E.resolve(s)) for s in ('G_frozen','L2_S05g@u019302','L2_S05h@u020742')]+[spm]
 h=json.loads((HOLD/'PROTOCOL.json').read_text());games={}
 for seat in (0,1):
  base=[g for g in h['games'][str(seat)] if g['mode']=='greedy'];assert len(base)==88
  games[str(seat)]=[dict(g,mode=m,temperature=t,evaluation_split='cx_temperature_20261106_v1') for m,t in MODES.items() for g in base]
 p=dict(status='FROZEN_BEFORE_NEW_MATCHES',created_unix=time.time(),models=models,modes=MODES,games=games,games_per_model=880,total_games=3520,source_seed_protocol=str(HOLD/'PROTOCOL.json'),source_seed_protocol_sha256=sha(HOLD/'PROTOCOL.json'),sp_latest_point_at_freeze=str(sp),source_sha256={s:sha(s) for s in SOURCES},opponent='G_frozen greedy',temperature_heads=['move','ability'],same_action_rng_across_temperatures=True,rule='W/L/D and deliveries first, causal kills second; longest actionable stationary clearly shorter; fountain streak nondecreasing; selfdeath reference, no veto; choose model+tau; web P1 screen before publishing',cpu_only=True,web_publish_authorized=False)
 atomic_json(f,p);return p

def reuse(p):
 # Reuse only the exact already-frozen 88 seeds, masks, simulator and model artifacts.
 files=[]
 for model in p['models']:
  if model['label'] not in ('G_frozen','L2_S05h_u020742'):continue
  for seat in (0,1):
   folder=HOLD/'pairs'/f"{model['label']}__vs__G_frozen__P{seat}"
   for result in folder.glob('shard_*/match-*/RESULT.json'):
    r=json.loads(result.read_text())
    if r['mode'] not in ('greedy','categorical'):continue
    assert r['review_verified'] and r['model_json_sha256']==model['model_json_sha256']
    review=result.parent/'review.json';assert sha(review)==r['review_sha256']
    g=json.loads(review.read_text())['meta']['game'];assert g in json.loads((HOLD/'PROTOCOL.json').read_text())['games'][str(seat)]
    files.append(dict(result=str(result),review=str(review),label=model['label'],seat=seat,mode='greedy' if r['mode']=='greedy' else 'tau10',seed=r['seed']))
 assert len(files)==704
 atomic_json(OUT/'REUSE.json',dict(source_protocol_sha256=sha(HOLD/'PROTOCOL.json'),rows=files,strict_matching_reused_games=len(files)))
 return {(x['label'],x['seat'],x['mode'],x['seed']):x for x in files}

def shard(p,model,seat,mode,i,games):
 folder=OUT/'pairs'/model['label']/f'P{seat}_{mode}'/f'shard_{i:03d}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'COMPLETE.json').exists():return len(games)
 cfg=folder/'CONFIG.json';gmodel=p['models'][0]
 atomic_json(cfg,dict(candidate=model['label'],target='G_frozen',model_json=model['model_json'],opponent_model_json=gmodel['model_json'],games=games,output=str(folder),source_sha256=p['source_sha256'],protocol_sha256=sha(OUT/'PROTOCOL.json')))
 with E.Slot() as cpu:
  c=manifest(dict(script='scripts/cx_temperature_worker_v1.js',args=[str(cfg)],env=E.CPU_ENV))
  with (folder/'worker.log').open('a') as log:
   r=subprocess.run(['taskset','-c',str(cpu)]+command('evaluation',c,node=True),cwd=public_project(),env=os.environ|E.CPU_ENV,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
  assert r.returncode==0,(folder,r.returncode)
 return len(games)

def main():
 p=freeze();old=reuse(p);tasks=[]
 for mode in MODES:
  for m in p['models']:
   for seat in (0,1):
    games=[g for g in p['games'][str(seat)] if g['mode']==mode and (m['label'],seat,mode,g['seed']) not in old]
    for i,k in enumerate(range(0,len(games),4)):tasks.append((p,m,seat,mode,i,games[k:k+4]))
 atomic_json(OUT/'STATUS.json',dict(status='RUNNING',unix=time.time(),completed_reused=704,completed_new=0,total=3520,protocol_sha256=sha(OUT/'PROTOCOL.json')))
 done=0
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(E.cpus())) as pool:
   fs=[pool.submit(shard,*t) for t in tasks]
   for f in concurrent.futures.as_completed(fs):
    done+=f.result();atomic_json(OUT/'STATUS.json',dict(status='RUNNING',unix=time.time(),completed_reused=704,completed_new=done,total=3520));print('COMPLETED_NEW',done,flush=True)
  from scripts.cx_temperature_report_v1 import report
  report()
 except Exception as e:
  atomic_json(OUT/'STATUS.json',dict(status='FAILED_REUSABLE_MATCHES',unix=time.time(),completed_new=done,error=f'{type(e).__name__}: {e}'));raise

if __name__=='__main__':main()
