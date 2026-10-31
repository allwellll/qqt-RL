"""Bounded, resumable five-model CPU Review recording; never controls training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.population_run_state import atomic_json,lock,sha

B=Path('runs/new_model_replays_20261031')
def read(p):return json.loads(Path(p).read_text())

def export_models(cfg):
    models={}
    existing=read('runs/v2_ownside_selection_20261031/MODELS.json')
    for label,source in cfg['models'].items():
        assert sha(Path(source['checkpoint'])/'policy.pt')==source['policy_sha256']
        m=next((v for v in existing.values() if v['policy_sha256']==source['policy_sha256']),None)
        if m is None:
            dest=B/'exports'/label;dest.mkdir(parents=True,exist_ok=True)
            with (dest/'EXPORT.log').open('ab') as log:
                subprocess.run([sys.executable,'-m','scripts.r3_items_export',source['checkpoint'],str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
            f=dest/'model.json';meta=read(f)['meta'];m=dict(model_json=str(f),model_sha256=sha(f),onnx_path=meta['onnx_path'],onnx_sha256=meta['onnx_sha256'])
        model=dict(source,**{k:m[k] for k in ['model_json','model_sha256','onnx_path','onnx_sha256']})
        assert sha(Path(model['model_json']))==model['model_sha256']
        assert sha(Path(model['onnx_path']))==model['onnx_sha256']
        models[label]=model
    atomic_json(B/'MODELS.json',models)
    return models

def selected(games):
    result=[dict(g,rule='固定hard清单前8局，不按结果挑选') for g in games[:8]]
    missing=[]
    for seat in (0,1):
        for outcome in ('loss','draw'):
            matches=[g for g in games if g['seat']==seat and
                     ((g['expected']['winner'] in (None,-1)) if outcome=='draw' else (g['expected']['winner'] not in (None,-1,seat)))]
            if matches:result.append(dict(matches[0],rule=f'披露补选：P{seat}按seed最早1'+('负' if outcome=='loss' else '平')))
            else:missing.append(dict(seat=seat,outcome=outcome))
    return result,missing

def record(label,core):
    games=[]
    for start in range(0,256,8):
        folder=B/'scan'/label/f'shard_{start:03d}';proof=folder/'ACCEPTED.json'
        if not proof.exists():
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.log').open('ab') as log:
                subprocess.run(['taskset','-c',str(core),'node','scripts/new_model_replay_record_20261031.js',label,str(start),str(start+8)],stdout=log,stderr=subprocess.STDOUT,check=True)
        d=read(proof);assert d['status']=='PASS'
        assert all(sha(Path(f))==h for f,h in d['files'].items())
        games+=d['games'];chosen,missing=selected(games)
        atomic_json(B/'scan'/label/'STATUS.json',dict(status='SCANNING' if missing else 'COMPLETE',scanned=len(games),missing=missing,unix=time.time()))
        if not missing:break
    out=dict(status='COMPLETE',label=label,scanned=len(games),selected=chosen,missing=missing,scan_files=[str(B/'scan'/label/f'shard_{start:03d}'/'ACCEPTED.json') for start in range(0,len(games),8)])
    atomic_json(B/'scan'/label/'COMPLETE.json',out)
    return out

def main():
    guard=lock(B/'OWNER.lock');cfg=read(B/'PROTOCOL.json');models=export_models(cfg)
    atomic_json(B/'EXECUTION_STATUS.json',dict(status='RECORDING_CPU',models=models,workers=cfg['workers'],unix=time.time()))
    results={}
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        todo={pool.submit(record,label,core):label for label,core in zip(models,cfg['cpu_cores'])}
        for f in as_completed(todo):
            results[todo[f]]=f.result()
            atomic_json(B/'EXECUTION_STATUS.json',dict(status='RECORDING_CPU',completed_models=list(results),unix=time.time()))
    games=[g for label in models for g in results[label]['selected']]
    manifest=dict(status='FULL_REVIEW_VERIFIED',model=next(iter(models.values())),games=games,
        model_label='新模型五组对比 · R3终点 / 双seed SP / 网页u3040',html='20261031-new-r3-sp-models-review.html',
        spawn_manifest=cfg['conditions']['hunter_hard']['spawn_manifest'],spawn_side_version=cfg['spawn_side_version'],
        rule_version='bun-native-v2',teacher=False,models=models,selection=results,
        limitations=cfg['limitations'],protocol_sha256=sha(B/'PROTOCOL.json'))
    atomic_json(B/'MANIFEST.json',manifest)
    atomic_json(B/'EXECUTION_STATUS.json',dict(status='RECORDINGS_COMPLETE_BUILD_NEXT',entries=len(games),independent_games=len({g['index'] for g in games}),unix=time.time()))
    print(json.dumps({'status':'RECORDINGS_COMPLETE','entries':len(games),'scanned':{k:v['scanned'] for k,v in results.items()},'missing':{k:v['missing'] for k,v in results.items()}}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(B/'BLOCKED.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e),unix=time.time()))
        raise
