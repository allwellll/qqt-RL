"""Consume R4 full-checkpoint pressure requests; retain the frozen bot v1."""
import json
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,lock,sha
from scripts import r4_pressure_eval_owner as evaluator
from scripts.r4_candidate_eval_owner import OUT as EVAL,model,requests

BASE=Path('runs/r4_combat_20261031');OUT=BASE/'candidate_pressure'


def read(p):return json.loads(Path(p).read_text())


def main():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'OWNER.lock')
    original=read(BASE/'interceptor/PRESSURE_PROTOCOL.json')
    f=OUT/'PROTOCOL.json'
    if not f.exists():
        atomic_json(f,dict(status='FROZEN',id='R4-CANDIDATE-PRESSURE-V1',parent_matrix=str(BASE/'interceptor/PRESSURE_PROTOCOL.json'),
            parent_protocol_sha256=sha(BASE/'interceptor/PRESSURE_PROTOCOL.json'),
            candidate='committed absolute400 checkpoint and exact bounded endpoint; both K1/K3, no peak',
            games_per_condition=256,conditions=original['conditions'],seed=original['seed'],modes=['greedy'],
            parent='u4002 existing256 shards, never rerun',teacher=False,cpu_only=True,cores=[164,165,166,167],
            gate='P1 interceptor wins-minus-loss >=parent-5; no publication from this owner',unix=time.time()))
    evaluator.OUT=OUT
    while True:
        rr=[r for r in requests() if r['kind']=='full' and not r['key'].startswith('parent')]
        for req in rr:
            proof=OUT/req['key']/'COMPLETE.json'
            if proof.exists():continue
            identity=model(req);p=dict(original,models={req['key']:identity})
            missing=[(c,i) for c in p['conditions'] for i in range(0,256,8)
                if not Path(str(OUT/'evaluation'/req['key']/c/f'shard_{i:03d}.json')+'.ACCEPTED.json').exists()]
            for offset in range(0,len(missing),4):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures=[pool.submit(evaluator.collect,req['key'],c,i,p,164+k) for k,(c,i) in enumerate(missing[offset:offset+4])]
                    for fut in futures:fut.result()
                atomic_json(OUT/'STATUS.json',dict(status='CPU_ASYNC',request=req['key'],remaining_shards=max(0,len(missing)-offset-4),unix=time.time()))
            results={c:evaluator.finish(req['key'],c) for c in p['conditions']}
            atomic_json(proof,dict(status='COMPLETE',source=req['source'],results=results,protocol_sha256=sha(f),unix=time.time()))
            subprocess.run(['jdme-notify','send','R4完整运包压力集完成 '+req['key']+'，拦截/协作各256原生JS局；P0/P1分母与源SHA已保存。'],capture_output=True)
        atomic_json(OUT/'STATUS.json',dict(status='WAIT_COMMITTED_FULL_CHECKPOINT' if not rr else 'REQUESTS_CONSUMED',requests=len(rr),unix=time.time()))
        if all((BASE/'formal'/('r4_'+a+'_c000000')/'COMPLETE.json').exists() for a in ('k1','k3')) and all((OUT/r['key']/'COMPLETE.json').exists() for r in rr):
            atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',requests=len(rr),unix=time.time()));return
        time.sleep(10)


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'TECHNICAL_BLOCKED.json',dict(status='BLOCKED',error=type(error).__name__,reason=str(error),unix=time.time()));raise
