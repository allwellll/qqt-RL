"""CPU-only matched production-JS evaluation for COMBAT checkpoints."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/combat_mix_v1'
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')

def main():
    name=sys.argv[1];cp=ROOT/sys.argv[2];out=BASE/'eval/js'/name
    out.mkdir(parents=True,exist_ok=True);guard=lock(out/'worker.lock')
    protocol=json.loads((BASE/'js_web_compare_v1/PROTOCOL.json').read_text())
    for n,h in protocol['source_sha256'].items():assert sha(ROOT/n)==h,n
    marker=json.loads((cp/'COMMITTED.json').read_text())
    assert sha(cp/'policy.pt')==marker['policy_sha256'] and sha(cp/'state.pkl')==marker['state_sha256']
    if (out/'COMPLETE.json').exists():
        done=json.loads((out/'COMPLETE.json').read_text())
        assert done['summary_sha256']==sha(out/'SUMMARY.json')
        return
    from scripts import je08_bun_authority as A
    A.OUT=BASE/'js_web_compare_v1/shared'
    model=A.export(cp/'policy.pt',out/'model')
    summary={};shards={};began=time.time()
    def progress(**more):
        atomic_json(out/'PROGRESS.json',dict(cpu_only=True,teacher=False,checkpoint=str(cp.relative_to(ROOT)),
            policy_sha256=marker['policy_sha256'],unix=time.time(),**more))
    for tier in protocol['hunter_difficulties']:
        rows=[]
        for start in range(0,64,8):
            target=out/f'hunter_{tier}_{start:03d}.json'
            if not target.exists():
                subprocess.run(['node','scripts/je05_web_hunter.js','--model',str(model.relative_to(ROOT)),
                    '--difficulty',tier,'--spawns',str((A.OUT/'spawns.json').relative_to(ROOT)),
                    '--start',str(start),'--end',str(start+8),'--seed',str(protocol['seed']),
                    '--checkpoint-sha',marker['policy_sha256'],'--out',str(target.relative_to(ROOT))],check=True,cwd=ROOT)
            result=json.loads(target.read_text());A.validate_hunter(result,marker['policy_sha256'],tier,start)
            rows.extend(result['episodes']);shards[target.name]=sha(target)
            progress(phase='hunter',difficulty=tier,games=len(rows))
        assert len(rows)==64;summary[tier]=A.summarize_hunter(rows)
    natural={}
    for mode in ('categorical','greedy'):
        target=out/f'natural_{mode}.json'
        if not target.exists():
            env=dict(os.environ,JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='64',
                JE10_ROUTE_START='0',JE10_ROUTE_END='64',JE10_ROUTE_SEED_START=str(protocol['seed']))
            subprocess.run(['node','scripts/expert_validation_js_eval.js',mode,str(model.relative_to(ROOT)),
                str(target.relative_to(ROOT)),protocol['regions']],env=env,cwd=ROOT,check=True)
        value=json.loads(target.read_text());episodes=value['results']['natural_full2400']['episodes']
        assert value['passed'] and len(episodes)==64 and value['model_json_sha256']==sha(model)
        natural[mode]=value['results']['natural_full2400'];shards[target.name]=sha(target)
        progress(phase='natural',mode=mode,games=len(episodes))
    atomic_json(out/'SUMMARY.json',dict(checkpoint=str(cp.relative_to(ROOT)),policy_sha256=marker['policy_sha256'],
        engine='production web/sim.js',teacher=False,hunter_js=summary,natural_js=natural,
        model_json_sha256=sha(model),wall_seconds=time.time()-began))
    atomic_json(out/'COMPLETE.json',dict(passed=True,summary_sha256=sha(out/'SUMMARY.json'),
        policy_sha256=marker['policy_sha256'],shards=shards,games=320,cpu_only=True,teacher=False,unix=time.time()))

if __name__=='__main__':main()
