"""Durable CPU-only 600/900tick native route/full probes; never gate GPU."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,time,concurrent.futures,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je06_bun_recipe import BASE,ROOT,PREVIOUS,read,sha,lock,atomic_json,notify,child_env
from scripts.je06_bun_candidate_audit import CANDIDATES,source
from scripts.je05_bun_authority import export

def main():
    out=BASE/'side_diagnostics_600_900_v1';out.mkdir(exist_ok=True);lease=lock(out/'owner.lock')
    atomic_json(out/'OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',unix=time.time()))
    jobs=[(label,g,task,horizon) for label,g in CANDIDATES for horizon in [600,900] for task in ['route_break','full']]
    def run(job):
        label,g,task,horizon=job;_,cp,_=source(label,g);folder=out/f'{label}_g{g:06d}'
        # Existing CPU audit exports are immutable, byte/hash checked.
        model=(BASE/'parent_candidates'/f'{label}_g{g:06d}'/'model/model.json') if label=='je04' else PREVIOUS/'authority_je05_v3'/f'bun392_combat_transport_pfsp8_g{g:06d}'/'model/model.json'
        assert read(model)['meta']['checkpoint_sha256']==sha(cp/'policy.pt')
        folder.mkdir(exist_ok=True);target=folder/f'{task}_{horizon}.json'
        if not target.exists():
            with (folder/f'{task}_{horizon}.log').open('a') as log:
                subprocess.run(['node','scripts/je06_web_tasks.js',str(model),str(target),task,str(PREVIOUS/'regions.json'),str(horizon)],cwd=ROOT,env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,check=True)
        v=read(target);assert v['passed'] and v['primary_ticks']==horizon
        return str(target.relative_to(out)),sha(target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:hashes=dict(pool.map(run,jobs))
    atomic_json(out/'EVALUATION_COMPLETE.json',dict(passed=True,hashes=hashes,screening_only=True,not_gpu_gate=True,unix=time.time()))
    atomic_json(out/'OWNER_FINISHED.json',dict(passed=True,unix=time.time()))
    notify('side_diagnostics_complete','JE06父候选600/900tick原生full/route旁路诊断完成，不阻塞GPU训练。')

if __name__=='__main__':main()
