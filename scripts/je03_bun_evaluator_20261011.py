"""CPU-only Bun first-stage authority queue, independent of GPU progress."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.je03_bun_stage1_20261011 import RUN,SOURCE,POINTS,read,verify_checkpoint
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bun_checkpoint_state import cpu_command
from scripts.je03_admission_hunter_outcomes import episode_outcome

OUT=RUN/'authority_bun_v1'
SPAWNS=Path(os.environ.get('BUN_EVAL_SPAWNS', ROOT/'data/evaluation/bun_hunter_spawns_v1.json'))
FILES=['scripts/je03_bun_evaluator_20261011.py','scripts/je03_original_hunter_20261011.js',
       'scripts/je03_native_vit_export_20261011.py','scripts/export_bun_web_model.py',
       'scripts/eval_web_hunter.js','scripts/eval_web_hunter_residual.js','web/sim.js','web/bun_hunter_bot.js',
       'web/bot_contract.js','jax_bomb/bun_tactical_eval.py','jax_bomb/bun_rule_bot.py']


def validate_hunter(data,policy_hash,difficulty,start):
    assert data['checkpoint_sha256']==policy_hash and data['difficulty']==difficulty
    assert data['real_web_hunter'] and data['primary_steps']==300 and data['residual_limit']==64
    assert data['game_range']==[start,start+8] and len(data['episodes'])==8
    manifest=read(OUT/'spawns.json')
    assert data['seed']==manifest['seed']
    for i,e in enumerate(data['episodes'],start):
        assert e['game']==i and e['spawn_cells']==manifest['spawn_cells'][i]
        assert e['primary_ticks']==300 and not e['residual']['hazards_remaining']


def summarize_hunter(rows):
    outcomes=[episode_outcome(e) for e in rows]
    metrics={k:sum(e[k] for e in outcomes) for k in outcomes[0]}
    metrics.update(games=len(rows),self_episodes=sum(bool(e['counts']['own_bomb_defeat']) for e in rows),
        trade_episodes=sum(bool(e['counts']['mutual_death']) for e in rows),
        by_bot_episodes=sum(bool(e['counts']['killed_by_bot']) for e in rows),
        bombs_per_game=sum(e['counts']['bombs'] for e in rows)/len(rows))
    return metrics


def export(policy,out):
    out.mkdir(parents=True,exist_ok=True);target=out/'model.json';onnx=out/'model.onnx'
    policy_hash=sha(policy)
    if not target.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/export_bun_web_model.py'),str(policy),'--output',str(target.with_suffix('.unsealed.json')),'--name',out.name],check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/je03_native_vit_export_20261011.py'),str(policy),str(onnx)],check=True)
        document=read(target.with_suffix('.unsealed.json'))
        assert document['meta']['rule']=='bun' and document['meta']['obs_shape']==[24,13,15]
        document['meta'].update(onnx_path=str(onnx),onnx_sha256=sha(onnx),checkpoint_sha256=policy_hash)
        assert 'input_encoding' not in document['meta'],'native Bun only; no non-Bun input adapter'
        atomic_json(target,document)
    assert read(target)['meta']['checkpoint_sha256']==policy_hash and sha(onnx)==read(target)['meta']['onnx_sha256']
    return target


def tactical(policy,target):
    from jax_bomb.bun_tactical_eval import TacticalOpponentEvaluator
    from scripts.eval_four_bot_matrix import tactical_episodes
    import jax
    assert jax.default_backend()=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']==''
    evaluator=TacticalOpponentEvaluator(host_workers=8)
    try:
        data=evaluator.run(str(policy),64,read(OUT/'spawns.json')['seed'],300,per_episode=True,
                           spawn_cells=read(OUT/'spawns.json')['spawn_cells'])
        episodes=tactical_episodes(data)
        data['settled_style_summary']=dict(games=64,safe_win=sum(e['win'] for e in episodes),
            death_episode=sum(e['killed_by_bot'] or e['own_bomb'] or e['mutual'] for e in episodes),
            self_episodes=sum(e['own_bomb'] for e in episodes),trade_episodes=sum(e['mutual'] for e in episodes),
            bombs_per_game=sum(e['bombs'] for e in episodes)/64,
            limitation='Tactical300 native JAX protocol, no web residual; report separately from Hunter')
        atomic_json(target,data)
    finally:evaluator.close()


def evaluate(label,policy,proof):
    out=OUT/label;out.mkdir(parents=True,exist_ok=True)
    complete=out/'COMPLETE.json';policy_hash=sha(policy)
    if complete.exists():
        previous=read(complete);assert previous['policy_sha256']==policy_hash
        assert all(sha(out/k)==h for k,h in previous['shards'].items());return previous
    start=time.time();model=export(policy,out/'model')
    atomic_json(out/'PREDECLARED.json',dict(policy=str(policy),policy_sha256=policy_hash,checkpoint_proof=proof,
        protocol_sha256=sha(OUT/'PROTOCOL.json'),cuda_visible_devices='',unix=time.time()))
    def hunter_cell(job):
        difficulty,s=job;target=out/f'hunter_{difficulty}_{s:03d}.json'
        if not target.exists():
            subprocess.run(['node',str(ROOT/'scripts/je03_original_hunter_20261011.js'),'--model',str(model),
                '--difficulty',difficulty,'--spawns',str(OUT/'spawns.json'),'--start',str(s),'--end',str(s+8),
                '--seed',str(read(OUT/'spawns.json')['seed']),'--checkpoint-sha',policy_hash,'--out',str(target)],check=True)
        validate_hunter(read(target),policy_hash,difficulty,s)
        return target.name,sha(target)
    target=out/'tactical_v2.json'
    cmd=[sys.executable,str(Path(__file__).resolve()),'tactical','--policy',str(policy),'--out',str(target)]
    child=None;log=None
    if not target.exists():
        log=(out/'tactical.log').open('a')
        child=subprocess.Popen(cpu_command(cmd, 'BUN_TACTICAL_CPUS'),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        shards=dict(pool.map(hunter_cell,[(d,s) for d in ('hard','normal','easy') for s in range(0,64,8)]))
    hunter_wall=time.time()-start
    metrics={d:summarize_hunter([e for s in range(0,64,8) for e in read(out/f'hunter_{d}_{s:03d}.json')['episodes']]) for d in ('hard','normal','easy')}
    atomic_json(out/'HUNTER_COMPLETE.json',dict(passed=True,metrics=metrics,shards=shards,wall_seconds=hunter_wall,policy_sha256=policy_hash,unix=time.time()))
    if child:
        assert child.wait()==0,'Tactical failure preserved; CPU owner retry cannot affect training'
        log.close()
    td=read(target)
    assert td['checkpoint_sha256']==policy_hash and td['summary']['games']==64
    assert td['runtime']['backend']=='cpu' and td['per_episode']['spawn_cells']==read(OUT/'spawns.json')['spawn_cells']
    shards['tactical_v2.json']=sha(target)
    document=dict(passed=True,label=label,policy_sha256=policy_hash,checkpoint_proof=proof,protocol_sha256=sha(OUT/'PROTOCOL.json'),
        hunter=metrics,tactical=td['settled_style_summary'],shards=shards,hunter_wall_seconds=hunter_wall,
        wall_seconds=time.time()-start,cuda_visible_devices='',stage2_authorized=False,unix=time.time())
    atomic_json(complete,document);return document


def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(namespace='authority_bun_v1',reference='original Bun phase2_it4000; no non-Bun model',
        paired=True,hunter_games_per_tier_per_arm=64,tactical_games_per_arm=64,hunter_primary=300,hunter_residual=64,
        tactical_primary=300,spawns_sha256=sha(SPAWNS),seed=read(SPAWNS)['seed'],checkpoints=POINTS,
        safety='Hunter safe win requires credited kill, zero actor deaths and alive after residual',
        cuda_visible_devices='',sources={p:sha(ROOT/p) for p in FILES},stage2_enabled=False)
    if (OUT/'PROTOCOL.json').exists():assert read(OUT/'PROTOCOL.json')==protocol
    else:atomic_json(OUT/'PROTOCOL.json',protocol)
    if (OUT/'spawns.json').exists():assert read(OUT/'spawns.json')==read(SPAWNS)
    else:atomic_json(OUT/'spawns.json',read(SPAWNS))


def main():
    setup();own=lock(OUT/'owner.lock')
    atomic_json(OUT/'OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time()))
    errors={}
    while True:
        jobs=[('original_bun_it4000',SOURCE,dict(reference_sha256=sha(SOURCE)))]
        for g in POINTS:
            for arm in ('control','treatment'):
                cp=RUN/'formal'/arm/f'gen{g:06d}'
                if (cp/'COMMITTED.json').exists():
                    m=verify_checkpoint(cp,sha(RUN/'plan.json'))
                    jobs.append((f'{arm}_g{g:06d}',cp/'policy.pt',dict(committed_sha256=sha(cp/'COMMITTED.json'),generation=g,policy_sha256=m['policy_sha256'])))
        completed=[]
        for label,policy,proof in jobs:
            try:evaluate(label,policy,proof);completed.append(label)
            except Exception as e:
                errors[label]=errors.get(label,0)+1
                atomic_json(OUT/f'TECHNICAL_STOP_{label}_attempt{errors[label]:03d}.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()))
                if errors[label]>=3:raise
        atomic_json(OUT/'STATE.json',dict(pid=os.getpid(),completed=completed,errors=errors,phase='CPU_ASYNC',unix=time.time()))
        for pair in sorted((RUN/'blocks/screen').glob('CHECKPOINT_PAIR_g*.json')):
            g=read(pair)['generation'];labels=[f'{a}_g{g:06d}' for a in ('control','treatment')]
            target=OUT/f'PAIR_g{g:06d}.json'
            if target.exists() or not all((OUT/l/'COMPLETE.json').exists() for l in labels):continue
            c,t=[read(OUT/l/'COMPLETE.json') for l in labels]
            delta={d:{k:t['hunter'][d][k]-c['hunter'][d][k] for k in c['hunter'][d]} for d in ('hard','normal','easy')}
            atomic_json(target,dict(passed=True,generation=g,pair_sha256=sha(pair),control_sha256=sha(OUT/labels[0]/'COMPLETE.json'),
                treatment_sha256=sha(OUT/labels[1]/'COMPLETE.json'),hunter_delta=delta,tactical_reported_separately=True,stage2_authorized=False,unix=time.time()))
        if (RUN/'NO_MORE_CHECKPOINTS.json').exists() and len(completed)==1+2*len(POINTS):
            atomic_json(OUT/'EVALUATION_COMPLETE.json',dict(passed=True,completed=completed,unix=time.time()))
            atomic_json(RUN/'AB_COMPLETE.json',dict(passed=True,technical_only=True,stage1_passed=False,stage2_authorized=False,
                training_sha256=sha(RUN/'TRAINING_COMPLETE.json'),evaluation_sha256=sha(OUT/'EVALUATION_COMPLETE.json'),unix=time.time()));return
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['owner','tactical']);parser.add_argument('--policy');parser.add_argument('--out');args=parser.parse_args()
    if args.command=='tactical':tactical(Path(args.policy),Path(args.out))
    else:main()
