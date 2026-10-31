"""New-rule JAX auxiliary only; independent from production JS evidence."""
import json,os,sys,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def main():
    request=json.loads(Path(sys.argv[1]).read_text());out=BASE/'eval'/request['name']/'native'
    out.mkdir(parents=True,exist_ok=True);guard=lock(out/'worker.lock');gpu_guard=lock(BASE/'GPU2_NATIVE.lock')
    if (out/'NATIVE_COMPLETE.json').exists():return
    cp=ROOT/request['checkpoint'];cm=json.loads((cp/'COMMITTED.json').read_text())
    from scripts.checkpoint_policy_artifact import verify_policy
    verify_policy(cp)
    split=request.get('split','dev');spawn=BASE/f'{split}_spawns.json';seed=json.loads(spawn.read_text())['seed']
    settings=dict(CUDA_VISIBLE_DEVICES='2',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.08',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        JAX_COMPILATION_CACHE_DIR=str(BASE/'native_gpu/cache'),JAXBOMB_RULE='bun',
        FULLMATCH_OUTPUT=str(out),FULLMATCH_SEED_START=str(seed),FULLMATCH_GAMES_PER_MODE='64')
    os.environ.update(settings)
    import jax
    jax.config.update('jax_default_matmul_precision','highest');assert jax.default_backend()=='gpu'
    from scripts import expert_reward_alignment_fullmatch as F
    os.environ.update(settings)
    from scripts.bomb_threat_runtime import install
    install();assert getattr(F.E.env.step,'_stock_window_v1',False)
    F.MODELS={'candidate':dict(policy=str((cp/'policy.pt').relative_to(ROOT)),checkpoint=request['checkpoint'],run_seed=0)}
    F.T.configure();F.provenance.install_provenance();scan,names=F.make_scan();params=F.load_params(F.MODELS['candidate'])
    started=time.time();timings=[];summary={};files={}
    manifest=dict(rule_version='stock-window-v1',checkpoint=request['checkpoint'],policy_sha256=cm['policy_sha256'],
        state_sha256=cm['state_sha256'],backend='gpu',gpu=2,seed=seed,split=split,games_per_mode=64,
        opponent='JAX rule3; not JS Hunter',teacher=False,natural=True,seat=0,
        state_hash_source='historical COMMITTED.json; state payload not needed or reverified',critic_evaluation=False,
        initial_conditions='native natural map806 unmodified bricks; native RNG birth, not frozen JS spawn cells',
        limitations=['seat0 only','JAX auxiliary cannot satisfy production JS gates','diagnostic reward traces are legacy/minimal recomputations, not bomb-threat-v1 reward'],
        source_sha256={p:sha(ROOT/p) for p in ('scripts/bomb_threat_native_eval.py','scripts/expert_reward_alignment_fullmatch.py','scripts/bomb_threat_runtime.py','jax_bomb/je07_bun_env.py')},unix=time.time())
    atomic_json(out/'MANIFEST.json',manifest)
    import numpy as np
    def progress_scan(*a,**kw):
        start=time.monotonic();result=scan(*a,**kw);jax.block_until_ready(result[0][0].t)
        timings.append(time.monotonic()-start)
        atomic_json(out/'PROGRESS.json',dict(scan_chunks=len(timings),last_scan_seconds=timings[-1],elapsed_seconds=time.time()-started,unix=time.time()))
        return result
    for mode in ('greedy','categorical'):
        rows=[]
        for offset in range(0,64,8):
            p=out/f'{mode}_{offset:03d}.json'
            if p.exists():
                v=json.loads(p.read_text());assert v['policy_sha256']==cm['policy_sha256']
                for e in v['episodes']:assert sha(ROOT/e['provenance']['trace'])==e['provenance']['trace_sha256']
                batch=v['episodes']
            else:
                batch=F.run_batch(params,'candidate',mode,list(range(offset,offset+8)),progress_scan,names)
                atomic_json(p,dict(policy_sha256=cm['policy_sha256'],rule_version='stock-window-v1',episodes=batch))
            assert [r['episode_index'] for r in batch]==list(range(offset,offset+8))
            rows+=batch;files[p.name]=sha(p)
        wins=sum(r['native_event_winner']==0 for r in rows);losses=sum(r['native_event_winner']==1 for r in rows)
        summary[mode]=dict(games=64,win=wins,draw=64-wins-losses,loss=losses,win_minus_loss=wins-losses,
            capture_episodes=sum(r['events']['capture']>0 for r in rows),death_episodes=sum(r['events']['death']>0 for r in rows),
            artificial_truncations=sum(bool(r['artificial_truncation']) for r in rows))
    atomic_json(out/'NATIVE_SUMMARY.json',dict(**manifest,natural_jax=summary,scan_seconds=timings,elapsed_seconds=time.time()-started))
    atomic_json(out/'NATIVE_COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'NATIVE_SUMMARY.json'),files=files,unix=time.time()))

if __name__=='__main__':main()
