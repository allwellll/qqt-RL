"""CPU-only fixed-opponent evaluation of the new model and replay export."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
FILES=('scripts/export_dcu_kill_replays.py','scripts/dcu_kill_replay_bridge.js',
       'scripts/kill_mode_bridge.js','web/sim-bun-native-v4-facing.js',
       'web/kill_opening_items.js','web/self_respawn_preview.js','web/r3_model_inputs.js',
       'web/assets/maps/levels.json','scripts/r4_k4_observation.py',
       'scripts/r3_items_observation.py','scripts/retention_phase_conditioning.py','jax_bomb/jax_net.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def statistics(replays):
    groups=defaultdict(list)
    for replay in replays:groups[(replay['spec']['version'],replay['spec']['opponent'])].append(replay)
    rows=[]
    for (version,opponent),games in sorted(groups.items()):
        row=dict(version=version,opponent=opponent,games=len(games),wins=0,draws=0,losses=0,kills=0,deaths=0,self_deaths=0)
        for game in games:
            p=game['spec']['seat'];summary=game['summary']
            if not summary['full_game'] or summary['ticks']!=1800:raise ValueError('Incomplete game')
            winner=summary['winner'];row['draws' if winner is None else 'wins' if winner==p else 'losses']+=1
            for key,k in (('kills',0),('deaths',1),('self_deaths',2)):row[key]+=summary['metrics'][p][k]
        row.update(win_rate=row['wins']/len(games),score_rate=(row['wins']+.5*row['draws'])/len(games))
        rows.append(row)
    return rows

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/kill-jax-8card-e512-20261115')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',default='2026111501,2026111502')
    parser.add_argument('--opponents',default='DUEL_latest,SP_latest,G_frozen')
    parser.add_argument('--temperature',type=float,default=.3)
    parser.add_argument('--cpu-cores',type=int,default=8)
    parser.add_argument('--max-seconds',type=int,default=1800)
    a=parser.parse_args();os.chdir(ROOT)
    seeds=[int(x) for x in a.seeds.split(',')];opponents=a.opponents.split(',')
    if a.out.exists() or not 0<a.temperature<=2 or not 1<=a.cpu_cores<=16:parser.error('New output and valid resource/temperature required')
    if not seeds or len(set(seeds))!=len(seeds) or any(not 0<=s<=0xffffffff for s in seeds):parser.error('Distinct uint32 seeds required')
    if not opponents or len(set(opponents))!=len(opponents) or set(opponents)-{'DUEL_latest','SP_latest','G_frozen'}:parser.error('Invalid fixed opponents')
    # Change only this evaluation process and its children; never touch DCUs.
    os.environ.update(JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',
                      JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_ENABLE_COMPILATION_CACHE='false')
    os.environ.pop('XLA_FLAGS',None)
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-a.cpu_cores:])
    import numpy as np
    import jax
    import jax.numpy as j
    from scripts import portable_training as portable
    from scripts.r4_k4_observation import forward
    from scripts.train_dcu_kill import Bridge
    from scripts.dcu_kill_replay_html import bundle
    assert jax.default_backend()=='cpu' and all(d.platform=='cpu' for d in jax.devices())
    started=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    checkpoint=(a.checkpoint or a.run/json.loads((a.run/'LATEST.json').read_text())['checkpoint']).resolve()
    meta=json.loads((checkpoint/'CHECKPOINT.json').read_text())
    for file,expected in meta['source_sha256'].items():
        if sha(ROOT/file)!=expected:raise ValueError('Checkpoint source mismatch: '+file)
    target=a.out/'candidate_actor.pt';shutil.copyfile(checkpoint/'actor.pt',target)
    if sha(target)!=meta['actor_sha256']:raise ValueError('Candidate SHA256 mismatch')
    (a.out/'SOURCE_CHECKPOINT.json').write_text(json.dumps(meta,indent=2)+'\n')
    candidate=pickle.loads(target.read_bytes())  # trusted local producer, hash checked
    portable.verify_bundle();protocol=portable.read(portable.BUNDLE/'protocols/DUEL.json')
    models={name:portable.load_actor(protocol,name) for name in set(opponents)|{'DUEL_latest'}}
    baseline=models['DUEL_latest']
    if jax.tree.structure(candidate)!=jax.tree.structure(baseline):raise ValueError('Model tree mismatch')
    for x,y in zip(jax.tree.leaves(candidate),jax.tree.leaves(baseline)):
        if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all():raise ValueError('Candidate schema/finite mismatch')
    models['candidate']=candidate;models=jax.tree.map(j.asarray,models)
    specs=[dict(version='current',opponent=o,seed=s,seat=p) for o in opponents for s in seeds for p in (0,1)]
    model_hashes={name:protocol['models'][name]['sha256'] for name in models if name!='candidate'}
    model_hashes['candidate']=meta['actor_sha256']
    declaration=dict(schema='qqt.kill-fixed-eval/v1',simulator='live web bun-native-v4-facing',
        recorded_during_training=False,evaluation_replay=True,game_ticks=1800,temperature=a.temperature,seeds=seeds,opponents=opponents,
        balanced_seats=True,versions=['current'],cpu_only=True,cpu_affinity=sorted(os.sched_getaffinity(0)),
        model_sha256=model_hashes,source_sha256={file:sha(ROOT/file) for file in FILES},specs=specs)
    (a.out/'PREDECLARED.json').write_text(json.dumps(declaration,indent=2)+'\n')
    class RecordingBridge(Bridge):
        def __init__(self):
            self.log=(a.out/'simulator.log').open('w');self.buffer=b''
            self.process=subprocess.Popen([str(ROOT/'.tools/node/bin/node'),str(ROOT/'scripts/dcu_kill_replay_bridge.js')],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
            try:self.initial=self.call(dict(command='init',specs=specs))
            except BaseException:self.close();raise
    actor_groups=defaultdict(list)
    for i,spec in enumerate(specs):
        p=spec['seat'];actor_groups['candidate' if spec['version']=='current' else 'DUEL_latest'].append((i,p))
        actor_groups[spec['opponent']].append((i,1-p))
    @jax.jit
    def inference(actor,obs,state):
        move,ability,_,_=forward(actor,obs,state)
        return move,ability
    randoms=[[np.random.default_rng(np.random.SeedSequence([spec['seed'],p,0xA17])) for p in (0,1)] for spec in specs]
    def sample(logits,mask,uniform):
        if not np.all(mask.any(-1)):raise ValueError('Empty action mask')
        z=np.where(mask,logits/a.temperature,-1e30).astype(np.float64);z-=z.max(-1,keepdims=True)
        probability=np.exp(z);probability/=probability.sum(-1,keepdims=True)
        return np.minimum((probability.cumsum(-1)<uniform[:,None]).sum(-1),logits.shape[-1]-1)
    print(json.dumps(dict(event='cpu_eval_ready',games=len(specs),checkpoint=str(checkpoint),cpu_affinity=sorted(os.sched_getaffinity(0)))),flush=True)
    bridge=RecordingBridge()
    try:
        row=bridge.initial
        for tick in range(1800):
            if time.monotonic()-started>a.max_seconds:raise TimeoutError('CPU evaluation wall budget exceeded')
            action=np.zeros((len(specs),2,2),np.int32)
            for name,indices in actor_groups.items():
                games,seats=np.asarray(indices).T
                mv,ab=inference(models[name],row['obs'][games,seats],row['state'][games,seats]);mv,ab=np.asarray(mv),np.asarray(ab)
                if not np.isfinite(mv).all() or not np.isfinite(ab).all():raise ValueError('Nonfinite inference')
                uniforms=np.array([randoms[g][p].random(2) for g,p in indices])
                action[games,seats,0]=sample(mv,row['mm'][games,seats].astype(bool),uniforms[:,0])
                action[games,seats,1]=sample(ab,row['bm'][games,seats].astype(bool),uniforms[:,1])
            row=bridge.call(dict(command='step',actions=action.tolist()))
            if tick%180==179:print(json.dumps(dict(event='cpu_eval_progress',tick=tick+1,elapsed_seconds=time.monotonic()-started)),flush=True)
        if not np.all(row['done']):raise ValueError('Incomplete evaluation')
        records=bridge.call(dict(command='finish'))['replays']
        # Preserve completed games before optional presentation/export work.
        (a.out/'COMPLETED_GAMES.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
    finally:bridge.close()
    document=dict(schema='qqt.kill-checkpoint-replays/v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        candidate=dict(update=checkpoint.name,actor_sha256=meta['actor_sha256'],source_checkpoint=str(checkpoint.relative_to(ROOT)),training_environment_steps=meta['global_environment_steps']),
        protocol={k:v for k,v in declaration.items() if k not in ('model_sha256','source_sha256','specs')},
        source_sha256=declaration['source_sha256'],model_sha256=model_hashes,replays=records,statistics=statistics(records),
        limitation='Small fixed-seed pilot; not a recording of live JAX training trajectories. Randomness and simulator implementations differ. No claim of statistically established improvement.')
    path=a.out/'replays.json';path.write_text(json.dumps(document,ensure_ascii=False,separators=(',',':'))+'\n')
    bundle(document,a.out/'watch.html')
    report=dict(status='PASS_CPU_FULL_GAME_EXPORT',games=len(records),ticks_per_game=1800,cpu_only=True,
        elapsed_seconds=time.monotonic()-started,statistics=document['statistics'],
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (path,a.out/'watch.html')})
    (a.out/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
