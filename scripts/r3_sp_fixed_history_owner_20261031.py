"""Bounded CPU policy-vs-policy evaluation, separate from GPU admission."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts import r3_selfplay_eval_20261031 as bot

OUT=bot.OUT
MODULE='scripts.r3_sp_fixed_history_owner_20261031'


def cpu_cores():
    return (150,151) if OUT.name=='seed02' else (146,147)


def summary(episodes,seat):
    rows=[e for e in episodes if e['seat']==seat];n=len(rows)
    assert n
    wins=sum(e['winner']==seat for e in rows)
    draws=sum(e['winner'] in (-1,None) for e in rows);losses=n-wins-draws
    score=(wins+.5*draws)/n
    elo=400*math.log10(score/(1-score)) if 0<score<1 else None
    kills=sum(e['causal_kills'][seat] for e in rows)
    bombs=[e['placed'][seat] for e in rows if 'placed' in e]
    contacts=[e['r3']['contacts'] for e in rows if 'contacts' in e.get('r3',{})]
    own_glue=sum(c['type']==2 and c['player']==seat and c['owner']==seat for cs in contacts for c in cs)
    draw_rows=[e for e in rows if e['winner'] in (-1,None)]
    draw_delta=sum(e['causal_kills'][seat]-e['causal_kills'][1-seat] for e in draw_rows)
    return dict(games=n,wins=wins,draws=draws,losses=losses,score=score,elo_difference=elo,
                win_rate=wins/n,placed_bombs=sum(bombs) if len(bombs)==n else None,
                placed_bombs_per_game=sum(bombs)/n if len(bombs)==n else None,
                own_glue_contacts=own_glue if len(contacts)==n else None,
                own_glue_contacts_per_game=own_glue/n if len(contacts)==n else None,
                metric_coverage=dict(placed_games=len(bombs),contact_games=len(contacts),games=n),
                draw_causal_kill_difference=draw_delta,
                draw_causal_kill_difference_per_draw=draw_delta/len(draw_rows) if draw_rows else None,
                own_glue_scope='observed type2 owner==player contacts; not attributed causal harm',
                elo_boundary='unbounded; not clipped' if elo is None else None,
                causal_kills=kills,causal_kills_per_game=kills/n,
                deaths=sum(e['deaths'][seat] for e in rows),
                scope='empirical fixed-opponent score transform, not fitted league Elo')


def export(source,key):
    folder=OUT/'fixed_evaluation/models'/key;folder.mkdir(parents=True,exist_ok=True)
    identity=folder/'IDENTITY.json'
    if identity.exists():
        r=bot.read(identity)
        assert r['policy_sha256']==source['policy_sha256'] and sha(Path(r['model_json']))==r['model_sha256']
        doc=bot.read(r['model_json']);assert sha(Path(doc['meta']['onnx_path']))==doc['meta']['onnx_sha256']
        return r
    cp=Path(source['checkpoint']);assert sha(cp/'policy.pt')==source['policy_sha256']
    code=('from pathlib import Path;from scripts.je08_bun_authority import export;import sys;export(Path(sys.argv[1])/"policy.pt",Path(sys.argv[2]).resolve())'
          if source['update']==2760 else 'from scripts.r3_items_export import export;import sys;export(*sys.argv[1:])')
    r=subprocess.run(['taskset','-c',','.join(map(str,cpu_cores())),sys.executable,'-c',code,str(cp),str(folder)],text=True,capture_output=True)
    (folder/'EXPORT.log').write_text((r.stdout+r.stderr).replace(str(Path.cwd()),'<WORKTREE>'))
    assert r.returncode==0,'fixed history CPU export failed; see EXPORT.log'
    f=folder/'model.json';assert bot.read(f)['meta']['checkpoint_sha256']==source['policy_sha256']
    result=dict(model_json=str(f),model_sha256=sha(f),policy_sha256=source['policy_sha256'],source=source)
    atomic_json(identity,result);return result


def requests():
    result=[];reg=bot.read(OUT/'REGISTRATION.json')
    result.append(dict(key='parent',source=reg['parent_source']))
    for f in (OUT/'formal').glob('r3_sp*/FIRST_FORMAL_UPDATE.json'):
        result.append(dict(key=f.parent.name+'_u003043',source=bot.read(f)['source']))
    for r in bot.requests():
        if r['key']=='parent':continue
        if r['source']['update']%40==0 or r['key'].endswith('_final'):result.append(r)
    seen=set();unique=[]
    for r in result:
        identity=(r['source']['checkpoint'],r['source']['policy_sha256'])
        if identity in seen:continue
        seen.add(identity);unique.append(r)
    return unique


def recompute_completed_metrics():
    """Add fields from existing immutable games without rerunning any episode."""
    for file in (OUT/'fixed_evaluation').glob('*/COMPLETE.json'):
        target=file.parent/'METRICS_1820.json'
        if target.exists():continue
        doc=bot.read(file);metrics={}
        for opponent in doc['metrics']:
            rows=[]
            for start in range(0,128,8):
                side=file.parent/opponent/f'shard_{start:03d}.json.outcome.json'
                assert sha(side)==doc['files'][str(side)]
                rows.extend(bot.read(side)['episodes'])
            assert len(rows)==128==len({(e['seed'],e['seat']) for e in rows})
            metrics[opponent]={f'P{s}':summary(rows,s) for s in (0,1)}
        atomic_json(target,dict(status='COMPLETE_RECOMPUTED',source_complete=str(file),
            source_complete_sha256=sha(file),metrics=metrics,new_games=0,unix=time.time()))


def collect(req,opponent_key,opponent,model,start,protocol,core):
    f=OUT/'fixed_evaluation'/req['key']/opponent_key/f'shard_{start:03d}.json'
    f.parent.mkdir(parents=True,exist_ok=True);guard=lock(Path(str(f)+'.lock'))
    accepted=Path(str(f)+'.ACCEPTED.json');side=Path(str(f)+'.outcome.json')
    if accepted.exists():
        r=bot.read(accepted);assert all(sha(Path(k))==v for k,v in r['files'].items());return
    birth=protocol['holdout'];assert sha(Path(birth['spawn_manifest']))==birth['spawn_sha256']
    from scripts.bun_native_v2_owner import processes
    # A CPU owner reload never cancels live eight-game shards. Adopt them
    # before inspecting outputs or launching any replacement.
    while list(processes('scripts/r3_sp_fixed_history_eval_20261031.js',str(f))):time.sleep(3)
    if not (f.exists() and side.exists()):
        env=dict(os.environ,QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',BUN_V2_TIER='hard',
                 JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='128',JE10_ROUTE_START=str(start),
                 JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(birth['seed']),
                 R3_SP_FIXED_OPPONENT=opponent['model_json'])
        r=subprocess.run(['taskset','-c',str(core),'node','scripts/r3_sp_fixed_history_eval_20261031.js',
                          'greedy',model['model_json'],str(f),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',birth['spawn_manifest']],
                         env=env,text=True,capture_output=True)
        f.with_suffix('.log').write_text((r.stdout+r.stderr).replace(str(Path.cwd()),'<WORKTREE>'))
        assert r.returncode==0,'fixed policy full-game shard failed; inspect shard log'
    d=bot.read(side);episodes=d['episodes']
    assert d['engine']=='production_JS' and d['mode']=='greedy' and d['rule_version']=='bun-native-v2' and not d['teacher']
    assert bot.read(f)['model_json_sha256']==model['model_sha256']
    assert len(episodes)==8 and [e['seed'] for e in episodes]==list(range(birth['seed']+start,birth['seed']+start+8))
    from scripts.bun_spawn_side import validate
    for i,e in enumerate(episodes):
        assert e['done'] and e['seat']==(start+i)%2 and e['spawn_side_version']=='own-side-native-v1'
        validate(e['spawn_cells']);validate(e['respawn_cells'])
    atomic_json(accepted,dict(status='PASS',opponent=opponent,model=model,protocol_sha256=sha(OUT/'FIXED_HISTORY_EVALUATION_V1.json'),
                             files={str(p):sha(p) for p in (f,side)},unix=time.time()))


def evaluate(req,protocol):
    model=export(req['source'],req['key']);metrics={};files={}
    for key,source in protocol['opponents'].items():
        opponent=export(source,key)
        # Two dedicated CPU cores; GPU training is never a dependency.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(collect,req,key,opponent,model,start,protocol,cpu_cores()[i%2]) for i,start in enumerate(range(0,128,8))]
            for future in futures:future.result()
        episodes=[]
        for start in range(0,128,8):
            f=OUT/'fixed_evaluation'/req['key']/key/f'shard_{start:03d}.json'
            r=bot.read(str(f)+'.ACCEPTED.json');assert all(sha(Path(k))==v for k,v in r['files'].items());files.update(r['files'])
            episodes.extend(bot.read(str(f)+'.outcome.json')['episodes'])
        assert len(episodes)==128==len({(e['seed'],e['seat']) for e in episodes})
        metrics[key]={f'P{s}':summary(episodes,s) for s in (0,1)}
    atomic_json(OUT/'fixed_evaluation'/req['key']/'COMPLETE.json',dict(status='COMPLETE',source=req['source'],
        metrics=metrics,games=256,mode='greedy',teacher=False,files=files,protocol_sha256=sha(OUT/'FIXED_HISTORY_EVALUATION_V1.json'),unix=time.time()))


def main():
    guard=lock(OUT/'FIXED_HISTORY_OWNER.lock');protocol=bot.read(OUT/'FIXED_HISTORY_EVALUATION_V1.json')
    recompute_completed_metrics()
    from scripts import r3_selfplay_owner_20261031 as owner
    owner.OUT=OUT
    while True:
        if owner.paused():atomic_json(OUT/'FIXED_HISTORY_STATUS.json',dict(status='USER_PAUSED',unix=time.time()));return
        if OUT==Path('runs/r3_selfplay_ramp_20261031'):
            from scripts.r3_sp_second_seed_20261031 import safe_advance
            safe_advance()
        request=next((r for r in requests() if not (OUT/'fixed_evaluation'/r['key']/'COMPLETE.json').exists()),None)
        if request is None:
            atomic_json(OUT/'FIXED_HISTORY_STATUS.json',dict(status='CAUGHT_UP_WAIT_SP_CHECKPOINT',unix=time.time()));time.sleep(20);continue
        atomic_json(OUT/'FIXED_HISTORY_STATUS.json',dict(status='CPU_FULL_GAMES',request=request['key'],unix=time.time()))
        evaluate(request,protocol)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(OUT/'FIXED_HISTORY_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()))
        raise
