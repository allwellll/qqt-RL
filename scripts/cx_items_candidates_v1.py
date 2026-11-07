"""Finite CPU selection under production item physics; never starts/stops training."""
import concurrent.futures,fcntl,json,os,subprocess,sys,time,uuid
from collections import defaultdict
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.model_launch import command,public_project

ROOT=Path('runs/items_candidates_20261107/v1')
TRAIN=Path('runs/cc_train_20261105/training')
REPORT=Path('reports/cx_web_candidates_items_v3_20261107.md')
CPU=dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CC_STRONG_CPUS='24-31')
ARMS=('L2_DUEL','L2_DUEL75','L2_DUEL25','L2_DUEL_SYNC2','L2_FTN2','L2_ATK','L2_SP')
BOTS=('bun.tactical_v2','bun.hunter@hard','bun.hunter@normal','bun.hunter@easy','bun.coop_hunter@hard')

def read(p):return json.loads(Path(p).read_text())
def config(v):
    p=Path('local_config')/time.strftime('%Y%m%d')/('parameters_'+uuid.uuid4().hex+'.json')
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2)
    return str(p)

def freeze():
    path=ROOT/'PROTOCOL.json'
    if path.exists():
        p=read(path)
        for f,h in p['source_sha256'].items():assert sha(f)==h,f
        return p
    ready=Path(os.environ.get('QQT_ITEM_RULE_READY', '.hermes/ITEM_RULE_READY.json'));r=read(ready)
    production=os.path.relpath((ready.parent/r['repository_relative_to_manifest']).resolve())
    engine=production+'/web/sim-bun-native-v3-items.js'
    assert sha(engine)==r['paths_sha256']['web/sim-bun-native-v3-items.js']
    from scripts import cc_strong_eval_v1 as E
    E.OUT=ROOT/'export_cache';E.OUT.mkdir(exist_ok=True,parents=True)
    models=[]
    for arm in ARMS:
        base=TRAIN/('items_v3_20261107_'+arm.lower());cp=sorted((base/arm).glob('u*/COMMITTED.json'))[-1].parent
        assert read(base/'TECHNICAL_ADMISSION.json')['status']=='PASS_NEW_PHYSICS_FINITE_EXACT_MIGRATION_FRESH2_8'
        cm=read(cp/'COMMITTED.json');assert sha(cp/'policy.pt')==cm['policy_sha256']
        m=E.ensure_export(E.resolve(str(cp)));m.update(physical_rule='bun-native-v3-items',policy_sha256=cm['policy_sha256'],role=arm)
        models.append(m);print('EXPORTED',m['label'],flush=True)
    refs=[E.ensure_export(E.resolve(s)) for s in ('G_frozen','L2_S05g@u019302')]
    for m in refs:
        m['role']=m['arm'];m['physical_rule']='historical_weights_evaluated_under_v3'
        if m.get('checkpoint'):m['policy_sha256']=read(Path(m['checkpoint'])/'COMMITTED.json')['policy_sha256']
    sources=['scripts/cx_items_candidates_v1.py','scripts/cx_items_candidate_worker_v1.js','scripts/cx_items_candidate_games_v1.js',
        'scripts/resume_gpu67_eval_v1.js','scripts/cx_idle_replay_v1.js','scripts/cc_strong_metrics_fountain_v1.js',
        'scripts/interceptor_primary_inference_v1.js','scripts/five_item_slots_inference.js','scripts/r3_items_inference.js',
        'scripts/je05_bun_encoding.js','web/five-item-slots-v1.js','web/five-item-slots-model-v1.js',
        'scripts/combat_specialist_g7_source_metrics.js','scripts/r3_items_metrics.js','scripts/combat_regions_metrics_v2.js',
        'scripts/interceptor_primary_chain_metrics_v1.js']
    sources += [production+'/web/'+n for n in ('sim-bun-native-v3-items.js','model_sampling_v1.js','bot_contract.js',
        'bun_hunter_bot.js','bun_coop_hunter_bot.js','bun_rule_bot.js','tactical_arena.js','assets/maps/levels.json')]
    catalog='runs/cc_train_20261105/training/duel_valid_20261107/SCENARIOS.json';sources.append(catalog)
    p=dict(schema='qqt.items-candidate-eval/v1',status='FROZEN_BEFORE_MATCHES',created_unix=time.time(),
        models=models,references=refs,production=production,engine=engine,engine_sha256=sha(engine),
        rule='bun-native-v3-items',source_sha256={f:sha(f) for f in sources},catalog=catalog,
        cpus='24-31 shared per-core flock; max8 subprocesses; no training CPU affinities used',
        player_seat=1,seat_note='production webpage P1 is engine seat1; only this seat in first delivery',
        modes=['argmax','tau0.3'],opponent='G_frozen tau0.3 paired action RNG',
        selection='specialist selectable model: independently repeated dev+holdout strength; no overall supremacy or zero-death gate; no default switch',
        screen=dict(full_games_per_mode=4,duel_games_per_mode=8,all_seven_arms=True),
        confirmation=dict(duel_dev_games_per_mode=16,duel_holdout_games_per_mode=16,
                          strong_full_holdout_per_mode=8,web_games_per_bot_per_mode=8),
        seed_ranges=dict(screen=2026791000,dev=2026793000,holdout=2026795000,web=2026797000),
        safeguards='CPU only; no online assistance; original policy/movement masks and reward; fixed horizon, all losses retained',
        cache='reuse only exact protocol/model/engine/game/action RNG match; v2 results excluded')
    atomic_json(path,p)
    REPORT.write_text('# 新物理网页可选模型筛选（20261107）\n\n已冻结七臂最新COMMITTED与G_frozen、健康u019302；所有比赛使用生产bun-native-v3-items引擎。旧物理评测不复用为新证据。模型固定网页P1（engine seat1），分别argmax/τ0.3，对照G_frozen τ0.3；不改训练、不push、不调用CC。协议与逐局RESULT/review保存在 `'+str(ROOT)+'`。专项长处可作为可选，不要求全面更强，不改变默认模型。\n\n阶段：实际导出完成，正在生产JS筛查；完成独立dev/holdout及网页快筛后才写ready候选。\n')
    return p

def games(p,kind,split,count):
    target=ROOT/'games'/f'{kind.replace("@","_")}_{split}_{count}.json';target.parent.mkdir(exist_ok=True)
    if not target.exists():
        c=config(dict(script='scripts/cx_items_candidate_games_v1.js',args=[config(dict(production=p['production'],engine=p['engine'],catalog=p['catalog'],kind=kind,split=split,count=count,output=str(target),seeds=p['seed_ranges']))],env=CPU))
        subprocess.run(command('evaluation',c,node=True),cwd=public_project(),env=os.environ|CPU,check=True,timeout=60)
    return read(target)

def run_job(p,phase,m,mode,kind,split,index,gs):
    folder=ROOT/phase/m['label']/mode/kind.replace('@','_')/split/f'shard_{index:03d}';folder.mkdir(parents=True,exist_ok=True)
    f=folder/'COMPLETE.json';ph=sha(ROOT/'PROTOCOL.json')
    if f.exists():
        rows=read(f)['results'];assert len(rows)==len(gs)
        assert all(r['protocol_sha256']==ph and r['model_json_sha256']==m['model_json_sha256'] and r['simulator_sha256']==p['engine_sha256'] for r in rows)
        return rows
    gs=[dict(g,mode='greedy' if mode=='argmax' else 'tau03',opponent_mode='tau03') for g in gs]
    value=dict(candidate=m['label'],target=kind,bot=kind if kind in BOTS else None,model_json=m['model_json'],
        opponent_model_json='@bot/'+kind if kind in BOTS else p['references'][0]['model_json'],
        greedy=mode=='argmax',temperature=.3,sampling_seed=0x515154,games=gs,split=split,
        output=str(folder),protocol_sha256=ph,source_sha256=p['source_sha256'],engine=p['engine'],production=p['production'])
    atomic_json(folder/'CONFIG.json',value)
    from scripts.cc_strong_eval_v1 import Slot
    with Slot() as cpu,(folder/'worker.log').open('a') as log:
        c=config(dict(script='scripts/cx_items_candidate_worker_v1.js',args=[str(folder/'CONFIG.json')],env=CPU))
        subprocess.run(['taskset','-c',str(cpu)]+command('evaluation',c,node=True),cwd=public_project(),env=os.environ|CPU,
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    rows=read(f)['results'];assert all(r['review_verified'] and r['extra_replay_hash_checks']>0 for r in rows)
    return rows

def phase(p,name,models,specs):
    tasks=[]
    # References interleaved with candidate checkpoints, so paired results arrive together.
    for kind,split,count in specs:
        gs=games(p,kind,split,count)
        for mode in ('argmax','tau0.3'):
            for start in range(0,len(gs),2):
                for m in models:tasks.append((p,name,m,mode,kind,split,start//2,gs[start:start+2]))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(run_job,*t) for t in tasks]
        for f in concurrent.futures.as_completed(fs):
            rows+=f.result();atomic_json(ROOT/'STATUS.json',dict(status='RUNNING',phase=name,completed=len(rows),total=sum(len(t[-1]) for t in tasks),unix=time.time()))
            print(name,'COMPLETE',len(rows),'/',sum(len(t[-1]) for t in tasks),flush=True)
    summary=summarize(rows);atomic_json(ROOT/(name+'_SUMMARY.json'),dict(status='COMPLETE',groups=summary,rows=rows))
    append_report(name,summary)
    return rows,summary

def summarize(rows):
    from scripts.cx_temperature_report_v1 import stats
    groups=defaultdict(list)
    for r in rows:groups['/'.join((r['candidate'],'argmax' if r['mode']=='greedy' else 'tau0.3',r['target'],r['split']))].append(r)
    return {k:stats(v) for k,v in groups.items()}

def append_report(name,groups):
    with REPORT.open('a') as f:
        f.write('\n## '+name+' 实际完成\n\n|模型/方式/任务/分集|局数|胜/负/平|score|交付|因果杀|自死|携包失败|可行动静止%/最长|c tick|喷泉串|\n|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|\n')
        for k,s in groups.items():f.write(f"|{k}|{s['games']}|{s['wins']}/{s['losses']}/{s['draws']}|{s['score']:.3f}|{s['deliveries']}|{s['causal_kills']}|{s['self_deaths']}|{s['carry_failures']}|{s['actionable_stationary_pct']:.2f}/{s['max_actionable_stationary']}|{s['c_ticks']}|{s['fountain_streak']:.3f}|\n")

def select(p,groups):
    # Predeclared role split: one kill-oriented and one flow/transport representative.
    scores=[]
    for m in p['models']:
        rs=[v for k,v in groups.items() if k.startswith(m['label']+'/')]
        kill=sum(v['causal_kills'] for v in rs);deliver=sum(v['deliveries'] for v in rs)
        fountain=sum(v['fountain_streak'] for v in rs);still=sum(v['actionable_stationary_pct'] for v in rs)
        net=sum(v['wins']-v['losses'] for v in rs)
        scores.append(dict(model=m,kill=kill,deliveries=deliver,fountain=fountain,stationary=still,net=net))
    attack=max(scores,key=lambda x:(x['kill'],x['net'],x['fountain']))
    other=max((s for s in scores if s!=attack),key=lambda x:(x['deliveries'],x['fountain'],-x['stationary'],x['net']))
    result=dict(status='SELECTED_FOR_INDEPENDENT_CONFIRMATION',candidates=[attack,other],all_screen=scores,
        caution='screen selection is exploratory; only disjoint holdout can justify optional recommendation')
    atomic_json(ROOT/'SELECTION.json',result)
    return [attack['model'],other['model']]

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ROOT.mkdir(parents=True,exist_ok=True);guard=(ROOT/'owner.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(CPU)
    p=freeze();action=sys.argv[1] if len(sys.argv)>1 else 'all'
    if action=='smoke':
        return phase(p,'smoke',p['models'][:1],[('duel','dev',1),('bun.coop_hunter@hard','web',1)])
    screen,summary=phase(p,'screen',p['models']+p['references'],[('strong_full','screen',4),('duel','screen',8)])
    candidates=select(p,summary)
    if action=='screen':return
    selected=candidates+p['references']
    phase(p,'confirm',selected,[('duel','dev',16),('duel','holdout',16),('strong_full','holdout',8)])
    phase(p,'web',selected,[(bot,'web',8) for bot in BOTS])
    from scripts.cx_items_candidates_report_v1 import finish
    finish(p)

if __name__=='__main__':main()
