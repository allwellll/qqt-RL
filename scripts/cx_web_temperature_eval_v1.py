"""Local-only P1 browser inference screen; no GPU, push or deployment."""
import concurrent.futures,json,os,subprocess,time
from pathlib import Path
from scripts import cc_strong_eval_v1 as E
from scripts.cx_temperature_eval_v1 import OUT as TEMP
from scripts.model_launch import manifest,command,public_project
from scripts.population_run_state import atomic_json,sha
OUT=E.OUT/'cx_web_temperature_20261106_v1'
BOTS=['bun.tactical_v2','bun.hunter@hard','bun.hunter@normal','bun.hunter@easy','bun.coop_hunter@hard']
SOURCES=['scripts/native_eval_simulator_v1.js','web/review-sim/bun-native-v2-20261104.js','scripts/cx_web_temperature_worker_v1.js','scripts/cx_web_temperature_setup_v1.js','scripts/cx_web_idle_adapter_v1.js','scripts/resume_gpu67_eval_v1.js','scripts/interceptor_primary_inference_v1.js','scripts/r3_items_inference.js','web/sim-bun-native-v2.js','scripts/model_sampling_v1.js','web/bot_contract.js','web/bun_hunter_bot.js','web/bun_coop_hunter_bot.js','web/bun_rule_bot.js','web/tactical_arena.js','web/assets/maps/levels.json']
# Pin the complete evaluation runtime above; production UI belongs to the release owner.
def make_games(bot,mode,count=64):
 dest=OUT/'games'/f"{bot.replace('@','_')}_{mode}_{count}.json";dest.parent.mkdir(parents=True,exist_ok=True)
 # Node generic entry consumes args pointing to a separate internal configuration.
 private=dest.with_suffix('.config.json');atomic_json(private,dict(bot=bot,mode=mode,count=count,output=str(dest)))
 c=manifest(dict(script='scripts/cx_web_temperature_setup_v1.js',args=[str(private)],env=E.CPU_ENV))
 subprocess.run(command('evaluation',c,node=True),cwd=public_project(),env=os.environ|E.CPU_ENV,check=True)
 return json.loads(dest.read_text())
def freeze(candidate):
 OUT.mkdir(parents=True,exist_ok=True);f=OUT/'PROTOCOL.json'
 if f.exists():return json.loads(f.read_text())
 source=json.loads((TEMP/'PROTOCOL.json').read_text());m=candidate['model'];tau=candidate['temperature']
 combinations=[dict(label=m['label']+'_tau'+str(tau),model=m,temperature=tau,greedy=False),dict(label=m['label']+'_argmax',model=m,temperature=1.,greedy=True),dict(label='G_frozen_tau'+str(tau),model=source['models'][0],temperature=tau,greedy=False)]
 games={bot:{mode:make_games(bot,mode) for mode in ('greedy','categorical')} for bot in BOTS}
 p=dict(status='FROZEN_BEFORE_WEB_SCREEN',unix=time.time(),selection_sha256=sha(TEMP/'SELECTION.json'),source_sha256={f:sha(f) for f in SOURCES},combinations=combinations,games=games,bots=BOTS,seat=1,games_per_bot=64,total=960,ruleset='bun-native-v2',native_items=True,native_traps=True,sampling_seed=0x515154,publish=False,comparison='same model argmax and G_frozen same tau; P1 separate; native full map hunter, actual browser Tactical arena')
 atomic_json(f,p);return p
def shard(p,combo,bot,i,games):
 folder=OUT/'pairs'/combo['label']/bot.replace('@','_')/f'shard_{i:03d}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'COMPLETE.json').exists():return len(games)
 cfg=folder/'CONFIG.json';atomic_json(cfg,dict(candidate=combo['label'],target=bot,bot=bot,model_json=combo['model']['model_json'],opponent_model_json='@bot/'+bot,temperature=combo['temperature'],greedy=combo['greedy'],sampling_seed=p['sampling_seed'],games=games,output=str(folder),protocol_sha256=sha(OUT/'PROTOCOL.json'),source_sha256=p['source_sha256']))
 with E.Slot() as cpu:
  c=manifest(dict(script='scripts/cx_web_temperature_worker_v1.js',args=[str(cfg)],env=E.CPU_ENV))
  with (folder/'worker.log').open('a') as log:subprocess.run(['taskset','-c',str(cpu)]+command('evaluation',c,node=True),cwd=public_project(),env=os.environ|E.CPU_ENV,stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=True)
 return len(games)
def report(p):
 from scripts.cx_temperature_report_v1 import stats
 groups={};inventory=[]
 for combo in p['combinations']:
  for bot in BOTS:
   rows=[]
   for f in (OUT/'pairs'/combo['label']/bot.replace('@','_')).glob('shard_*/COMPLETE.json'):
    for r in json.loads(f.read_text())['results']:
     assert r['seat']==1 and r['protocol_sha256']==sha(OUT/'PROTOCOL.json') and r['review_verified']
     review=f.parent/f"match-{r['seed']}-P1-{r['mode']}-{r['opponent_mode']}"/'review.json';assert sha(review)==r['review_sha256']
     rows.append(r);inventory.append(dict(review=str(review),sha256=r['review_sha256']))
   assert len(rows)==64 and len({r['seed'] for r in rows})==64
   groups[combo['label']+'/'+bot]=stats(rows)
 atomic_json(OUT/'SUMMARY.json',dict(status='COMPLETE_960_AUDITED',groups=groups,inventory=inventory,protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time()))
 lines=['','## 网页v2快筛（模型固定P1）','','真实网页bot_contract适配器；Tactical为实际网页清砖近距场，三档猎手和默认协作猎手困难为完整806地图，native道具/糖泡和v2规则。每组固定新seed64局，三个组合共960局。仅本地、未push。','','| 模型+决策 / 对手 | W/L/D | score | 交付 | 因果杀 | 自死 | 携包失败 | 可行动静止% | 最长静止 | c tick | 喷泉串 |','|---|---|---|---|---|---|---|---|---|---|---|']
 for k,r in groups.items():lines.append(f"| {k} | {r['wins']}/{r['losses']}/{r['draws']} | {r['score']:.3f} | {r['deliveries']} | {r['causal_kills']} | {r['self_deaths']} | {r['carry_failures']} | {r['actionable_stationary_pct']:.2f} | {r['max_actionable_stationary']} | {r['c_ticks']} | {r['fountain_streak']:.3f} |")
 with Path('reports/cx_temperature_eval_20261106.md').open('a') as f:f.write('\n'.join(lines)+'\n\n推送和发布等待用户确认。\n')
 atomic_json(OUT/'STATUS.json',dict(status='COMPLETE_960_AUDITED',report='reports/cx_temperature_eval_20261106.md',unix=time.time()))
def main():
 selection=json.loads((TEMP/'SELECTION.json').read_text());assert selection['candidate'],'No eligible combination'
 p=freeze(selection['candidate']);tasks=[]
 for combo in p['combinations']:
  for bot in BOTS:
   gs=p['games'][bot]['greedy' if combo['greedy'] else 'categorical']
   tasks.extend((p,combo,bot,i,gs[k:k+4]) for i,k in enumerate(range(0,64,4)))
 done=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=min(24,len(E.cpus()))) as pool:
  for f in concurrent.futures.as_completed([pool.submit(shard,*t) for t in tasks]):
   done+=f.result();atomic_json(OUT/'STATUS.json',dict(status='RUNNING',completed=done,total=960,unix=time.time()));print('WEB_COMPLETE',done,flush=True)
 report(p)
if __name__=='__main__':main()
