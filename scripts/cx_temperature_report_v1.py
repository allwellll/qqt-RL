"""Full paired temperature report; artifact audit before selecting model + tau."""
import concurrent.futures,json,os,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.cx_temperature_eval_v1 import OUT,MODES,REPORT,freeze
from scripts import cc_strong_eval_v1 as E
from scripts.cx_holdout_acceptance_v1 import total
from scripts.model_launch import manifest,command,public_project
from scripts.population_run_state import atomic_json,sha

def reused_diagnostics(files):
 def run(i,seq):
  f=OUT/f'reuse_diagnostics_{i:02d}.jsonl'
  if not f.exists():
   cfg=OUT/f'reuse_input_{i:02d}.json';atomic_json(cfg,dict(files=seq,output=str(f)))
   with E.Slot() as cpu:
    c=manifest(dict(script='scripts/cx_idle_replay_v1.js',args=[str(cfg)],env=E.CPU_ENV))
    with (OUT/f'reuse_log_{i:02d}.txt').open('a') as log:subprocess.run(['taskset','-c',str(cpu)]+command('evaluation',c,node=True),cwd=public_project(),env=os.environ|E.CPU_ENV,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
  rows=[json.loads(s) for s in f.read_text().splitlines()];assert len(rows)==len(seq) and all('error' not in r and r['r3_chain_check'] for r in rows);return rows
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  fs=[pool.submit(run,i,files[k:k+44]) for i,k in enumerate(range(0,len(files),44))]
  return {r['review_sha256']:r for f in fs for r in f.result()}

def stats(rows):
 x=total(rows);m=[r['idle'][r['seat']] for r in rows];f=[r['fountain'][r['seat']] for r in rows]
 tick=sum(z['ticks'] for z in m);act=sum(z['actionable'] for z in m)
 x.update(stationary_pct=100*sum(z['stationary'] for z in m)/tick,actionable_stationary_pct=100*sum(z['stationary_actionable'] for z in m)/act,
          max_actionable_stationary=max(z['max_stationary_actionable'] for z in m),p95_actionable_longest=float(np.quantile([z['max_stationary_actionable'] for z in m],.95)),
          mean_actionable_longest=float(np.mean([z['max_stationary_actionable'] for z in m])),c_ticks=sum(z['categories']['c'] for z in m),c_pct_actionable=100*sum(z['categories']['c'] for z in m)/act,
          max_c_run=max(z['max_c_run'] for z in m),fountain_streak=float(np.mean([z['fountain_streak'] for z in f])),strict_fountain_streak=float(np.mean([z['strict_fountain_streak'] for z in f])),ticks=tick)
 return x

def interval(left,right):
 a={(r['seed'],r['seat']):r for r in left};b={(r['seed'],r['seat']):r for r in right};assert set(a)==set(b)
 blocks=defaultdict(list)
 util=lambda r:.5 if r['winner'] is None else float(r['winner']==r['seat'])
 for k,r in a.items():blocks[k[0]].append(util(b[k])-util(r))
 ds=np.array([sum(v) for v in blocks.values()]);ns=np.array([len(v) for v in blocks.values()]);idx=np.random.default_rng(2026616099).integers(0,len(ds),(10000,len(ds)));boot=ds[idx].sum(-1)/ns[idx].sum(-1)
 return dict(mean=float(ds.sum()/ns.sum()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),seeds=len(ds),games=len(a))

def report():
 p=freeze();reused=json.loads((OUT/'REUSE.json').read_text())['rows'];diag=reused_diagnostics([x['review'] for x in reused]);rows=[];inventory=[]
 for z in reused:
  r=json.loads(Path(z['result']).read_text());dd=diag[r['review_sha256']];r=dict(r,mode=z['mode'],temperature=MODES[z['mode']],idle=dd['idle'],fountain=dd['players']);rows.append(r)
  inventory.append(dict(result=z['result'],result_sha256=sha(z['result']),review=z['review'],review_sha256=r['review_sha256'],reused=True))
 for f in sorted((OUT/'pairs').glob('*/P*/shard_*/COMPLETE.json')):
  shard=json.loads(f.read_text())['results']
  for r in shard:
   folder=f.parent/f"match-{r['seed']}-P{r['seat']}-{r['mode']}-{r['opponent_mode']}";actual=json.loads((folder/'RESULT.json').read_text());assert r==actual
   assert r['protocol_sha256']==sha(OUT/'PROTOCOL.json') and sha(folder/'review.json')==r['review_sha256'];assert r['extra_replay_hash_checks']>0
   rows.append(r);inventory.append(dict(result=str(folder/'RESULT.json'),result_sha256=sha(folder/'RESULT.json'),review=str(folder/'review.json'),review_sha256=r['review_sha256'],reused=False))
 assert len(rows)==3520 and len({(r['candidate'],r['mode'],r['seat'],r['seed']) for r in rows})==3520
 groups={};all_groups={};strata={};paired={}
 for m in p['models']:
  label=m['label']
  for mode in MODES:
   rs=[r for r in rows if r['candidate']==label and r['mode']==mode];assert len(rs)==176
   all_groups[label+'/'+mode]=stats(rs)
   for seat in (0,1):
    rr=[r for r in rs if r['seat']==seat];assert len(rr)==88;groups[f'{label}/{mode}/P{seat}']=stats(rr)
   by=defaultdict(list)
   for r in rs:by[f"{r['seat']}/{r['start_region']}/{r['brick_stage']}"].append(r)
   strata[label+'/'+mode]={k:stats(v) for k,v in sorted(by.items())}
   if mode!='greedy':paired[label+'/'+mode]=interval([r for r in rows if r['candidate']==label and r['mode']=='greedy'],rs)
 eligible=[];checks={}
 for m in p['models'][1:]:
  label=m['label'];base=all_groups[label+'/greedy']
  for mode,tau in MODES.items():
   if mode=='greedy':continue
   r=all_groups[label+'/'+mode];c=dict(longest_both_seats_halved=all(groups[f'{label}/{mode}/P{s}']['max_actionable_stationary']<=.5*groups[f'{label}/greedy/P{s}']['max_actionable_stationary'] for s in (0,1)),fountain_mean_nondecreasing=r['fountain_streak']>=base['fountain_streak'])
   checks[label+'/'+mode]=c
   if all(c.values()):eligible.append(dict(model=m,mode=mode,temperature=tau,screening_key=[r['wins']-r['losses'],r['deliveries'],r['causal_kills']],stats=r,reference=all_groups['G_frozen/'+mode],paired_vs_own_argmax=paired[label+'/'+mode]))
 eligible.sort(key=lambda x:tuple(x['screening_key']),reverse=True)
 candidate=eligible[0] if eligible else None
 value=dict(status='COMPLETE_3520_AUDITED',unix=time.time(),protocol_sha256=sha(OUT/'PROTOCOL.json'),games=3520,groups=groups,both_seats=all_groups,strata=strata,paired_vs_own_argmax=paired,eligibility_checks=checks,candidate=candidate,selection='fixed-seed screen, not independent holdout; web P1 screen required; no push yet',inventory=inventory)
 atomic_json(OUT/'SUMMARY.json',value);atomic_json(OUT/'SELECTION.json',dict(status='CANDIDATE_FOR_WEB_P1_SCREEN' if candidate else 'NO_COMBINATION_SATISFIES_BOTH_MECHANISM_GATES',candidate=candidate,unix=time.time(),rule='rank W-L, deliveries, causal kills after both-seat longest>=50% reduction and nondecreasing empirical fountain mean; selfdeath reference, no veto'))
 lines=['# 温度采样与网页候选评估（20261106）','','## 冻结范围与口径','','健康源u019302、h u020742、冻结时SP最新完整u019862，以及同批G_frozen，五种方式：argmax、τ=.3/.5/.7/1.0。每个模型×方式×座88局，总3520局。使用已冻结强评估44场景×2seed，范围2026616000–2026616087；对手始终G_frozen argmax。移动/能力两头均先logits/τ再原生mask+采样，seed及每头RNG序列可复现。既有h/G的argmax与τ1共704局逐项核验复用，其余2816局新跑；旧文件不改，全部额外生产模拟器重放提取发呆/喷泉。','','这批是固定局筛查，h源与seed曾用于选点，不能当新的独立holdout。c是当前已存在危险下安全推进的保守证据，不是完整安全专家。静止%为物理坐标不变；可行动=存活且未受控。最长是全组逐局max，另给P95/均值，避免只看单一极端。喷泉为跨tick有效批次最长串均值。','','## 双座合并结果','','| 模型/方式 | W/L/D | score | 交付 | 因果杀 | 自死 | 携包失败 | 静止%/可行动静止% | 最长/P95/均值静止 | c tick/% | 宽/严格串均值 |','|---|---|---|---|---|---|---|---|---|---|']
 def line(k,r):return f"| {k} | {r['wins']}/{r['losses']}/{r['draws']} | {r['score']:.4f} | {r['deliveries']} | {r['causal_kills']} | {r['self_deaths']} | {r['carry_failures']} | {r['stationary_pct']:.2f}/{r['actionable_stationary_pct']:.2f} | {r['max_actionable_stationary']}/{r['p95_actionable_longest']:.1f}/{r['mean_actionable_longest']:.1f} | {r['c_ticks']}/{r['c_pct_actionable']:.2f} | {r['fountain_streak']:.3f}/{r['strict_fountain_streak']:.3f} |"
 lines += [line(k,r) for k,r in all_groups.items()]
 lines+=['','## 分座结果（P1单列）','','| 模型/方式/座 | W/L/D | score | 交付 | 因果杀 | 自死 | 携包失败 | 静止%/可行动静止% | 最长/P95/均值静止 | c tick/% | 宽/严格串均值 |','|---|---|---|---|---|---|---|---|---|---|']+[line(k,r) for k,r in groups.items()]
 lines+=['','## 同checkpoint温度与argmax配对','','score差按88个环境seed聚类，两座不当独立样本；95%区间是观察估计，多温度选优有选择偏差，不据此宣称整体显著提升。','','| 模型/温度 | score差 | 95%区间 | 最长段双座减半/串不降 |','|---|---|---|---|']
 for k,r in paired.items():lines.append(f"| {k} | {r['mean']:+.4f} | [{r['lower95']:+.4f},{r['upper95']:+.4f}] | {checks.get(k,'G参照')} |")
 lines+=['','## 网页候选选择','','“明显缩短”预先采用双座各自最长可行动静止段至少减半；要求经验宽喷泉串均值不低于同模型argmax，再按W−L、交付、因果杀排序，G同模式参照完整保留。自死不作否决门。']
 if candidate:lines.append('暂选组合：**'+candidate['model']['label']+' + τ='+str(candidate['temperature'])+'**。仅本地实现并跑网页v2 P1快筛，push/发布等待用户确认。')
 else:lines.append('没有组合同时满足静止明显缩短与串不下降；不静默放宽规则、不擅自选默认网页候选，保留真实阻塞证据。')
 lines+=['','网页快筛与FTN2训练回执将作为后续章节追加；所有GPU训练未被本温度评估更改。原始协议、逐局RESULT/review与完整分层/配对JSON位于 '+str(OUT)+'。']
 REPORT.write_text('\n'.join(lines)+'\n');atomic_json(OUT/'STATUS.json',dict(status=value['status'],unix=time.time(),games=3520,report=str(REPORT),web_screen='PENDING' if candidate else 'BLOCKED_NO_ELIGIBLE_COMBINATION'))
 print(json.dumps({'status':value['status'],'candidate':None if not candidate else [candidate['model']['label'],candidate['temperature']]}),flush=True)
 return value

if __name__=='__main__':report()
