"""Passive, recomputable metrics table. Engine/rule/split/mode never mix."""
from __future__ import annotations
import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from scripts.population_run_state import atomic_json,lock,sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/iter_pbt_v1'
OUT=BASE/'natural_js_table_20261028'
NEW=ROOT/'runs/bomb_threat_v1'
COLS=[('model','模型'),('mode','动作'),('rule','规则'),('split','集合'),('seed','seed起点'),
 ('games','局数'),('win','胜'),('draw','平'),('loss','负'),('win_minus_loss','胜-负'),('draw_kill_difference','平局击杀差'),('causal_kills','因果击杀'),
 ('surviving_kills','击杀且存活'),('deaths','死亡次数'),('self_deaths','自伤死亡'),('delivery','交付'),
 ('bombs_per_game','放泡/局'),('zero_bomb_games','单方零泡局'),('structural_zero_bomb_games','双方零泡局（结构口径）'),('raw_zero_bomb_games','我方零泡合计'),('cage_kill_fraction','笼内击杀占比'),
 ('draw_fraction','平局率'),('threat_reward_fraction','威胁+炸砖/总奖励'),
 ('topology_reward_events_per_training_episode','炸砖事件/训练重置'),('best_gate','BEST绝对闸门'),('pbt_gate','PBT相对闸门'),('previous_best_gate','旧BEST判定'),('previous_pbt_gate','旧PBT判定'),('flags','异常/缺失')]
COLS[-1:-1]=[(f'{seat}_{key}',f'{seat} {label}') for seat in ('P0','P1')
 for key,label in [('games','局数'),('w_d_l','胜/平/负'),('bombs_per_game','放泡/局'),
                   ('zero_bomb_games','单方零泡'),('structural_zero_bomb_games','双方零泡'),('causal_kills','击杀')]]
LOW={'loss','deaths','self_deaths','zero_bomb_games','cage_kill_fraction','draw_fraction','threat_reward_fraction'}
NEUTRAL={'games','seed','bombs_per_game'}

def load(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}

def legacy_rows():
 groups=defaultdict(list)
 for p in sorted(OUT.glob('*.json')):
  match=re.fullmatch(r'(.+)_(greedy|categorical)_(\d+)\.json',p.name)
  if match:groups[match.group(1),match.group(2)].append(p)
 rows=[]
 for (name,mode),paths in sorted(groups.items()):
  episodes=[];settled=[];seen=set();flags=[]
  for p in paths:
   eps=load(p).get('results',{}).get('natural_full2400',{}).get('episodes',[])
   side=load(Path(str(p)+'.outcome.json'));ss=side.get('episodes',[])
   # The legacy game0 Review verifier created one extra exact replay Sim.
   # Discard only its provable adjacent duplicate; otherwise outcomes unknown.
   if len(ss)==len(eps)+1 and len(ss)>1 and ss[0]==ss[1]:ss=ss[:1]+ss[2:]
   valid=len(ss)==len(eps) and all(s.get('done') and s.get('ticks')==e.get('ticks') for s,e in zip(ss,eps))
   if not valid:flags.append('结算sidecar缺失/不匹配')
   for i,e in enumerate(eps):
    if e['seed'] in seen:raise ValueError('duplicate legacy seed: '+name)
    seen.add(e['seed']);episodes.append(e)
    if valid:settled.append((e,ss[i]))
  if not episodes:continue
  n=len(episodes);good=len(settled)==n
  win=sum(s['winner']==e['team'] for e,s in settled) if good else None
  draw=sum(s['winner'] in (None,-1) for _,s in settled) if good else None
  row=dict(model=name,mode=mode,rule='legacy_stock',split='historical_fixed',seed=min(seen),games=n,
   win=win,draw=draw,loss=n-win-draw if good else None,
   causal_kills=sum(e['causal_kill'] for e in episodes),surviving_kills=sum(e['surviving_kill'] for e in episodes),
   deaths=sum(e['death'] for e in episodes),self_deaths=sum(e['self_death'] for e in episodes),
   delivery=sum(e['delivery'] for e in episodes),
   bombs_per_game=sum(s['bombs'][e['team']] for e,s in settled)/n if good else None,
   zero_bomb_games=sum(s['bombs'][e['team']]==0 and s['bombs'][1-e['team']]>0 for e,s in settled) if good else None,
   structural_zero_bomb_games=sum(s['bombs']==[0,0] for _,s in settled) if good else None,
   raw_zero_bomb_games=sum(s['bombs'][e['team']]==0 for e,s in settled) if good else None,
   draw_fraction=draw/n if good else None,flags='; '.join(sorted(set(flags))) or '历史规则，不参与新规则晋升')
  rows.append(row)
 return rows

def rows():
 out=legacy_rows()
 for p in sorted((NEW/'eval').glob('*/NATURAL_COMPLETE.json')):
  v=load(p)
  if v.get('measurement_version')!=2:continue
  if (p.parent/'EXCLUDED_INVALID.json').exists():continue
  summary=load(p.parent/'SUMMARY.json');gate=load(p.parent/'GATE_METRICS.json');hack=gate.get('hack_monitor',summary.get('hack_monitor',{}))
  protocol=load(NEW/'protocols'/(Path(v.get('checkpoint') or '').parent.name+'.json'))
  entropy_variant=['STEER1320: A2已停退火，完整state续训，固定熵0.005'] if protocol.get('entropy_restored_1320') else []
  if protocol.get('iter2_arm')=='A1':entropy_variant+=['原rule3/jax_hunter均为hunter_tier3a/b；A1未降低独立弱对手比例']
  for mode,r in gate.get('natural_js_full2400',v.get('natural_js_full2400',{})).items():
   seats=seat_columns(r)
   flags=entropy_variant+summary.get('hack_flags',[])+gate.get('best_gate_failures',summary.get('hard_gate_failures',[]))
   flags=flags+gate.get('best_gate_missing',[])+gate.get('pbt_gate_missing',[])+gate.get('pbt_gate_failures',[])
   out.append(dict(model=p.parent.name,mode=mode,rule='stock-window-v1',split=v['split'],seed=v['seed'],**seats,
    **{k:r.get(k) for k in ('games','win','draw','loss','win_minus_loss','draw_kill_difference','causal_kills','deaths','self_deaths','delivery','bombs_per_game','zero_bomb_games','structural_zero_bomb_games','raw_zero_bomb_games','cage_kill_fraction','draw_fraction')},
    best_gate=gate.get('absolute_gate_status',('PASS' if gate.get('best_eligible') else 'FAIL')) if gate else 'INCOMPLETE',
    pbt_gate=gate.get('pbt_gate_status',('PASS' if gate.get('pbt_selection_eligible') else 'FAIL')) if gate else 'INCOMPLETE',
    previous_best_gate=gate.get('previous_best_gate'),previous_pbt_gate=gate.get('previous_pbt_gate'),
    threat_reward_fraction=hack.get('threat_reward_fraction_total'),
    topology_reward_events_per_training_episode=hack.get('topology_reward_events_per_training_episode'),
    flags='; '.join(dict.fromkeys(([f'ITER2 {v["iter2_arm"]}; {v.get("reward_version")}'] if v.get('iter2_arm') else [])+flags)) or ('完整自然局；快筛待完成' if not summary else 'BEST与PBT分级判定；未执行复制')))
  half=load(p.parent/'TEMPERATURE_HALF_COMPLETE.json')
  if half and half.get('policy_sha256')==v.get('policy_sha256') and half.get('games')==64 and half.get('tau')==.5 and half.get('gate_input') is False:
   from scripts.bomb_threat_eval import summarize
   half_rows=[]
   for start in range(0,64,8):
    side=p.parent/f'natural_tau0.5_{start:03d}.json.outcome.json'
    if side.exists() and side.is_relative_to(ROOT) and half.get('files',{}).get(str(side.relative_to(ROOT)))==sha(side):
     half_rows+=load(side)['episodes']
   half_metrics=dict(half['metrics'])
   if len(half_rows)==64:
    measured=summarize(half_rows)
    half_metrics.update({k:measured[k] for k in ('zero_bomb_games','structural_zero_bomb_games','raw_zero_bomb_games','zero_bomb_measurement','by_seat')})
   out.append(dict(model=p.parent.name,mode='categorical_tau0.5',rule='stock-window-v1',split=v['split'],seed=v['seed'],
    **half_metrics,**seat_columns(half_metrics),best_gate='NOT_APPLICABLE',pbt_gate='NOT_APPLICABLE',
    flags=f'ITER2 {v.get("iter2_arm")}; 独立温度诊断，老师关闭，不参与argmax闸门'))
 return out

def seat_columns(metrics):
 result={}
 for seat,values in metrics.get('by_seat',{}).items():
  if values is None:continue
  result.update({f'{seat}_{key}':values.get(key) for key in
   ('games','bombs_per_game','zero_bomb_games','structural_zero_bomb_games','causal_kills')})
  result[f'{seat}_w_d_l']=f"{values['win']}/{values['draw']}/{values['loss']}"
 return result

def color(vals,v,low=False):
 nums=[x for x in vals if isinstance(x,(int,float))]
 if not isinstance(v,(int,float)) or len(set(nums))<2:return ''
 rank=(v-min(nums))/(max(nums)-min(nums));rank=1-rank if low else rank
 return f'background:rgb({int(248-149*rank)},{int(105+85*rank)},{int(107+16*rank)})'

def display(v):
 return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)

def table(data,cols):
 heads=''.join(f'<th>{html.escape(h)}</th>' for _,h in cols);body=[]
 for row in data:
  cells=[]
  peers=[r for r in data if (r.get('rule'),r.get('mode'),r.get('split'),r.get('seed'))==(row.get('rule'),row.get('mode'),row.get('split'),row.get('seed'))]
  for k,_ in cols:
   v=row.get(k);style='' if k in NEUTRAL else color([x.get(k) for x in peers],v,k in LOW)
   if 'window' in row:style=''
   if k=='anomalies' and v:style='background:#f6a0a0'
   if k=='data_gaps':style=''
   if k=='flags' and any(f in str(v) for f in ('HIGH_','REGRESSION','BOMBS_UP','ARGMAX_','缺失','不匹配')):style='background:#f6a0a0'
   cells.append(f'<td style="{style}">{html.escape(display(v))}</td>')
  body.append('<tr>'+''.join(cells)+'</tr>')
 h='<div class="wrap"><table><thead><tr>'+heads+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
 m='|'+'|'.join(h for _,h in cols)+'|\n|'+'|'.join('---' for _ in cols)+'|\n'
 for r in data:m+='|'+'|'.join(display(r.get(k)).replace('|','/') for k,_ in cols)+'|\n'
 return h,m

def auxiliary_rows():
 data=[]
 for root in (BASE/'eval',NEW/'eval'):
  for p in sorted(root.glob('*/SUMMARY.json')):
   if (p.parent/'EXCLUDED_INVALID.json').exists():continue
   v=load(p);rule=v.get('rule_version','legacy_stock')
   for tier,r in v.get('hunter_js',v.get('hunter_js_clear',{})).items():
    data.append(dict(model=p.parent.name,engine='JS Hunter 清砖/'+tier,rule=rule,split=v.get('split'),seed=v.get('seed'),games=r.get('games'),win=r.get('safe_win'),deaths=r.get('death_episode'),mode='greedy'))
   for mode,r in v.get('natural_jax',{}).items():
    data.append(dict(model=p.parent.name,engine='JAX native',rule=rule,split=v.get('split'),seed=v.get('seed'),games=r.get('games'),delivery=r.get('capture_episodes'),deaths=r.get('death_episodes'),mode=mode))
 for p in sorted((NEW/'eval').glob('*/native/NATIVE_SUMMARY.json')):
  complete=load(p.parent/'NATIVE_COMPLETE.json')
  if complete.get('summary_sha256')!=sha(p):continue
  v=load(p)
  for mode,r in v.get('natural_jax',{}).items():
   data.append(dict(model=p.parent.parent.name,engine='JAX rule3 原生出生/seat0（非JS Hunter）',
    rule=v.get('rule_version'),split=v.get('split'),seed=v.get('seed'),games=r.get('games'),
    win=r.get('win'),deaths=r.get('death_episodes'),delivery=r.get('capture_episodes'),mode=mode))
 for p in sorted((NEW/'eval').glob('*/native_hunter256_v1/NATIVE_HUNTER_SUMMARY.json')):
  complete=load(p.parent/'NATIVE_HUNTER_COMPLETE.json')
  if complete.get('files',{}).get(p.name)!=sha(p) or complete.get('gate_input') is not False:continue
  v=load(p)
  for mode,r in v.get('natural_jax_hunter',{}).items():
   data.append(dict(model=p.parent.parent.name,engine='JAX Hunter 带砖整局256（辅助/非闸门）',
    rule=v.get('version'),split=v.get('split'),seed=v.get('seed_start'),games=r.get('games'),
    win=r.get('win'),draw=r.get('draw'),loss=r.get('loss'),causal_kills=r.get('causal_kills'),
    bombs_per_game=r.get('bombs_per_game'),zero_bomb_games=r.get('zero_bomb_games'),
    deaths=r.get('deaths'),delivery=r.get('delivery'),mode=mode))
 return data

def training_rows():
 data=[]
 for b,rule in ((BASE,'legacy_stock'),(NEW,'stock-window-v1')):
  for run in sorted((b/'formal').glob('*')):
   updates={}
   try:
    for l in (run/'updates.jsonl').read_text().splitlines():
     try:r=json.loads(l);updates[r['new_update']]=r
     except (ValueError,KeyError):continue
   except OSError:pass
   for p in sorted(run.glob('u*/COMMITTED.json')):
    v=load(p);r=updates.get(v.get('update'),v.get('metrics',{}));c=r.get('configuration',{})
    data.append(dict(model=run.name+'/'+p.parent.name,rule=rule,update=v.get('update'),steps=v.get('environment_steps'),finite=r.get('finite'),
     env_s=r.get('actual_environment_steps_this_update',0)/r['cycle_seconds'] if r.get('cycle_seconds') else None,
     lr=r.get('actor_lr',c.get('lr')),entropy=r.get('entropy_coefficient',c.get('entropy_coefficient')),envs=c.get('envs_per_device'),mb=c.get('global_minibatch'),
     curriculum=c.get('curriculum_fraction'),opponents=c.get('opponent_weights'),
     status='EXCLUDED' if any((run/f).exists() for f in ('INVALID_STARTING_POINT.json','EXCLUDED_INVALID.json','EXCLUDED_EVALUATION.json')) else '原子保存',policy_hash=v.get('policy_sha256')))
 return data

def monitor_rows():
 """Training-side 40-update windows; unavailable collector fields stay visible."""
 p=NEW/'TRAIN_MONITOR.jsonl'
 if not p.exists():return []
 out=[];alerts=defaultdict(list)
 try:
  for line in (NEW/'TRAIN_ALERTS.jsonl').read_text().splitlines():
   a=json.loads(line);alerts[a['run'],a['window']].append(a)
 except (OSError,ValueError):pass
 try:lines=p.read_text().splitlines()
 except OSError:return []
 for line in lines:
  try:r=json.loads(line)
  except (ValueError,TypeError):continue
  e=r.get('exposure',{});q=r.get('per_100_tick',{});rew=r.get('reward',{})
  frac=rew.get('fractions_of_absolute_total',{})
  out.append(dict(run=r.get('run'),window=r.get('window'),updates=f"{r.get('first_update')}–{r.get('last_update')}",
   bombs_per100=q.get('bombs_placed'),rejects_per100=q.get('requests_rejected'),
   threats_per100=q.get('threat_events'),topology_per100=q.get('topology_events'),
   captures_per100=q.get('captures'),deaths_per100=q.get('deaths'),kills_per100=q.get('kills'),
   avg_ticks=r.get('avg_episode_ticks'),clear_ratio=r.get('clear_scene_ratio'),bank_ratio=r.get('bank_ratio'),
   transport=rew.get('transport_chain'),kill_control=rew.get('kill_control'),shaping=rew.get('threat_topology_chain'),route_pbrs=rew.get('route_pbrs'),
   shaping_fraction=frac.get('threat_topology_chain'),entropy=r.get('policy_entropy'),entropy_source=r.get('policy_entropy_source'),bomb_probability=r.get('bomb_action_probability'),bomb_argmax=r.get('bomb_argmax_fraction'),scope=r.get('scope'),
   env_s=r.get('env_per_second'),actor_l2=r.get('update_l2',{}).get('actor'),critic_l2=r.get('update_l2',{}).get('critic'),
   finite=r.get('finite'),anomalies=','.join(a['code'] for a in alerts[r['run'],r['window']] if a.get('severity')=='ANOMALY'),
   data_gaps=','.join(a['code'] for a in alerts[r['run'],r['window']] if a.get('severity')=='DATA_GAP'),missing=','.join(r.get('missing_data',[]))))
 return out

def render(data):
 now=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
 note='生产JS带砖整局与清砖/原生JAX分开；legacy_stock与stock-window-v1不可跨规则比较。每行披露seed与局数；缺失显示—，不当成0。颜色仅同规则/seed/集合/动作内比较；放泡数不等于能力，不按越少越好着色。威胁奖励占比是截至checkpoint训练前缀的原生净总奖励分母；非整局终局奖励，修订2以三项新增塑形合并实付的30%门槛判定，不能据此前缀推断每种信号整局付费。闸门仅按argmax判定，采样同行只作报告；INCOMPLETE不计FAIL；BEST通过基础条件后仍须fresh每档256局复核且当前不晋升。老师关闭。'
 note+=' 单方零泡仅计我方实际0且对手实际>0；双方0单列结构口径，不证明永久不可达，不删除任何胜负/局数分母；相对≤parent+3，绝对≤7。原生JAX辅助历史列保留各自原口径。'
 sections=[('一、真实网页带砖整局',data,COLS),('二、清砖快筛与原生 JAX（辅助，不作为新规则晋升依据）',auxiliary_rows(),[('model','模型'),('engine','引擎/对手'),('rule','规则/辅助协议'),('split','集合'),('mode','动作'),('seed','seed'),('games','局数'),('win','胜/历史安全胜'),('draw','平'),('loss','负'),('causal_kills','因果击杀'),('bombs_per_game','泡/局'),('zero_bomb_games','零泡局'),('deaths','死亡/历史死亡局'),('delivery','交付/历史交付局')]),
 ('三、训练侧监控（每40 update；缺字段明确未记录）',monitor_rows(),[('run','run'),('window','窗口'),('updates','更新范围'),('bombs_per100','落泡/100tick'),('rejects_per100','拒绝/100tick'),('threats_per100','威胁/100tick'),('topology_per100','拓扑/100tick'),('captures_per100','交付/100tick'),('deaths_per100','死亡/100tick'),('kills_per100','击杀/100tick'),('avg_ticks','平均局tick'),('clear_ratio','清场占比'),('bank_ratio','Bank占比'),('transport','运包链'),('kill_control','击杀/控制'),('shaping','威胁/拓扑/连锁'),('route_pbrs','route PBRS'),('shaping_fraction','塑形/绝对回报'),('entropy','总熵'),('entropy_source','熵口径'),('bomb_probability','放泡概率'),('bomb_argmax','argmax放泡比例'),('env_s','env/s'),('actor_l2','Actor L2'),('critic_l2','Critic L2'),('finite','finite'),('scope','当前/历史'),('anomalies','真实异常'),('data_gaps','数据缺失告警'),('missing','缺失字段')]),
 ('四、全部原子 checkpoint 与训练配置',training_rows(),[('model','checkpoint'),('rule','规则'),('update','更新'),('steps','累计环境步'),('finite','finite'),('env_s','有效env/s'),('lr','LR'),('entropy','熵系数'),('envs','env/卡'),('mb','minibatch'),('curriculum','课程'),('opponents','对手权重'),('status','状态'),('policy_hash','policy SHA256')])]
 h='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>模型指标表</title><style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ccc;padding:4px 7px;white-space:nowrap}th{background:#eee;position:sticky;top:0}.wrap{overflow:auto;max-height:75vh}</style><h1>模型评估指标总表</h1><p>更新时间：'+now+'</p><p>'+html.escape(note)+'</p>'
 m='更新时间：'+now+'\n\n'+note+'\n'
 for title,rs,cs in sections:
  th,tm=table(rs,cs);h+='<h2>'+html.escape(title)+'</h2>'+th;m+='\n## '+title+'\n\n'+tm
 status=load(NEW/'TRAIN_MONITOR_STATUS.json')
 counts='训练告警分级（缺失数据不标红）：'+json.dumps(status.get('alert_counts_by_severity',{}),ensure_ascii=False)+'；当前/历史：'+json.dumps(status.get('alert_counts_by_scope',{}),ensure_ascii=False)
 h+='<p>'+html.escape(counts)+'</p>';m+='\n\n'+counts+'\n'
 return h+'</html>',m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--upload',action='store_true');args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'BUILD.lock')
 from scripts.bomb_threat_train_monitor import refresh
 try:refresh(NEW)
 except BlockingIOError:pass
 h,m=render(rows())
 hp=OUT/'20261028-model-metrics-table.html';mp=ROOT/'reports/model_metrics_table.md'
 for p,v in ((hp,h),(mp,m)):
  tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(v);tmp.replace(p)
 if args.upload:
  wrapper=Path(os.environ.get('QQT_METRICS_UPLOAD_WRAPPER',str(NEW/'upload_metrics_local.py')))
  if not wrapper.exists():raise SystemExit('Missing ignored local upload wrapper; upload remains pending')
  subprocess.run([sys.executable,str(wrapper),str(hp.relative_to(ROOT))],check=True,timeout=120)
 atomic_json(NEW/'METRICS_GENERATED.json',dict(unix=time.time(),html=str(hp.relative_to(ROOT)),markdown=str(mp.relative_to(ROOT))))
if __name__=='__main__':main()
