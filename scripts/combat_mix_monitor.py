"""Passive report/notification consumer for frozen COMBAT tasks; no GPU control."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.combat_mix_owner import notify
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/combat_mix_v1'
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def model_rows():
 p=read(BASE/'js_web_compare_v1/PROTOCOL.json',{})
 result={}
 for name in p.get('models',{}):
  path=BASE/'js_web_compare_v1'/name/'SUMMARY.json';v=read(path)
  if not v:continue
  result[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
    hunter={tier:dict(games=d['games'],safe_win=d['safe_win'],deaths=d['death_episode'],bombs_per_game=d['bombs_per_game']) for tier,d in v['hunter_js'].items()},
    natural={mode:{k:d.get(k) for k in ('games','window_events','window_stock_pickup','window_stock_delivery','pickup','delivery','death','stock_pickup_free_alive')} for mode,d in v['natural_js'].items()})
 return result

def main():
 guard=lock(BASE/'MONITOR.lock')
 while True:
  from scripts.combat_mix_summary import main as refresh_paired
  try:refresh_paired()
  except Exception as exc:
   atomic_json(BASE/'SUMMARY_FAILURE.json',dict(error=type(exc).__name__,detail=str(exc).replace(str(ROOT),'<WORKTREE>'),unix=time.time()))
  status=read(BASE/'STATUS.json',{});rows=model_rows()
  online_path=BASE/'js_online_actual_v1/online_catalog_first_je10g8/SUMMARY.json'
  online=read(online_path)
  summary=dict(status=status.get('status'),active=status.get('active'),complete=list(status.get('complete',{})),
   failures=status.get('failures'),JS_web_comparison=rows,unix=time.time(),no_GPU_control=True)
  atomic_json(BASE/'LIVE_EVIDENCE.json',summary)
  lines=['# 网页模型与试点同协议比较（20261027）','',
   '冻结协议 runs/combat_mix_v1/js_web_compare_v1/PROTOCOL.json；未改变网页默认，未发布。实际局数、模型hash、完整逐局记录保留在各模型目录。生产JS快速Hunter与自然完整局、categorical与greedy分列。原12模型冻结时线上下载503；后续真实线上身份、发布raw/ONNX双hash已核验，补充协议见 runs/combat_mix_v1/js_online_actual_v1/PROTOCOL.json。原键 online_web_default 实际是 selfplay3072，非当前目录首项；页面默认对手是协作Hunter困难，模型不自动加载。','',
   '|模型|Hunter难/中/易存活击杀（各64）|自然categorical交付/局|自然greedy交付/局|',
   '|---|---|---|---|']
  for n,r in rows.items():
   h='/'.join(str(r['hunter'][t]['safe_win']) for t in ('hard','normal','easy'))
   a=r['natural']['categorical'];b=r['natural']['greedy']
   label="selfplay3072（原键online_web_default）" if n=="online_web_default" else n
   lines.append(f"|{label}|{h}|{a['delivery']}/{a['games']}|{b['delivery']}/{b['games']}|")
  lines+=['','当前完成 '+str(len(rows))+'/12 模型；未完成项不视为0。自然交付是事件数，不能不检查逐局表就当胜率。','',
   'COMBAT训练状态见 runs/combat_mix_v1/LIVE_EVIDENCE.json；报告消费不控制训练。单seed/单点不作为最佳晋升；三个配对seed和尾部需齐全后汇总。']
  if online:
   h='/'.join(str(online['hunter_js'][t]['safe_win']) for t in ('hard','normal','easy'))
   a=online['natural_js']['categorical'];b=online['natural_js']['greedy']
   lines+=['','补充真实线上目录首项（发布raw/ONNX双hash相同，原12模型不改）：',
    f"JE10 r01 g8：Hunter难/中/易={h}（各64局）；自然categorical交付事件{a['delivery']}/{a['games']}，greedy={b['delivery']}/{b['games']}。",
    '补充协议及逐局证据：runs/combat_mix_v1/js_online_actual_v1/。']
   notify('JS_ACTUAL_ONLINE_COMPLETE','真实线上目录首项JE10g8同种子JS对比完成：192局Hunter+128局自然完整，raw/ONNX双hash与线上一致；GPU训练继续。',dict(summary=str(online_path.relative_to(ROOT)),summary_sha256=sha(online_path)))
  (ROOT/'reports/js_web_compare_20261027.md').write_text('\n'.join(lines)+'\n')
  for n in rows:
   notify('JS_COMPARE_'+n,f'网页同协议对比完成：{n}，192局JS Hunter+128局自然完整，分模式结果已落盘，GPU训练继续。',rows[n])
  # Append/replace only a bounded live section in this experiment's report.
  report=ROOT/('reports/combat_mix_experiment_'+time.strftime('%Y%m%d')+'.md');old=report.read_text() if report.exists() else (ROOT/'reports/combat_mix_experiment_20261027.md').read_text()
  marker='\n## 自动实测状态\n'
  if marker in old:
   prefix,tail=old.split(marker,1)
   if '\n<!-- END_AUTO_STATE -->' in tail:
    old=prefix+tail.split('\n<!-- END_AUTO_STATE -->',1)[1]
   elif '\n## Control 三seed固定终态' in tail:
    old=prefix+'\n## Control 三seed固定终态'+tail.split('\n## Control 三seed固定终态',1)[1]
   else:old=prefix
  live=[marker,'训练状态：'+str(status.get('status'))+'。']
  for a in status.get('active',[]):
   p=a.get('progress',{});live.append(f"- {a['run']}：GPU{a['gpu']}，u{p.get('update')}，finite={p.get('finite')}，env/s={p.get('env_per_second')}。")
  live.append('已验收固定run：'+str(len(status.get('complete',{})))+'/'+str(len(read(BASE/'PROTOCOL.json',{}).get('arms',{}))*3)+'；完整更新/SHA/吞吐见 STATUS.json 和各run updates.jsonl。')
  live.append('CPU网页完成：'+str(len(rows))+'/12；统计见 reports/js_web_compare_20261027.md。')
  report.write_text(old+'\n'.join(live)+'\n<!-- END_AUTO_STATE -->\n')
  if status.get('status')=='COMPLETE' and read(BASE/'eval/JS_COMPLETE.json') and read(BASE/'eval/OWNER_STATUS.json',{}).get('status','').startswith('COMPLETE') and read(BASE/'js_web_compare_v1/COMPLETE.json'):
   notify('ALL_TRAIN_EVAL_TERMINAL','COMBAT 15run训练与固定CPU评估均终态；报告与原始证据已落盘，旧暂停队列未恢复。',summary);return
  time.sleep(60)
if __name__=='__main__':main()
