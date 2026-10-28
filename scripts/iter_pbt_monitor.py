"""Passive, dated report and stale-progress alerts; no GPU launch authority."""
import json,time
from pathlib import Path
from scripts.population_run_state import lock,atomic_json
from scripts.iter_pbt_owner import read,notify
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/iter_pbt_v1'
def main():
 guard=lock(BASE/'MONITOR.lock')
 while not any((BASE/n).exists() for n in ('USER_STOP.json','USER_PAUSE.json')):
  now=time.time();s=read(BASE/'STATUS.json',{});rows=[];health={};lags=read(BASE/'EVAL_LAG.json',{})
  for g,v in s.get('slots',{}).items():
   name=v['run'];out=BASE/'formal'/name;launch=read(out/'LAUNCH.json',{});p=read(out/'PROGRESS.json',{}).get('last',{})
   age=now-p.get('unix',launch.get('unix',now));stage=launch.get('stage','waiting_previous_candidate')
   health[g]=dict(run=name,stage=stage,update=p.get('new_update'),progress_age_seconds=age,finite=p.get('finite'))
   rate=262144/p['cycle_seconds'] if p.get('cycle_seconds') else None
   lag=lags.get(g,{})
   health[g]['evaluation_lag']=lag
   rows.append(f"| {g} | {name} | {stage} | {p.get('new_update','—')} | {round(rate) if rate else '—'} | {p.get('finite','—')} | {lag.get('latest_saved','—')} | {lag.get('latest_scored','—')} | {lag.get('lag_chunks','—')} |")
   # A compiler running is neither proof of effective training nor an idle card.
   if launch and age>600 and name in s.get('active',[]):
    notify(name+'_STALE_'+stage,f'ITER-PBT进度超过10分钟未增长：{name}，stage={stage}，最近更新={p.get("new_update")}；需要检查编译/阻塞，未把进程存在当训练。',evidence=health[g])
  atomic_json(BASE/'HEALTH.json',dict(owner_status_age=now-s.get('updated_unix',now),slots=health,unix=now))
  execution=read(BASE/'EXECUTION_STATUS.json',{})
  execution['latest_updates']={v['run']:v['progress']['last'] for v in s.get('progress',{}).values() if v.get('progress')}
  execution.update(status='CONTINUOUS_PBT_RUNNING' if now-s.get('updated_unix',now)<120 else 'OWNER_HEARTBEAT_STALE',active_PBT=s.get('active',[]),live_status_source='runs/iter_pbt_v1/STATUS.json',progress=s.get('progress',{}),work_totals=s.get('work_totals',{}),gpu_samples=s.get('gpu_samples',[]),unix=now)
  atomic_json(BASE/'EXECUTION_STATUS.json',execution)
  report=ROOT/'reports/iter_pbt_experiment_20261028.md'
  if report.exists():
   text=report.read_text().split('<!-- ITER_PBT_AUTO -->')[0]
   text+='<!-- ITER_PBT_AUTO -->\n\n## 持久监控最新快照\n\n| slot/GPU | run | 阶段 | update | 环境步/秒 | finite | 最新保存 | 已评分 | 滞后chunk |\n|---|---|---|---:|---:|---|---|---|---:|\n'+'\n'.join(rows)+'\n\n实际新增工作量见 `runs/iter_pbt_v1/WORK_TOTALS.json`；谱系计数不混入本slot新增步数。连续利用率/功率见 `GPU_SAMPLES.jsonl`，正式首更新与恢复SHA见 `GPU_RESUME_ACCEPTANCE.json`。\n'
   native=read(BASE/'native_gpu/STATUS.json',{})
   if native:
    text+='\n共卡原生评估：启用='+str(native.get('enabled'))+'，并发上限='+str(native.get('max_concurrent'))+'，活跃='+str(len(native.get('active',{})))+'；分段耗时/显存/训练吞吐见 `native_gpu/SAMPLES.jsonl` 与各点评估 `native/NATIVE_SUMMARY.json`。\n'
   best=read(BASE/'BEST.json')
   text+='\n'+('尚无通过保留确认的新最佳。' if not best else '最新保留确认：`'+best['checkpoint']+'`，dev分='+str(best['dev_score'])+'，保留分='+str(best['holdout_score']))+'\n'
   report.write_text(text)
  time.sleep(60)
if __name__=='__main__':main()
