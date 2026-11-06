"""Persist live task progress until fixed training/evaluation budgets complete."""
import json
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

B=Path('runs/combat_regions_independent_20261104')


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def notify(key,message):
    path=B/'notifications'/(key+'.json')
    if path.exists():return
    result=subprocess.run(['jdme-notify','send',message],capture_output=True,text=True,timeout=20)
    atomic_json(path,dict(message=message,returncode=result.returncode,unix=time.time()))


def main():
    guard=lock(B/'COMPLETION_OWNER.lock')
    current=read(B/'training/CURRENT.json');p=read(current['protocol'])
    out=Path(p['namespace'])/current['run'];start=p['campaign_start_update'];end=start+1280
    while True:
        progress=read(out/'PROGRESS.json',{});latest=progress.get('iterations',start)
        first=read(B/'FIRST_FORMAL_UPDATE.json');steady=read(B/'STEADY_UPDATES.json')
        training=read(out/'COMPLETE.json');evaluation=read(B/'evaluation/COMPLETE.json')
        evalstatus=read(B/'evaluation/STATUS.json',{});failure=read(B/'evaluation/FAILURE.json')
        owner=read(B/'training/OWNER_STATUS.json',{})
        state='RUNNING_1280_FIRST_FORMAL_ACCEPTED' if first else 'STARTUP_VALIDATION'
        if training:state='TRAINING_1280_COMPLETE_EVALUATION_PENDING'
        if failure:state='TRAINING_CONTINUES_EVALUATION_TECHNICAL_BLOCKED'
        if owner.get('status')=='TECHNICAL_BLOCKED':state='TRAINING_TECHNICAL_BLOCKED'
        value=dict(status=state,updated_at_unix=time.time(),source_update=start,target_update=end,
            latest_counted_update=latest,completed_new_updates=latest-start,remaining_new_updates=max(0,end-latest),
            smoke_updates_counted=2,full_restore_diagnostic_steps_not_counted=True,lost_source_updates=0,
            implementation_and_CPU='PASS',first_formal_receipt=str(B/'FIRST_FORMAL_UPDATE.json') if first else None,
            steady_receipt=str(B/'STEADY_UPDATES.json') if steady else None,
            live_training_progress=str(out/'PROGRESS.json'),live_evaluation_status=str(B/'evaluation/STATUS.json'),
            accepted_evaluation_shards=evalstatus.get('accepted_shards',0),
            evaluation_failure=failure,capability_improvement=read(B/'evaluation/SUMMARY.json',{}).get('capability_improvement','NOT_ESTABLISHED'))
        if first and steady:
            notify('first_formal',f"多区域连续进攻独立训练首验通过：源u{start}安全交接，正式u{first['update']} Actor/Critic/ability非零finite，GPU fresh全PPO精确恢复；5轮稳态中位{steady['median_env_per_second']:.0f}env/s。1280更新及固定区域评估继续，尚未宣称能力改善。")
        if training:
            assert training['update']==end
            assert sha(Path(training['checkpoint'])/'state.pkl')==training['state_sha256']
            final=B/'TRAINING_1280_COMPLETE.json'
            if not final.exists():
                rows=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
                counted=[row['new_update'] for row in rows]
                assert counted==list(range(start+1,end+1)), 'missing or duplicated counted PPO updates'
                assert all(row['finite'] for row in rows)
                atomic_json(final,dict(status='PASS_1280_REAL_UPDATES',source_update=start,end_update=end,
                    updates=1280,all_finite=True,all_updates_contiguous=True,checkpoint=training['checkpoint'],
                    state_sha256=training['state_sha256'],capability='see frozen evaluation; training completion is not capability proof'))
                notify('training_complete',f"多区域连续进攻独立阶段1280次真实更新完成：u{start}→u{end}，更新连续且finite，完整终点已存盘；区域/自然/运包配对评估继续汇总。")
        if training and evaluation and not failure:
            subprocess.run([sys.executable,'-m','scripts.combat_regions_report_v2'],check=True)
            summary=read(B/'evaluation/SUMMARY.json')
            value.update(status='COMPLETE_FIXED_TRAINING_AND_EVALUATION',capability_improvement=summary['capability_improvement'])
            atomic_json(B/'FINAL_ACCEPTANCE.json',dict(status=value['status'],training=str(B/'TRAINING_1280_COMPLETE.json'),
                evaluation=str(B/'evaluation/COMPLETE.json'),summary=str(B/'evaluation/SUMMARY.json'),
                capability_improvement=summary['capability_improvement'],comparisons=summary['comparisons'],unix=time.time()))
            notify('evaluation_complete',f"多区域连续进攻固定评估完成：{summary['capability_improvement']}。最弱区域、右下、自然和同链交付检查及配对置信区间已写入日期报告与FINAL_ACCEPTANCE；不以泡量或loss替代能力。")
        atomic_json(B/'PROGRESS.json',value);atomic_json('.hermes/PROGRESS.json',value)
        if value['status']=='COMPLETE_FIXED_TRAINING_AND_EVALUATION':return
        if owner.get('status')=='TECHNICAL_BLOCKED' or failure:
            notify('technical_blocked','多区域独立任务出现技术阻塞，已保存PROGRESS和日志；保留完整checkpoint，未停止其他有效训练。')
        time.sleep(15)


if __name__=='__main__':main()
