"""Persist actual warm full-update throughput after the exact hot switch."""
import json
import statistics
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

BASE=Path('runs/bun_native_v2_20261030');OUT=BASE/'custody_performance_v1'


def rows(run):
    return [json.loads(l) for l in (BASE/'formal'/run/'updates.jsonl').read_text().splitlines()]


def notify(key,text):
    f=OUT/'notifications'/f'{key}.json'
    if f.exists():return
    atomic_json(f,dict(message=text,unix=time.time()))
    with (OUT/'NOTIFY.log').open('ab') as log:subprocess.run(['jdme-notify','send',text],stdout=log,stderr=subprocess.STDOUT,check=False)


def refresh():
    slots=json.loads((BASE/'R2_SLOTS.json').read_text());switch=json.loads((OUT/'ATOMIC_SWITCH_READY.json').read_text())
    compared={};restores={}
    a=slots['a'];ar=[r for r in rows(a['run']) if r['new_update']>switch['states']['a']['checkpoint']['update']][2:][-5:]
    if len(ar)<5:ar=rows(a['run'])[-5:]
    abase=statistics.median(262144/r['cycle_seconds'] for r in ar)
    for arm in ('b','c'):
        entry=slots[arm];folder=BASE/'formal'/entry['run'];restore=folder/'PERFORMANCE_RESTORE.json'
        if not restore.exists():continue
        r=json.loads(restore.read_text());assert all(r.get(k) for k in ('passed','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
        restores[arm]=dict(receipt=str(restore),sha256=sha(restore),atomic_source=switch['states'][arm]['checkpoint'])
        history=rows(entry['run']);at=switch['states'][arm]['checkpoint']['update']
        before=[r for r in history if r['new_update']<=at][-5:]
        after=[r for r in history if r['new_update']>at]
        # First resumed update may compile the large rollout and optimizer.
        warm=after[2:][-5:]
        info=dict(before_points=[r['new_update'] for r in before],after_points=[r['new_update'] for r in warm],
            cold_updates=[dict(update=r['new_update'],cycle_seconds=r['cycle_seconds'],rollout_seconds=r['rollout_seconds']) for r in after[:2]],
            before_env_per_second=statistics.median(262144/r['cycle_seconds'] for r in before),
            resumed_updates=len(after),warm_points=len(warm),target_ratio=.70)
        if warm:
            info.update(after_env_per_second=statistics.median(262144/r['cycle_seconds'] for r in warm),
                cycle_seconds=statistics.median(r['cycle_seconds'] for r in warm),
                rollout_seconds=statistics.median(r['rollout_seconds'] for r in warm),
                finite=all(r['finite'] and min(r['update_l2'].values())>0 for r in warm))
            info['fraction_of_R2a']=info['after_env_per_second']/abase
            info['full_update_speedup']=info['after_env_per_second']/info['before_env_per_second']
        info['target_accepted']=len(warm)>=5 and info.get('finite') and info['fraction_of_R2a']>=.70
        compared[arm]=info
    status='PASS' if len(compared)==2 and all(r['target_accepted'] for r in compared.values()) else 'WARMUP_OR_BELOW_TARGET'
    result=dict(status=status,full_restore=restores,arms=compared,R2a_env_per_second=abase,
        no_reward_changes=True,no_discarded_updates=True,no_repeated_smoke=True,source='actual formal update logs',
        shared_gpu_confound='b/c retain shared R1 GPU4/7; a has GPU3; ratio is observed scheduling performance, not isolated hardware causality',unix=time.time())
    atomic_json(OUT/'FULL_UPDATE_ACCEPTANCE.json',result)
    if status=='PASS':
        notify('WARM_TARGET_PASS','custody Φ等价优化完成：2048环境×32tick逐位一致、b/c完整恢复PASS；暖态整更新吞吐 b %.0f、c %.0f env/s，分别为R2-a %.0f%%/%.0f%%，达到70%%目标。原奖励/预算不变、无更新丢失。'%(compared['b']['after_env_per_second'],compared['c']['after_env_per_second'],100*compared['b']['fraction_of_R2a'],100*compared['c']['fraction_of_R2a']))
    report(result)
    return result


def report(r):
    lines=['# custody Φ 等价计算优化（20261030）','',
        '旧实现 dirty.any() 触发2048环境×4图完整BFS，且笼距随移动目标重算。新实现将笼拓扑与散包/己源携带目标分开缓存，压缩实际dirty环境后64行分块精确BFS；cache ABI及奖励数值不变。现有route_distance有可穿砖权重，不能直接替换硬障碍距离。',
        '', 'GPU原生固定动作32tick×2048环境：Φ、缓存、p/q与归属布尔逐位一致。微核总时7.67倍（仅该样本、共用评估GPU2，非整段训练加速承诺）。全部输入/SHA/时序见 `runs/bun_native_v2_20261030/custody_performance_v1/PARITY_PROFILE.json`。',
        '', '旧worker只有namespace暂停机制，因此a也短暂原子保存一次后原配置恢复，R1始终继续。a u3031、b u2762、c u2769完整state/Adam/env/aux/RNG/ring已保存；零更新丢失，不重复smoke，b/c先独立next-step/Adam/env/aux/RNG/ring验收再同run续训。新协议为独立performance01元数据，原协议保留，reward/network/opponent/batch/LR/budget均未修改。',
        '', '当前验收：`'+r['status']+'`；R2-a实测参考 %.0f env/s。首两更新作为冷编译点分列，后续至少5暖态点再判断≥70%%。'%r['R2a_env_per_second'],
        '', '|arm|暖态点|旧env/s|新env/s|cycle/rollout秒|相对a|目标|','|---|---|---:|---:|---|---|---|']
    for k,v in r['arms'].items():lines.append('|%s|%s|%.0f|%s|%s|%s|%s|'%(k,v['after_points'],v['before_env_per_second'],v.get('after_env_per_second','待测'),
        str((v.get('cycle_seconds'),v.get('rollout_seconds'))),str(v.get('fraction_of_R2a')),v['target_accepted']))
    lines+=['','限制：b/c仍分别与R1共GPU4/7，a用GPU3；不能把ratio当成排除了资源混杂的因果效应。当前未达足量暖态点时不得标达标。完整收据 `FULL_UPDATE_ACCEPTANCE.json`；不发布、不变网页、未恢复旧队列。']
    Path('reports/custody_performance_20261030.md').write_text('\n'.join(lines)+'\n')


def main():
    guard=lock(OUT/'MONITOR.lock');started=time.time()
    while True:
        result=refresh()
        if result['status']=='PASS':return
        if time.time()-started>1200:
            notify('TARGET_NOT_YET_MET','custody等价切换已实施，但20分钟内未取得b/c各5暖态点并达到a的70%；完整实测已落盘，健康训练继续，需要进一步profile，不伪称达标。')
        time.sleep(15)


if __name__=='__main__':main()
