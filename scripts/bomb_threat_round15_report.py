"""Recompute round15 evidence and keep an honest, resumable execution receipt."""
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json, sha
from scripts import bomb_threat_round15 as R

BASE=Path('runs/bomb_threat_v1')
OUT=BASE/'iter2/a3sr/round15'
DIAG=OUT/'diagnostic'
REPORT=Path('reports/bomb_threat_v1_round15_20261030.md')


def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def native_audit():
    out=BASE/'eval/fresh256_a3sr_u640/native_hunter256_v1'
    done=read(out/'NATIVE_HUNTER_COMPLETE.json')
    candidate=read(BASE/'fresh256_a3sr_u640/PROTOCOL.json')['models']['candidate']
    assert done['policy_sha256']==candidate['policy_sha256']
    for f,h in done['files'].items():assert sha(out/f)==h
    episodes=read(out/'EPISODES.json')['episodes']
    trace=np.load(out/'TRACE.npz')
    arrays={k:trace[k] for k in ('events','seats','seeds','initial_pos','actions')}
    spawns=read(BASE/'fresh256_a3sr_u640/hard_spawns.json')
    assert len(episodes)==768
    per_mode={};zero_rows=[]
    for i,e in enumerate(episodes):
        index=e['seed']-spawns['seed'];seat=e['seat']
        assert seat==index%2==int(arrays['seats'][i])
        assert int(arrays['seeds'][i])==e['seed']
        assert np.array_equal(arrays['initial_pos'][i],np.asarray(spawns['spawn_cells'][index%64],np.float32)+.5)
        events=arrays['events'][:,i].astype(np.int32).sum(0)
        assert events[:,0].tolist()==e['placed']
        if e['placed'][seat]==0:
            actions=arrays['actions'][:min(100,e['ticks']),i,seat]
            zero_rows.append(dict(seed=e['seed'],mode=e['mode'],seat=seat,
                unilateral=e['placed'][1-seat]>0,geometry_id=index%64,
                first100_move_counts=np.bincount(actions[:,0],minlength=5).tolist(),
                first100_ability_counts=np.bincount(actions[:,1],minlength=3).tolist()))
    for mode in ('greedy','categorical','categorical_tau0.5'):
        per_mode[mode]={}
        for seat in (0,1):
            rows=[e for e in episodes if e['mode']==mode and e['seat']==seat]
            per_mode[mode][f'P{seat}']=dict(games=len(rows),
                raw_zero=sum(e['placed'][seat]==0 for e in rows),
                unilateral_zero=sum(e['placed'][seat]==0 and e['placed'][1-seat]>0 for e in rows),
                structural_zero=sum(e['placed']==[0,0] for e in rows))
    result=dict(status='PASS',initial_position_seat_mapping_exact=768,
        policy_sha256=done['policy_sha256'],per_mode=per_mode,raw_zero_episodes=zero_rows,
        native_complete_sha256=sha(out/'NATIVE_HUNTER_COMPLETE.json'),
        limitations=['Hunter implementation and PRNG differ between JAX and JS',
                    'initial position/seat match does not prove observation or per-tick physics equality'],unix=time.time())
    atomic_json(OUT/'NATIVE_SEAT_ZERO_AUDIT.json',result)
    return result


def aggregate(rows):
    return dict(games=len(rows),never_moved=sum(r['whole_game_never_moved'] for r in rows),
        draw_games=sum(r['expected']['winner'] in (None,-1) for r in rows),
        unique_cells_at_most4=sum(r['unique_cells']<=4 for r in rows),
        local_roaming=sum(r['spawn_local_roaming'] for r in rows),
        early_suppression=sum(r['early_suppression_observed'] for r in rows),
        death_games=sum(r['own_first_death_tick'] is not None for r in rows),
        invalid_spawn=sum(not r['initial']['spawn_walkable'] for r in rows),
        initially_brick_separated=sum(r['initial']['enemy_distance']<0 and r['initial']['enemy_distance_ignore_bricks']>=0 for r in rows),
        first100_single_cell=sum(r['first100_unique_cells']==1 for r in rows),
        pre_enemy_bomb_reversal_at_least5=sum(r['pre_enemy_bomb_opposite_action_flips']>=5 for r in rows),
        zero_placement_but_requested=sum(r['expected']['requests'][r['seat']]>0 for r in rows),
        total_requests=sum(r['expected']['requests'][r['seat']] for r in rows),
        first100_legal_ticks=sum(r['first100']['legal_bomb_ticks'] for r in rows),
        first100_mean_legal_bomb_probability=(sum((r['first100']['mean_legal_bomb_probability'] or 0)*r['first100']['legal_bomb_ticks'] for r in rows)/sum(r['first100']['legal_bomb_ticks'] for r in rows))
            if sum(r['first100']['legal_bomb_ticks'] for r in rows) else None,
        geometry_counts=dict(sorted(Counter(str(r['geometry_id']) for r in rows).items(),key=lambda x:int(x[0]))))


def geometry_comparison():
    cases={}
    for role in ('candidate','parent'):
        cases[role]=[r for p in sorted((BASE/f'fresh256_a3sr_u640/hard/{role}').glob('greedy_*.json.outcome.json')) for r in read(p)['episodes']]
        assert len(cases[role])==256
    cases['jax_candidate']=[r for r in read(BASE/'eval/fresh256_a3sr_u640/native_hunter256_v1/EPISODES.json')['episodes'] if r['mode']=='greedy']
    result=[]
    for g in range(64):
        item=dict(geometry_id=g,seat=g%2)
        for role,rows in cases.items():
            rr=[r for r in rows if (r['seed']-2026153900)%64==g]
            assert len(rr)==4
            item[role]=dict(games=4,unilateral_zero=sum(r['placed'][r['seat']]==0 and r['placed'][1-r['seat']]>0 for r in rr),
                raw_zero=sum(r['placed'][r['seat']]==0 for r in rr))
        result.append(item)
    atomic_json(OUT/'GEOMETRY_COMPARISON.json',dict(rows=result,
        scope='descriptive matched birth IDs only; engine/Hunter/PRNG and selected zero-case conditioning are confounded'))
    return result


def write_report():
    rows=[]
    rejected=[]
    for p in sorted(DIAG.glob('game_*/ACCEPTED.json')):
        proof=read(p)
        for f,h in proof['files'].items():assert sha(Path(f))==h
        r=read(p.parent/'METRICS.json')
        trace=read(p.parent/'TRACE.json')['rows']
        actions=trace[:100];moves=[a[3] for a in actions]
        r['first100_opposite_action_flips']=sum((a,b) in ((0,1),(1,0),(2,3),(3,2)) for a,b in zip(moves,moves[1:]))
        pre=[a[3] for a in trace if a[0]<min(100,r['opponent_first_bomb_tick'] or 100)]
        r['pre_enemy_bomb_opposite_action_flips']=sum((a,b) in ((0,1),(1,0),(2,3),(3,2)) for a,b in zip(pre,pre[1:]))
        for a in trace:
            if a[4]==1 and not a[7]:
                same_cell=tuple(int(np.floor(v)) for v in a[1])==tuple(int(np.floor(v)) for v in a[2])
                rejected.append(dict(seed=r['seed'],seat=r['seat'],geometry_id=r['geometry_id'],
                    pre_step_tick=a[0],own_position=a[1],enemy_position=a[2],
                    same_cell=same_cell,enemy_placed=bool(a[8]),pre_step_bomb_legal=bool(a[10]),
                    reason='P0 placed first in same cell' if r['seat']==1 and same_cell and a[8] else 'not identified from saved observer'))
        rows.append(r)
    atomic_json(OUT/'REJECTED_REQUEST_AUDIT.json',dict(events=rejected,
        placement_order='web/sim-stock-window-v1.js:1211-1226, P0 then P1; no rule change',
        source_sha256=sha(Path('web/sim-stock-window-v1.js'))))
    groups={name:aggregate(rr) for name,rr in {
        'fresh_hard_P1':[r for r in rows if r['group']=='fresh_hard' and r['seat']==1],
        'fresh_hard_P0':[r for r in rows if r['group']=='fresh_hard' and r['seat']==0],
        'holdout_normal_P1':[r for r in rows if r['group']=='holdout_normal' and r['seat']==1],
        'holdout_normal_P0':[r for r in rows if r['group']=='holdout_normal' and r['seat']==0]}.items()}
    native=read(OUT/'NATIVE_SEAT_ZERO_AUDIT.json') or native_audit()
    geometry=geometry_comparison()
    auth=read(OUT/'AUTHORIZATION.json',{})
    fresh=read(BASE/'fresh256_a3sr_u640/RESULTS.json',{})
    first=read(OUT/'SLOT07_FIRST_FORMAL.json',{})
    diag_complete=(DIAG/'COMPLETE.json').exists()
    state='FIRST_FORMAL_ACCEPTED' if first else 'WAITING_SLOT07_ATOMIC_HANDOFF' if auth.get('donor') else 'WAITING_ELIGIBLE_DONOR'
    summaries=dict(status=state,diagnostic_games=len(rows),diagnostic_complete=diag_complete,
        groups=groups,native=native['per_mode'],fresh_completed_games=fresh.get('completed_games'),
        fresh_collection_status=fresh.get('collection_status'),fresh_gate=fresh.get('status'),
        donor=auth.get('donor'),first_formal=first or None,unix=time.time())
    if diag_complete:
        summaries['diagnostic_wall_seconds']=read(DIAG/'COMPLETE.json')['unix']-read(DIAG/'LAUNCH.json')['unix']
    atomic_json(OUT/'RESULTS.json',summaries)
    slots=read(BASE/'SLOTS.json',{});live=[]
    for slot,e in slots.items():
        p=BASE/'formal'/e['run']/'updates.jsonl'
        if p.exists():
            r=json.loads(p.read_text().splitlines()[-1]);live.append(dict(slot=slot,run=e['run'],update=r['new_update'],
                finite=r['finite'],update_l2=r['update_l2'],env_per_second=262144/r['cycle_seconds'],
                state='ATOMIC_COMPLETE' if (p.parent/'COMPLETE.json').exists() else 'UPDATING'))
    summaries['training_snapshot']=live
    atomic_json(OUT/'EXECUTION_STATUS.json',summaries)
    text=['# BOMB-THREAT round15 — 20261030','',
        f"执行状态：`{state}`；诊断 {len(rows)}/58 局已逐局匹配，fresh {fresh.get('completed_games')}/3840、{fresh.get('collection_status')}，B={fresh.get('status')}。未晋升、未发布、未改奖励或解码。",'',
        '## 授权与 slot07 接管','',
        '采用 06:15 更正：slot07 现行 c27 u1400–1520 dev 胜负差 +9/+8/+14/+15，单方零泡17–20（主要P0），泡15–21/局。06:10引用旧chunk的u1800–1920/+6/+11/+2/+4及8.5泡不再作为依据。替换决定不变。', '',
        'donor 必须为slot05当前链完整state，dev胜负差至少+25、单方零泡至多4；先pin再验分数/SHA。u1000为31/31/2（+29、零泡1）、u1040为29/34/1（+28、零泡1），但完整state已删除，不能充当可恢复donor。u1080为25/35/4（+21、零泡5），明确不合格。后续cadence预先pin并逐点审计，不降低门槛。', '',
        f"冻结授权 `{OUT}/AUTHORIZATION.json`；准入 `{OUT}/DONOR_ADMISSION.json`。",'',
        '调度范围仅slot07，旧线当前chunk完整提交后接管；旧线末state预先pin。复制Actor/Critic/Adam/env/aux/ring/计数器，先核验donor next-step parity，再独立slot07 RNG rekey；无Critic预热、无奖励账本重置。新chunk保持已有320更新长度，完成后等待明确扩展；预算解释见BUDGET_AMENDMENT01。slot05仍按原1280预算，其他slot不动。', '']
    admissions=[read(p) for p in sorted((OUT/'donor_admissions').glob('u*.json'), reverse=True)]
    if not admissions:admissions=read(OUT/'DONOR_ADMISSION.json',{}).get('candidates',[])
    if admissions:
        text += ['| 完整候选update | 胜/平/负 | 胜负差 | 单方零泡 | 准入 |',
                 '|---:|---|---:|---:|---|']
        for e in admissions:
            text.append(f"|{Path(e['checkpoint']).name[1:]}|{e['win']}/{e['draw']}/{e['loss']}|{e['win_minus_loss']:+d}|{e['unilateral_zero']}|{e['eligible']}|")
        text.append('')
    if auth.get('donor'):
        d=auth['donor'];text += [f"实际donor：`{d['checkpoint']}`，**u{d['update']}，不是u640**；dev `{auth['donor_dev']}`；state SHA `{d['state_sha256']}`，policy SHA `{d['policy_sha256']}`。",'']
    if first:
        text += [f"首恢复后正式更新：`{first['run']}` u{first['first_post_restore_update']}，finite；Actor/Critic L2 `{first['update_l2']}`，{first['env_per_second']:.1f} env/s（首步含冷启动，非稳态）；smoke/独立fresh恢复及完整source parity通过。",'']
    text += ['## 完整局被动诊断','',
        f"冻结清单 `{DIAG}/MANIFEST.json`：fresh hard 45局P1+10局P0，holdout normal3局P1；全部是同候选u640、teacher off、生产stock-window-v1 argmax。重放不是新增独立样本；逐局终局、泡/请求、击杀/死亡、时长/seed/seat严格匹配。",'',
        '动作编码：移动[上,下,左,右,等待]；ability[不释放,放泡,道具]。局部徘徊仅按整局最大出生曼哈顿位移≤2定义，不等于卡死；early suppression仅指前100tick死亡/原生糖浆事件，不从敌首泡推断压制。出生连通BFS把可行走crate当作可通，不复用旧错误障碍口径。', '',
        '| 组 | 已核验局 | 从未移动 | 出生附近徘徊 | 前100受控/死亡 | 死亡局 | 初始砖隔开 | 首100仅1格 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,g in groups.items():text.append(f"|{name}|{g['games']}|{g['never_moved']}|{g['local_roaming']}|{g['early_suppression']}|{g['death_games']}|{g['initially_brick_separated']}|{g['first100_single_cell']}|")
    text += ['', '几何池只有64个，fresh每个几何循环4次；geometry id奇偶固定seat，因此几何与座位混杂，不能将P1差异独立归因给座位实现。holdout使用normal Hunter而fresh为hard，不能当仅seed变化的对照。出生位置/seat在JS逐局assert，JAX768例初态位置及seat与清单完全一致；未见索引反转证据，但不能据此排除所有观测/物理差异。', '',
        '这是以零泡为条件选取的诊断集合，不能据组内均值估计所有P1/P0局的策略概率差。逐局可观测事实用于区分硬卡死、局部振荡与早期受控；威胁躲避的意图、几何的因果作用仍需反事实，当前不修改训练。全部原始逐tick位置、动作、mask/概率在各game的TRACE.json，可独立复算。', '']
    if diag_complete:
        p1=groups['fresh_hard_P1'];p0=groups['fresh_hard_P0']
        text += ['判别：', '',
            f"- **整局硬卡死证据不足**：P1从未移动{p1['never_moved']}/45，出生不可行走{p1['invalid_spawn']}/45；局部徘徊{p1['local_roaming']}/45，首100仅1格{p1['first100_single_cell']}/45，整局至多访问4格{p1['unique_cells_at_most4']}/45（P0 为{p0['unique_cells_at_most4']}/10）。P1及P0这些零泡局全部平局。局部振荡和整体不动必须分开。",
            f"- **前100被击杀/包裹压制不是主证据**：P1早期受控/死亡{p1['early_suppression']}/45，整局发生死亡{p1['death_games']}/45；敌首泡前已有至少5次反向动作切换{p1['pre_enemy_bomb_reversal_at_least5']}/45。不能据此排除后续威胁躲避，但不把敌首泡事件等同实际压制。",
            f"- **出生分区与低放泡选择同时存在**：P1初始砖隔开{p1['initially_brick_separated']}/45，P0 为{p0['initially_brick_separated']}/10；P1前100合法放泡分母{p1['first100_legal_ticks']}tick，条件均值概率{p1['first100_mean_legal_bomb_probability']:.6f}。合法机会并非全部被mask掉；其中{p1['zero_placement_but_requested']}/45局确实请求过放泡、共{p1['total_requests']}次但未实际放置，须与从不请求分开，不把所有零泡等同argmax始终不选放泡。",
            '- **未查到初始化座位索引反转**：JS逐局出生/seat和JAX768例出生/seat均匹配。P1与几何ID奇偶绑定、两端Hunter/随机流不同，因此本次不足以证明或排除更深层JAX/JS不一致；不声称找到唯一根因。', '']
        text += ['4次拒绝均为几何id23、pre-step tick4（step后tick5），P1与P0同时位于中心格(2,12)，P0对手同tick实际放泡，P1预步mask允许但未落泡。生产JS按P0→P1逐人放置，后者遇到前者新泡而拒绝；这是明确的同格放置顺序事件，不足以解释其余41局，亦未证明JAX对此处理不同。证据REJECTED_REQUEST_AUDIT.json及原始TRACE，不改训练规则。', '']
    text += ['相同几何id的完整4局计数（全部64项见GEOMETRY_COMPARISON.json；此处列JS候选有单方零泡的几何）：', '',
        '|几何id/seat|JS候选单方零泡/4|JS父点单方零泡/4|JAX候选单方零泡/4|',
        '|---|---:|---:|---:|']
    for e in geometry:
        if e['candidate']['unilateral_zero']:
            text.append(f"|{e['geometry_id']}/P{e['seat']}|{e['candidate']['unilateral_zero']}|{e['parent']['unilateral_zero']}|{e['jax_candidate']['unilateral_zero']}|")
    text += ['',
        '## GPU2 辅助对照','',
        '| 模式 | P0 raw/单方/结构零泡 | P1 raw/单方/结构零泡 | 每seat局数 |',
        '|---|---|---|---:|']
    for mode,ss in native['per_mode'].items():
        a,b=ss['P0'],ss['P1'];text.append(f"|{mode}|{a['raw_zero']}/{a['unilateral_zero']}/{a['structural_zero']}|{b['raw_zero']}/{b['unilateral_zero']}/{b['structural_zero']}|128|")
    text += ['', '生产JS hard argmax：P0单方10/128（raw26、结构16），P1单方45/128（raw45、结构0）。JAX greedy P1单方3/128而raw10/128，禁止把raw10和JS单方45混口径；Hunter实现与PRNG均不同，不是engine-only对照。原生768例初态seat/出生及事件总数复算通过，见NATIVE_SEAT_ZERO_AUDIT.json。', '',
        '## 验证与当前训练','',
        'focused：`.venv/bin/python -m pytest -q tests/test_bomb_threat_round15.py tests/test_bomb_threat_round14.py tests/test_bomb_threat_a3sr.py tests/test_bomb_threat_seat5050_schedule.py tests/test_checkpoint_retention.py`，29通过；`node --check scripts/bomb_threat_round15_record.js`通过。诊断两路CPU墙钟591.76秒，完整58局；仅owner reload，7个原worker全部收养，训练worker信号0。round14冻结源码/worker不改。', '',
        '表中env/s为最后更新历史吞吐；ATOMIC_COMPLETE不是当前仍计算。slot05固定1280预算已结束、GPU5已释放，不擅自续训。', '',
        '| slot | run | update | 最后更新env/s | finite | 状态 |','|---|---|---:|---:|---|---|']
    for e in live:text.append(f"|{e['slot']}|{e['run']}|{e['update']}|{e['env_per_second']:.1f}|{e['finite']}|{e['state']}|")
    text += ['', '## 逐局明细','',
        '时间均为Sim tick；“无”为完整局未发生该事件。首100动作向量按上述编码，非全局总数倒推。', '',
        '|组/seat|seed|几何id|敌首泡|己首死|从未动/局部徘徊|首100移动向量|首100ability向量|首100格数/反向切换|',
        '|---|---:|---:|---:|---:|---|---|---|---|']
    for r in rows:
        text.append(f"|{r['group']}/P{r['seat']}|{r['seed']}|{r['geometry_id']}|{r['opponent_first_bomb_tick'] if r['opponent_first_bomb_tick'] is not None else '无'}|{r['own_first_death_tick'] if r['own_first_death_tick'] is not None else '无'}|{r['whole_game_never_moved']}/{r['spawn_local_roaming']}|{r['first100']['move_counts']}|{r['first100']['ability_counts']}|{r['first100_unique_cells']}/{r['first100_opposite_action_flips']}|")
    REPORT.write_text('\n'.join(text)+'\n')
    return summaries


if __name__=='__main__':
    native_audit()
    print(json.dumps(write_report(),ensure_ascii=False))
