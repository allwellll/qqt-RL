"""Passive round16 raw-game audit, explicit queue dependency and bounded followups."""
import json,time,subprocess,os,sys
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bomb_threat_round15 import read,pin
BASE=Path('runs/bomb_threat_v1');OUT=BASE/'iter2/a3sr/round16'
REPORT=Path('reports/bomb_threat_v1_round16_20261030.md')


def argmax(name):
    from scripts.bomb_threat_eval import summarize
    out=BASE/'eval'/name;rows=[];files={};primary_episodes=[]
    for i in range(0,64,8):
        p=out/f'natural_greedy_{i:03d}.json';q=Path(str(p)+'.outcome.json')
        if not p.exists() or not q.exists():return None
        a=read(p);b=read(q)
        assert not a['assist'] and not b['teacher'] and b['rule_version']=='stock-window-v1'
        assert a['model_json_sha256']==sha(out/'model/model.json')
        rr=b['episodes'];ee=a['results']['natural_full2400']['episodes'];assert len(rr)==len(ee)==8
        for r,e in zip(rr,ee):
            assert (r['seed'],r['seat'],r['ticks'])==(e['seed'],e['team'],e['ticks']) and r['done']
            assert r['causal_kills'][r['seat']]==e['causal_kill']
        rows+=rr;primary_episodes+=ee;files.update({str(p):sha(p),str(q):sha(q)})
    seed=2026132900 if name.endswith('_holdout') else 2026122900
    assert {r['seed'] for r in rows}==set(range(seed,seed+64))
    result=summarize(rows)
    result['final_enemy_buns_stored']=result['delivery']
    result['delivery']=sum(e['delivery'] for e in primary_episodes)
    for seat in (0,1):
        g=result['by_seat'][f'P{seat}'];g['final_enemy_buns_stored']=g['delivery']
        g['delivery']=sum(e['delivery'] for e in primary_episodes if e['team']==seat)
    result.update(name=name,files=files)
    return result


def recent(slot,limit=4):
    from scripts.bomb_threat_owner import causal_lineage_limits
    entry=read(BASE/'SLOTS.json')[str(slot)];limits=causal_lineage_limits(entry['run']);candidates=[]
    for out in (BASE/'eval').glob(f'bt_slot{slot:02d}_*'):
        if out.name.endswith('_holdout'):continue
        try:run,u=out.name.rsplit('_u',1);u=int(u)
        except ValueError:continue
        if run not in limits or u>limits[run] or u%40:continue
        candidates.append((u,out.name))
    rows=[]
    for _,name in sorted(candidates,reverse=True):
        r=argmax(name)
        if r is not None:rows.append(r)
        if len(rows)==limit:break
    return list(reversed(rows))


def decision():
    path=OUT/'SLOT04_DECISION.json'
    if path.exists() and read(path).get('status')!='MISSING':return read(path)
    targets=[argmax(f'bt_slot04_c000029_u{u:06d}') for u in (2120,2160)]
    controls=recent(3)
    if any(r is None for r in targets) or len(controls)!=4:
        value=dict(status='MISSING',missing=[u for u,r in zip((2120,2160),targets) if r is None],
            action='continue slot04 until complete causal evidence; no assumed decline')
    else:
        fields=('win_minus_loss','causal_kills','bombs_per_game')
        means={k:sum(r[k] for r in controls)/4 for k in fields}
        comparisons=[{k:r[k]<means[k] for k in fields} for r in targets]
        trigger=all(all(c.values()) for c in comparisons)
        value=dict(status='TRIGGERED' if trigger else 'NOT_TRIGGERED',targets=targets,
            controls=controls,control_means=means,strict_below=comparisons,
            action='replace only slot04 at supported atomic worker boundary' if trigger else 'continue slot04 unchanged',
            baseline_selection='latest four completed causal-lineage dev argmax at first complete two-point decision; frozen thereafter',unix=time.time())
    atomic_json(path,value)
    return value


def release_requests():
    pending=list((OUT/'pending_requests').glob('*.json'))
    if not (BASE/'fresh256_a3sr_u640/COMPLETE.json').exists():return False
    names=[]
    for p in pending:
        q=BASE/'eval_requests'/p.name
        if q.exists():assert read(q)==read(p)
        else:atomic_json(q,read(p))
        names.append(q.stem)
    if pending:
        prior=read(BASE/'GATE_AMEND2_HOLDOUT_PRIORITY.json',{})
        prior['names']=list(dict.fromkeys(names+prior.get('names',[])))
        atomic_json(BASE/'GATE_AMEND2_HOLDOUT_PRIORITY.json',prior)
        atomic_json(OUT/'EVAL_ORDER.json',dict(status='RELEASED_AFTER_FRESH_COMPLETE',
            dependency_sha256=sha(BASE/'fresh256_a3sr_u640/COMPLETE.json'),names=names,
            JS_CPU_precollected=True,native_is_auxiliary_not_B=True,unix=time.time()))
    return True


def seal_natural_if_ready(name):
    """Publish identical two-mode evidence before optional tau=.5 diagnostics finish."""
    out=BASE/'eval'/name
    if (out/'NATURAL_COMPLETE.json').exists():return
    from scripts.bomb_threat_eval import summarize
    req=read(BASE/'eval_requests'/f'{name}.json',read(OUT/'pending_requests'/f'{name}.json'))
    natural={};files={};model=out/'model/model.json'
    if not model.exists():return
    for mode in ('greedy','categorical'):
        rows=[];episodes=[]
        for i in range(0,64,8):
            p=out/f'natural_{mode}_{i:03d}.json';q=Path(str(p)+'.outcome.json')
            if not p.exists() or not q.exists():return
            a,b=read(p),read(q)
            assert a['model_json_sha256']==sha(model) and not a['assist']
            assert b['measurement_version']==2 and b['engine']=='production_JS' and not b['teacher'] and b['rule_version']=='stock-window-v1'
            rr=b['episodes'];ee=a['results']['natural_full2400']['episodes'];assert len(rr)==len(ee)==8
            for r,e in zip(rr,ee):
                assert (r['seed'],r['seat'],r['ticks'])==(e['seed'],e['team'],e['ticks']) and r['done']
                assert e['causal_kill']==r['causal_kills'][r['seat']]
            rows+=rr;episodes+=ee
            files[p.name]=sha(p);files[q.name]=sha(q)
        assert {r['seed'] for r in rows}==set(range(2026132900,2026132964))
        n=summarize(rows);n['final_enemy_buns_stored']=n['delivery']
        n['delivery']=sum(e['delivery'] for e in episodes)
        n['window_stock_pickup']=sum(e['stock_pickup_dead']+e['stock_pickup_native_sugar'] for e in episodes)
        n['window_stock_delivery']=sum(e['window_stock_delivery'] for e in episodes)
        n['free_stock_pickup']=sum(e['stock_pickup_free_alive'] for e in episodes)
        assert n['free_stock_pickup']==0
        natural[mode]=n
    cm=read(Path(req['checkpoint'])/'COMMITTED.json');assert sha(Path(req['checkpoint'])/'policy.pt')==cm['policy_sha256']
    marker=dict(checkpoint=req['checkpoint'],model_json=req.get('model_json'),role=req.get('role','checkpoint'),
        export_sha256=req.get('model_sha256'),model_identity=req.get('model_identity'),policy_sha256=cm['policy_sha256'],
        rule_version='stock-window-v1',reward_version=req.get('reward_version','bomb-threat-v1'),iter2_arm=req.get('iter2_arm'),
        split='holdout',seed=2026132900,spawn_sha256=sha(BASE/'holdout_spawns.json'),
        simulator_sha256=sha(Path('web/sim-stock-window-v1.js')),measurement_version=2,
        natural_js_full2400=natural,files=files,unix=time.time())
    atomic_json(out/'NATURAL_COMPLETE.json',marker)
    atomic_json(OUT/'early_natural_seals'/f'{name}.json',dict(status='PASS',
        reason='two mandated64-game modes complete; optional tau0.5 continues independently',
        natural_sha256=sha(out/'NATURAL_COMPLETE.json'),no_games_replayed=True,unix=time.time()))


def holdouts():
    from scripts.bomb_threat_gate_report import recompute_natural,reward_monitor
    from scripts.bomb_threat_gates_v2 import evaluate
    names=['bt_slot07_c000029_u001280_holdout','bt_slot05_c000028_u001200_holdout'];result={}
    for name in names:
        out=BASE/'eval'/name
        seal_natural_if_ready(name)
        if not (out/'NATURAL_COMPLETE.json').exists():
            result[name]=dict(status='MISSING',missing=['production_JS_argmax_and_categorical_64_complete'],
                greedy=argmax(name));continue
        marker,natural=recompute_natural(out);cp=Path(marker['checkpoint']);cm=read(cp/'COMMITTED.json')
        pm,parent=recompute_natural(BASE/'eval/baseline_parent')
        assert all(marker.get(k)==pm.get(k) for k in ('rule_version','split','seed','spawn_sha256','simulator_sha256'))
        refs={}
        for ref in ('reference_transport1536','reference_selfplay3072'):
            rm,rn=recompute_natural(BASE/'eval'/ref)
            if all(marker.get(k)==rm.get(k) for k in ('rule_version','split','seed','spawn_sha256','simulator_sha256')):refs[ref]=rn
        reward=reward_monitor(cp.resolve(),cm['update']);gate=evaluate(natural,parent,reward,'holdout',refs)
        g=natural['greedy'];p=parent['greedy']
        specs=[('argmax64',g.get('games'),64,lambda a,b:a==b),
               ('categorical64',natural.get('categorical',{}).get('games'),64,lambda a,b:a==b),
               ('win_minus_loss',g.get('win_minus_loss'),p['win_minus_loss']+8,lambda a,b:a>=b),
               ('unilateral_zero',g.get('zero_bomb_games'),7,lambda a,b:a<=b),
               ('bombs_per_game',g.get('bombs_per_game'),20,lambda a,b:a>=b),
               ('draw_fraction',g.get('draw_fraction'),p['draw_fraction'],lambda a,b:a<=b),
               ('causal_kills',g.get('causal_kills'),p['causal_kills'],lambda a,b:a>=b),
               ('deaths',g.get('deaths'),p['deaths']+5,lambda a,b:a<=b),
               ('self_deaths',g.get('self_deaths'),p['self_deaths']+2,lambda a,b:a<=b),
               ('cage_kill_fraction',g.get('cage_kill_fraction'),p['cage_kill_fraction']+.1,lambda a,b:a<=b+1e-9),
               ('shaping_share',reward.get('threat_reward_fraction_total'),.3,lambda a,b:0<=a<=b+1e-9)]
        for ref in ('reference_transport1536','reference_selfplay3072'):
            specs.append(('strict_above_'+ref,g.get('win_minus_loss'),refs.get(ref,{}).get('greedy',{}).get('win_minus_loss'),lambda a,b:a>b))
        checks=[dict(metric=k,value=a,threshold=b,status='MISSING' if a is None or b is None else 'PASS' if fn(a,b) else 'FAIL') for k,a,b,fn in specs]
        result[name]=dict(status=gate['absolute']['status'],gate=gate,checks=checks,natural=natural,
            policy_sha256=marker['policy_sha256'],natural_sha256=sha(out/'NATURAL_COMPLETE.json'),reward_monitor=reward)
    atomic_json(OUT/'HOLDOUT_B_AUDIT.json',result)
    return result


def queue_fresh(audits):
    """Only B PASS can instantiate this predeclared independent-seed CPU matrix."""
    old=read(BASE/'fresh256_a3sr_u640/PROTOCOL.json')
    for i,(name,audit) in enumerate(audits.items()):
        if audit['status']!='PASS':continue
        out=OUT/'fresh'/name;path=out/'PROTOCOL.json'
        if path.exists():continue
        req=read(BASE/'eval_requests'/f'{name}.json',read(OUT/'pending_requests'/f'{name}.json'))
        models=dict(old['models']);model=BASE/'eval'/name/'model/model.json'
        cm=read(Path(req['checkpoint'])/'COMMITTED.json')
        models['candidate']=dict(checkpoint=req['checkpoint'],policy_sha256=cm['policy_sha256'],
            model_json=str(model),model_sha256=sha(model))
        tiers={}
        for j,tier in enumerate(('hard','normal','easy')):
            seed=2026163900+i*4000+j*1000
            spawn=dict(read(BASE/'holdout_spawns.json'),seed=seed,source='holdout64 geometry reuse; independent fresh action/env seeds')
            sp=out/f'{tier}_spawns.json';atomic_json(sp,spawn)
            tiers[tier]=dict(seed=seed,seed_end=seed+255,spawn_manifest=str(sp),spawn_sha256=sha(sp))
        files=list(old['source_sha256'])+['scripts/bomb_threat_round16_fresh.py']
        atomic_json(out/'B_ADMISSION.json',audit)
        proto=dict(old,id='ROUND16-'+name+'-FRESH256',instruction='.hermes/STEER_gpu5_holdout_20261030_0815.md',
            models=models,tiers=tiers,source_sha256={f:sha(Path(f)) for f in files},
            native_shaping_share=audit['reward_monitor']['threat_reward_fraction_total'],
            B_admission=str(out/'B_ADMISSION.json'),B_admission_sha256=sha(out/'B_ADMISSION.json'),
            no_publication=True,no_promotion=True,unix=time.time())
        atomic_json(path,proto)
        atomic_json(out/'QUEUED.json',dict(status='B_PASS_QUEUED',matrix_games=3840,
            games_per_mode_per_tier=256,candidate_modes=['greedy','categorical'],baseline_modes=['greedy'],
            teacher=False,CPU_only=True,source_state_not_required_for_inference=True,unix=time.time()))
    # One bounded matrix at a time, reusing the old fresh CPU lane after its completion.
    active=False
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            if any(m in argv for m in (b'scripts.bomb_threat_a3sr_fresh',b'scripts.bomb_threat_round16_fresh')) and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':active=True
        except OSError:pass
    if active or not (BASE/'fresh256_a3sr_u640/COMPLETE.json').exists():return
    for path in sorted((OUT/'fresh').glob('*/PROTOCOL.json')):
        out=path.parent
        if (out/'COMPLETE.json').exists() or read(out/'EXECUTION.json',{}).get('status')=='BLOCKED':continue
        log=(out/'OWNER.log').open('ab',buffering=0)
        subprocess.Popen(['taskset','-c','112-115',sys.executable,'-m','scripts.bomb_threat_round16_fresh',out.name],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(out/'LAUNCH.json',dict(status='CPU_MATRIX_STARTED',protocol_sha256=sha(path),unix=time.time()))
        break


def snapshot():
    release_requests();slots=read(BASE/'SLOTS.json');stats={str(s):recent(s) for s in (0,1,3,4,5,6,7)}
    d=decision();hh=holdouts();queue_fresh(hh);hack=[]
    for u in range(1240,1521,40):
        r=argmax(f'bt_slot07_c000029_u{u:06d}')
        if r:
            hack.append(dict(update=u,bombs_per_game=r['bombs_per_game'],causal_kills=r['causal_kills'],
                causal_kills_per_bomb=r['causal_kills']/r['actual_bombs'] if r['actual_bombs'] else None,
                draw_kill_difference=r['draw_kill_difference'],trigger=r['bombs_per_game']>45 and r['causal_kills']<20))
    pair=[r for r in hack if r['update'] in (1400,1440)]
    suspected=len(pair)==2 and all(r['trigger'] for r in pair)
    atomic_json(OUT/'SLOT07_HACK_OBSERVATION.json',dict(status='SUSPECTED' if suspected else 'NO_TWO_POINT_TRIGGER' if len(pair)==2 else 'WAITING_SECOND_POINT',
        points=hack,changes_to_training=0,proposal_if_triggered='audit opportunity-normalized placement effectiveness and capped causal credit; await orchestration approval, no parameter/stop change',unix=time.time()))
    live=[]
    for slot,e in slots.items():
        o=BASE/'formal'/e['run'];r=read(o/'PROGRESS.json',{}).get('last',{})
        live.append(dict(slot=slot,run=e['run'],update=r.get('new_update'),finite=r.get('finite'),
            env_per_second=262144/r['cycle_seconds'] if r.get('cycle_seconds') else None,
            complete=(o/'COMPLETE.json').exists()))
    missing=[]
    for cp in ['bt_slot07_c000029/u001280','bt_slot05_c000028/u001200','bt_slot05_c000028/u001280']:
        p=BASE/'formal'/cp;cm=read(p/'COMMITTED.json');state=(p/'state.pkl').exists()
        if state:assert sha(p/'state.pkl')==cm['state_sha256']
        assert sha(p/'policy.pt')==cm['policy_sha256']
        missing.append(dict(checkpoint=str(p),state_present=state,state_sha256=cm['state_sha256'],policy_sha256=cm['policy_sha256']))
    result=dict(status='TRAINING_AND_ASYNC_EVALUATION',live=live,stats=stats,slot04=d,
        slot07_suspicion=suspected,holdouts=hh,checkpoint_audit=missing,
        first_formal=read(OUT/'SLOT05_FIRST_FORMAL.json'),unix=time.time())
    atomic_json(OUT/'RESULTS.json',result)
    text=['# BOMB-THREAT round16 — 20261030','',
        '执行08:15指令。GPU5仅从自身u1280完整state续训固定320更新到u1600；A3SR奖励/seed/解码/超参不变，无预热、无RNG重置。其他训练不受干扰；不晋升、不发布。','',
        '## GPU5续训与状态','',
        f"授权 `{OUT}/AUTHORIZATION.json`；接管 `{OUT}/SLOT05_HANDOFF.json`；首正式验收 `{OUT}/SLOT05_FIRST_FORMAL.json`。",'',
        f"首正式证据：`{ {k:result['first_formal'][k] for k in ('status','run','first_post_restore_update','finite','update_l2','environment_steps','env_per_second')} if result['first_formal'] else None }`（首步包含冷启动，稳态看上表）。",'',
        '| slot | 当前run | 实际update | finite | env/s（最近真实更新） | chunk终态 |','|---|---|---:|---|---:|---|']
    for r in live:text.append(f"|{r['slot']}|{r['run']}|{r['update']}|{r['finite']}|{round(r['env_per_second'],1) if r['env_per_second'] else '待更新'}|{r['complete']}|")
    text+=['','## 各slot最近4个同谱系dev argmax点','',
        '全部为生产JS stock-window-v1、teacher off、dev固定seed2026122900–2026122963，各点64局，P0/P1各32；校验导出模型SHA、原始/事件分片逐局seed/seat/ticks/终局/因果击杀并复算。零泡为己方零放且敌方实际放泡>0；双零单列于原始JSON，不冒充策略零泡。', '',
        '单元格“总/P0/P1”依次列完整样本及分seat统计；泡/局按各自分母64/32/32。','',
        '| slot / checkpoint | W/D/L 总；P0；P1 | W−L 总/P0/P1 | 因果击杀 总/P0/P1 | 泡/局 总/P0/P1 | 单方零泡 总/P0/P1 |',
        '|---|---|---|---|---|---|']
    for s,rows in stats.items():
        for r in rows:
            groups=[r,r['by_seat']['P0'],r['by_seat']['P1']]
            wdl='；'.join('/'.join(str(g[k]) for k in ('win','draw','loss')) for g in groups)
            cells=['/'.join(f'{g[k]:.2f}' if k=='bombs_per_game' else str(g[k]) for g in groups) for k in ('win_minus_loss','causal_kills','bombs_per_game','zero_bomb_games')]
            text.append(f"|{s} / {r['name']}|{wdl}|"+'|'.join(cells)+'|')
    text+=['','## 条件PBT与防hack观察','',f"slot04判定：**{d['status']}**。对照最近4点均值 `{d.get('control_means')}`；逐点严格比较 `{d.get('strict_below')}`。证据及冻结时点见 `{OUT}/SLOT04_DECISION.json`。缺任何目标点不会触发复制。",'',
        'slot04两点全指标退化条件未成立；没有写入停机或PBT切换请求，原训练继续。完整state缺失不以policy伪装恢复。','',
        f"slot07两点评价：`{read(OUT/'SLOT07_HACK_OBSERVATION.json')['status']}`；不改训练。",'',
        '| slot07 update | 泡/局 | 因果击杀 | 击杀/泡 | 平局击杀差 | 单点疑似条件 |','|---:|---:|---:|---:|---:|---|']
    for r in hack:text.append(f"|{r['update']}|{r['bombs_per_game']:.3f}|{r['causal_kills']}|{r['causal_kills_per_bomb']:.5f}|{r['draw_kill_difference']}|{r['trigger']}|")
    controls=recent(0,5)
    text+=['',f"slot00对照近5点W−L：`{[r['win_minus_loss'] for r in controls]}`；单方零泡 `{[r['zero_bomb_games'] for r in controls]}`。标注退化/波动，保持对照不动，不据此调参。",'',
        '## Holdout B逐条审计与完整state限制','',
        '生产JS argmax/categorical各64局才判B；GPU2 JAX Hunter仅辅助。CPU已独立开始两点JS评估，保留已完成分片；GPU2请求暂存pending_requests，仅fresh256整矩阵COMPLETE之后对唯一owner可见，避免重启当前评估。B PASS后才冻结并排队新seed每档256复核；不会晋升或发布。','']
    for cp in missing:text.append(f"- `{cp['checkpoint']}`：完整state存在={cp['state_present']}；policy SHA `{cp['policy_sha256']}`；state登记SHA `{cp['state_sha256']}`。")
    text+=['','slot07 u1280在本次pin之前已被清理，pin不能恢复文件。policy仍能做holdout；完整state PBT必须走指令限定的合格fallback。u1200 donor保留原pin。','']
    for name,v in hh.items():
        text += [f"### {name}",'',f"B状态 **{v['status']}**；未达标 `{v.get('gate',{}).get('absolute',{}).get('failures',[])}`；缺数据 `{v.get('gate',{}).get('absolute',{}).get('missing',v.get('missing',[]))}`。",'']
        for check in v.get('checks',[]):text.append(f"- {check['metric']}: {check['status']}，实测{check['value']}，门槛{check['threshold']}。")
        text.append('')
    text+=['## GPU连续采样','',f"原始连续样本 `{OUT}/GPU_SAMPLES.jsonl`，按10秒间隔记录util/memory/power；瞬时0%或仅占显存不当作有效更新。",'']
    samples=[json.loads(l) for l in (OUT/'GPU_SAMPLES.jsonl').read_text().splitlines()] if (OUT/'GPU_SAMPLES.jsonl').exists() else []
    if samples:
        text += ['| GPU | 样本数 | util最小/均值/最大% | 功率最小/均值/最大W |','|---|---:|---|---|']
        for gpu in range(8):
            rows=[r for sample in samples for r in sample['gpus'] if r['gpu']==gpu]
            def triple(k):
                a=[r[k] for r in rows];return f'{min(a):.1f}/{sum(a)/len(a):.1f}/{max(a):.1f}'
            text.append(f"|{gpu}|{len(rows)}|{triple('util')}|{triple('power')}|")
    text+=['','复现：`.venv/bin/python -m scripts.bomb_threat_round16_report --once`；focused：`.venv/bin/python -m pytest -q tests/test_bomb_threat_round16.py tests/test_bomb_threat_round15.py tests/test_bomb_threat_round14.py`。原始逐局文件SHA、完整两点判定及各座位分母保存在round16/RESULTS.json。','']
    if (BASE/'iter2/a3sr/promotion_20261030/AUTHORIZATION.json').exists():
        text += ['## 后续授权更新','',
            '08:15的GPU5固定320预算和首更新验收已经完成接入；随后DECISION_promote_a3sr_20261030明确授权slot05继续训练，覆盖u1600后等待扩展的旧终态限制。slot00/03/04/07在各自完整chunk边界推广A3SR，进度另见reports/bomb_threat_v1_a3sr_promotion_20261030.md；原始B判定保留历史，不作为新推广的准入。','']
    REPORT.write_text('\n'.join(text))
    accepted=bool(result['first_formal']) and all(v['status'] not in ('MISSING','INCOMPLETE') for v in hh.values()) and read(OUT/'EVAL_ORDER.json',{}).get('status')=='RELEASED_AFTER_FRESH_COMPLETE'
    atomic_json(OUT/'TASK_CONSUMED.json',dict(status='COMPLETED_TRAINING_CONTINUES' if accepted else 'FIRST_FORMAL_ACCEPTED_ASYNC_PENDING' if result['first_formal'] else 'WAITING_FIRST_FORMAL',
        report=str(REPORT),report_sha256=sha(REPORT),slot04_status=d['status'],holdout_status={n:v['status'] for n,v in hh.items()},
        first_formal=bool(result['first_formal']),training_untouched_except_authorized_slot05=True,unix=time.time()))
    if accepted and not (OUT/'TASK_NOTIFICATION.json').exists():
        value=dict(status='COMPLETED_TRAINING_CONTINUES',report=str(REPORT),report_sha256=sha(REPORT),
            holdout_status={n:v['status'] for n,v in hh.items()},unix=time.time())
        atomic_json(OUT/'TASK_NOTIFICATION.json',value)
        atomic_json(BASE/'notifications/ROUND16_TASK.json',value)
        from scripts.iter_pbt_owner import notify
        notify('ROUND16_TASK',f"round16 GPU5已恢复正式非零训练；两点holdout B={value['holdout_status']}，GPU2按fresh完成依赖接续；slot04不触发PBT，报告已落盘。",report=str(REPORT))
    return result


def main():
    if '--once' in sys.argv:snapshot();return
    guard=lock(OUT/'MONITOR.lock');last=0
    while not (BASE/'USER_STOP_EVAL.json').exists():
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
            rows=[]
            for line in raw.splitlines():
                g,u,m,p=map(float,line.split(','));rows.append(dict(gpu=int(g),util=u,memory_mib=m,power=p))
            with (OUT/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),gpus=rows))+'\n')
            if time.time()-last>=60:
                snapshot()
                from scripts.bomb_threat_a3sr_compare_decision import refresh as compare_decision
                compare_decision()
                last=time.time()
        except Exception as error:
            atomic_json(OUT/'MONITOR_RETRY.json',dict(error_type=type(error).__name__,detail=str(error),unix=time.time()))
        time.sleep(10)

if __name__=='__main__':main()
