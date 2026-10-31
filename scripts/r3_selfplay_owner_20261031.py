"""Single durable R3-sp owner; original R3 and legacy queues are never edited."""
import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
from scripts.r3_selfplay_gate_20261031 import BASE, OUT, evaluate
from scripts.bun_native_v2_owner import processes

WORKER='scripts.r3_selfplay_train_20261031'
MODULE='scripts.r3_selfplay_owner_20261031'
EVALUATOR='scripts.r3_selfplay_eval_20261031'


def read(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def tail_rows(path, count=64):
    # Historical reference JSONL files are large; monitoring must not reread
    # their complete trajectories every10s on the shared checkpoint filesystem.
    with Path(path).open('rb') as stream:
        stream.seek(0,2);end=stream.tell();data=b'';position=end
        while position and data.count(b'\n')<=count:
            size=min(position,131072);position-=size;stream.seek(position);data=stream.read(size)+data
    lines=data.splitlines()
    if position:lines=lines[1:]
    result=[]
    for line in lines[-count:]:
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue  # current append may be incomplete
    return result


def notify(key, message, **evidence):
    f=OUT/'notifications'/f'{key}.json'
    if f.exists():return
    atomic_json(f,dict(message=message,**evidence,unix=time.time()))
    sent=subprocess.run(['jdme-notify','send',message],capture_output=True,check=False)
    atomic_json(f,dict(message=message,**evidence,notify_returncode=sent.returncode,unix=time.time()))


def paused():
    return any((root/f).exists() for root in (OUT,Path('runs/r3_items_route_20261030'),Path('runs/bun_native_v2_20261030'),
        Path('runs/bomb_threat_v1'),Path('runs/iter_pbt_v1')) for f in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))


def write_parent_rejection(gate):
    report=Path('reports/r3_selfplay_ramp_20261031.md')
    text=report.read_text() if report.exists() else '# R3-sp（20261031）\n'
    text=text.split('\n<!-- FINAL_PARENT_GATE -->')[0]
    lines=['','<!-- FINAL_PARENT_GATE -->','',
           '完整本侧6144局已完成，R3-sp未启动：预声明排序的最好R3父点 `'+gate['parent']+'` 未满足全部门槛。原R3保持30%继续训练；不换次优点绕过门槛。','',
           '|条件|席位|局数|胜/平/负|胜负差|', '|---|---|---:|---|---:|']
    for condition,value in gate['conditions'].items():
        for seat in ('P0','P1'):
            x=value[seat]
            lines.append(f"|{condition}|{seat}|{x['games']}|{x['wins']}/{x['draws']}/{x['losses']}|{x['win_minus_loss']}|")
        lines.append('\n未通过项：'+str([k for k,v in value['checks'].items() if not v])+f"；P1胜率差{value['win_rate_gap']:.6f}、负率超额{value['loss_rate_excess']:.6f}。\n")
    report.write_text(text+'\n'.join(lines)+'\n')


def choose_gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    samples=[tuple(int(x.strip()) for x in line.split(',')) for line in raw.splitlines()]
    candidates=[x for x in samples if x[1]>=48*1024]
    if not candidates:return None
    # Prefer GPU2 if it has capacity; never move an existing reference worker.
    return min(candidates,key=lambda x:(x[0]!=2,x[2],-x[1],x[0]))[0]


def reference_window_snapshot(rows):
    """Real five-update throughput; match the existing alternating scheduler phase."""
    if len(rows)<6:return None
    windows={}
    for window in (rows[-6:-1],rows[-5:]):
        indices=[r['new_update'] for r in window]
        assert indices==list(range(indices[0],indices[0]+5)), 'noncontiguous reference window'
        seconds=sum(r['cycle_seconds'] for r in window)
        steps=sum(r['actual_environment_steps_this_update'] for r in window)
        value=dict(update=indices[-1],window_updates=indices,window_seconds=seconds,
                   window_environment_steps=steps,env_per_second=steps/seconds)
        windows[str(indices[-1]%2)]=value
    return dict(windows[str(rows[-1]['new_update']%2)],phase_baselines=windows)


def reference_snapshot():
    result={}
    for root,table in [('runs/bomb_threat_v1','SLOTS.json'),('runs/bun_native_v2_20261030','R2_SLOTS.json'),
                       ]:
        for slot,entry in read(Path(root)/table,{}).items():
            protocol=read(Path(root)/'protocols'/(entry['run']+'.json'),{})
            module=entry.get('worker_module',protocol.get('worker_module','scripts.bomb_threat_train' if root.endswith('bomb_threat_v1') else 'scripts.bun_native_v2_train'))
            if entry.get('suspended') or not list(processes(module,entry['run'])):
                continue
            f=Path(root)/'formal'/entry['run']/'updates.jsonl'
            if not f.exists():continue
            rows=tail_rows(f,18)
            rows=[r for r in rows if r.get('finite') and r.get('cycle_seconds',0)>0]
            if len(rows)>=8:
                result[root+':'+slot]=dict(run=entry['run'],**reference_window_snapshot(rows))
    return result


def reference_minimum_ratio(key):
    """16:00 authorization: legacy BT may lose15%; R2/R3 may lose5%."""
    root, _, slot = key.partition(':')
    return .85 if root == 'runs/bomb_threat_v1' else .95


def resource_guard():
    f=OUT/'RESOURCE_GUARD.json';state=read(f,dict(baseline={},strikes={},last_update={}))
    authority=(OUT.parent if OUT.name=='seed02' else OUT)/'AMENDMENT05_RECLAIM_ACTIVE_REFERENCES.json'
    if authority.exists() and state.get('baseline_authorization_sha256')!=sha(authority):
        declaration=read(authority)
        state.update(baseline=declaration['reference_baselines'],strikes={},
                     last_update={k:v['update'] for k,v in declaration['reference_baselines'].items()},
                     baseline_authorization_sha256=sha(authority))
    correction=authority.parent/'AMENDMENT06_COLD_REFERENCE_EXCLUSION.json'
    if correction.exists() and state.get('baseline_correction_sha256')!=sha(correction):
        state['baseline'].update(read(correction)['baseline_overrides'])
        state['baseline_correction_sha256']=sha(correction)
    current=reference_snapshot();comparisons={}
    for key,value in current.items():
        base=state['baseline'].get(key)
        if base is None or base['run']!=value['run']:
            state['baseline'][key]=value;state['strikes'][key]=0;state['last_update'][key]=value['update'];continue
        # Retain the original pre-SP update anchor during the measurement fix.
        # Reconstruct from its immutable rows, never rebase onto slowed training.
        if 'phase_baselines' not in base and 'phase_baselines' in value:
            root=key.rsplit(':',1)[0]
            history=tail_rows(Path(root)/'formal'/base['run']/'updates.jsonl',256)
            history=[r for r in history if r.get('finite') and r.get('cycle_seconds',0)>0 and r['new_update']<=base['update']]
            fixed=reference_window_snapshot(history)
            assert fixed and fixed['update']==base['update'],'original reference anchor unavailable'
            base=dict(run=base['run'],**fixed);state['baseline'][key]=base
            state['strikes'][key]=0;state['last_update'][key]=value['update']
        phase=str(value['update']%2)
        matched=base.get('phase_baselines',{}).get(phase,base)
        ratio=value['env_per_second']/matched['env_per_second']
        threshold=reference_minimum_ratio(key)
        previous=state['last_update'].setdefault(key,base['update'])
        if value['update']-previous>=5:
            state['strikes'][key]=state['strikes'].get(key,0)+1 if ratio<threshold else 0
            state['last_update'][key]=value['update']
        comparisons[key]=dict(ratio=ratio,minimum_ratio=threshold,baseline=dict(matched,run=base['run']),current=value,strikes=state['strikes'].get(key,0),
                              estimator='five update total steps / total cycle seconds; baseline matched by alternating update phase')
    failures=[k for k,v in comparisons.items() if v['strikes']>=3]
    state.update(comparisons=comparisons,status='PAUSE_SP_ONLY' if failures else 'OBSERVING',unix=time.time());atomic_json(f,state)
    if failures and not (OUT/'RESOURCE_PAUSE.json').exists():
        request=read(OUT/'DEDICATED_CAPACITY_RESUME.json',{})
        atomic_json(OUT/'RESOURCE_PAUSE.json',dict(status='PAUSE_SP_ONLY_AT_ATOMIC_UPDATE',reason='reference warm env/s below authorized per-line threshold over three disjoint5-update windows',references=failures,thresholds={k:comparisons[k]['minimum_ratio'] for k in failures},unix=time.time()))
        notify('RESOURCE_GUARD_'+request.get('attempt','legacy'),'R3-sp资源护栏：活跃参照吞吐连续三个5更新窗口越界（BT15%、R2 5%），仅请求新arm原子保存暂停；冻结R3对照不恢复。',receipt=str(f))


def dedicated_attempt(request):
    name=request.get('attempt',f"dedicated_gpu{request['exclusive_gpu']}")
    assert name and Path(name).name==name and name not in ('.','..'), 'invalid attempt name'
    return OUT/'attempts'/name


def compute_app_count(gpu):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid',
                                  '--format=csv,noheader,nounits'],text=True)
    identities={int(a.strip()):b.strip() for a,b in (line.split(',') for line in rows.splitlines())}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                  '--format=csv,noheader,nounits'],text=True)
    return sum(line.strip()==identities[gpu] for line in apps.splitlines())


def consume_dedicated_capacity():
    """Adopt the authorized exclusive card through the existing unique owner."""
    request_file=OUT/'DEDICATED_CAPACITY_RESUME.json'
    request=read(request_file,{})
    if request.get('status') not in ('AUTHORIZED','VERIFYING_EXCLUSIVE_RESTORE'):return False
    gpu=request['exclusive_gpu']
    assert gpu in (3,4),'protected/shared cards cannot be claimed as exclusive'
    registration=read(OUT/'REGISTRATION.json');entry=registration['entry']
    folder=OUT/'formal'/entry['run']
    if request['status']=='AUTHORIZED':
        if list(processes(WORKER,entry['run'])):return True
        evidence=request['paused_R2']
        assert evidence and any(x['receipt']['gpu']==gpu for x in evidence)
        for item in evidence:
            assert sha(Path(item['path']))==item['sha256']
            receipt=read(item['path']);assert receipt['status']=='PAUSED_COMPLETE_CHUNK'
            cp=Path(receipt['source']['checkpoint'])
            assert sha(cp/'state.pkl')==receipt['source']['state_sha256']
            assert sha(cp/'policy.pt')==receipt['source']['policy_sha256']
        if compute_app_count(gpu):
            atomic_json(OUT/'STATUS.json',dict(status='WAIT_EXCLUSIVE_GPU_DRIVER_RELEASE',gpu=gpu,unix=time.time()))
            return True
        cp=Path(request['source_checkpoint']);saved=read(cp/'COMMITTED.json')
        assert saved['update']==3040 and saved['metrics']['formal_updates_discarded']==0
        assert sha(cp/'state.pkl')==saved['state_sha256']
        assert sha(cp/'policy.pt')==saved['policy_sha256']
        assert (folder/'NEXT_STEP_u003040.json').exists()
        archive=dedicated_attempt(request)
        archive.mkdir(parents=True,exist_ok=True)
        for source in (OUT/'REGISTRATION.json',folder/'LAUNCH.json',folder/'RESTORE.json'):
            if source.exists():
                dest=archive/('BEFORE_'+source.name)
                assert not dest.exists(),'do not overwrite a previous resource attempt'
                atomic_json(dest,read(source))
                if source.name in ('LAUNCH.json','RESTORE.json'):source.rename(archive/source.name)
        entry['gpu']=gpu;registration.update(entry=entry,exclusive_gpu=True,placement_revision=3,
            placement_authorization=request['authorization'],unix=time.time())
        atomic_json(OUT/'REGISTRATION.json',registration)
        launch(entry,'RESTORE')
        request.update(status='VERIFYING_EXCLUSIVE_RESTORE',launched_unix=time.time())
        atomic_json(request_file,request)
        atomic_json(OUT/'STATUS.json',dict(status='EXCLUSIVE_GPU_RESTORE_NOT_FORMAL_TRAINING',gpu=gpu,unix=time.time()))
        return True
    restored=read(folder/'RESTORE.json')
    if list(processes(WORKER,entry['run'])):return True
    if not restored:
        if time.time()-request['launched_unix']<30:return True
        raise RuntimeError('exclusive GPU restore exited without acceptance; inspect RESTORE.log')
    assert restored['checkpoint']==request['source_checkpoint']
    assert all(restored.get(k) for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
    assert compute_app_count(gpu)==0,'exclusive restore must exit before counted smoke'
    archive=dedicated_attempt(request)
    if (OUT/'DEDICATED_GPU_RESTORE_ACCEPTANCE.json').exists():
        dest=archive/'PREVIOUS_DEDICATED_GPU_RESTORE_ACCEPTANCE.json'
        assert not dest.exists()
        (OUT/'DEDICATED_GPU_RESTORE_ACCEPTANCE.json').rename(dest)
    (folder/'RESTORE.json').rename(OUT/'DEDICATED_GPU_RESTORE_ACCEPTANCE.json')
    (folder/'LAUNCH.json').rename(archive/'CROSS_CARD_RESTORE_LAUNCH.json')
    # The next smoke must earn a fresh restore at u3042. u3040's relocation
    # acceptance cannot satisfy that independent checkpoint requirement.
    atomic_json(OUT/'RESOURCE_GUARD.json',dict(baseline=reference_snapshot(),strikes={},last_update={},unix=time.time()))
    for name in ('RESOURCE_PAUSE.json','RESOURCE_CAPACITY_BLOCK.json'):
        p=OUT/name
        if p.exists():p.rename(archive/name)
    request.update(status='CONSUMED',restore_sha256=sha(OUT/'DEDICATED_GPU_RESTORE_ACCEPTANCE.json'),
                   smoke_and_fresh_restore_required=True,unix=time.time())
    atomic_json(request_file,request)
    notify('DEDICATED_GPU_READY_'+request.get('attempt','legacy'),f'R3-sp独占GPU{gpu}：R2完整chunk已暂停并核验释放，u3040完整恢复PASS；同seed原640预算继续，接smoke和fresh restore，护栏BT15%、R2/R3 5%。',receipt=str(request_file))
    return False


def register(gate):
    if (OUT/'REGISTRATION.json').exists():return read(OUT/'REGISTRATION.json')
    assert gate['status']=='PASS'
    gpu=choose_gpu()
    if gpu is None:return None
    parent=gate['parent_source'];cp=Path(parent['checkpoint']);cm=read(cp/'COMMITTED.json')
    assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']==parent['policy_sha256']
    source=read('runs/r3_items_route_20261030/protocols/'+cp.parent.name+'_performance02.json')
    p=copy.deepcopy(source);start=cm['update'];name='r3_sp55_c000000'
    recipe=copy.deepcopy(next(iter(p['arms'].values())));weights=[.05,.15,.25,.25,.20,.10]
    recipe['opponent_weights']=weights
    contract=read(OUT/'PROTOCOL.json')
    fixed=[read(Path(f)/'COMMITTED.json') for f in contract['fixed_history_checkpoints']]+[cm]
    for identity in fixed:assert sha(Path(identity['checkpoint'])/'policy.pt')==identity['policy_sha256']
    p.update(namespace=str(OUT/'formal'),namespace_id='R3-SP-RAMP-20261031',arms={name:recipe},
        continuation_source=cm,policy_source=cm,chunk_start_update=start,chunk_updates=640,
        environment_steps_per_run=(start+640)*262144,r2_same_version=True,spawn_side_migration=True,
        spawn_side_version='own-side-native-v1',spawn_manifest=str(BASE/'train_spawns.json'),
        spawn_manifest_sha256=sha(BASE/'train_spawns.json'),r1_seed=2026323001,round14_collector_seed=2026323001,
        A1_weights=weights,selfplay_fixed_pool=fixed)
    # Coefficient requests are copied, never edited in the original R3 arm.
    for key in ('r1_coefficient_request','r1_item_coefficient_request','r3_item_coefficient_request'):
        old=Path(p[key]);dest=OUT/'controls'/old.name
        if old.exists():atomic_json(dest,read(old))
        p[key]=str(dest)
    p['source_sha256']={f:sha(Path(f)) for f in p['source_sha256']}
    for f in ('scripts/bun_spawn_side.py','scripts/bun_spawn_side_train.py','scripts/r3_selfplay_train_20261031.py','scripts/r3_selfplay_monitor.py'):
        p['source_sha256'][f]=sha(Path(f))
    protocol=OUT/'protocols'/f'{name}.json';atomic_json(protocol,p)
    entry=dict(run=name,protocol=str(protocol),gpu=gpu,cpu_affinity='128-135',seed=2026323001)
    result=dict(status='REGISTERED_NOT_YET_UPDATED',parent_source=cm,parent_gate=gate,entry=entry,
        protocol_sha256=sha(protocol),new_updates=640,new_environment_steps=640*262144,
        baseline_diagnostic_steps=20*262144,original_R3_unchanged=True,old_queues_unchanged=True,unix=time.time())
    atomic_json(OUT/'REGISTRATION.json',result)
    atomic_json(OUT/'RESOURCE_GUARD.json',dict(baseline=reference_snapshot(),strikes={},last_update={},unix=time.time()))
    with Path('docs/training_experiment_ledger.md').open('a') as stream:
        stream.write('\nR3-SP-55-20261031正式注册：`'+str(OUT/'REGISTRATION.json')+'`，父点`'+str(cp)+'`，policy SHA `'+cm['policy_sha256']+'`；'
            'seed2026323001，640×262144真实on-policy步，另20个无更新train rollout作同机会监控基线；'
            '奖励/网络/batch/LR不变，本侧env/RNG显式重建；55% current25/ring20/fixed10，bot45%。原R3不变。\n')
    return result


def launch(entry, phase):
    folder=OUT/'formal'/entry['run'];folder.mkdir(parents=True,exist_ok=True)
    command=['taskset','-c',entry['cpu_affinity'],sys.executable,'-m',WORKER,entry['protocol'],entry['run'],str(entry['seed'])]
    if phase=='SMOKE':command.append('--smoke')
    if phase=='RESTORE':command.append('--verify-restore')
    with (folder/f'{phase}.log').open('ab') as log:
        subprocess.Popen(command,env=dict(os.environ,JAXBOMB_DEVICE_OVERRIDE=str(entry['gpu'])),stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    atomic_json(folder/'LAUNCH.json',dict(status='LAUNCHED_NOT_ACCEPTED',phase=phase,gpu=entry['gpu'],cpu_affinity=entry['cpu_affinity'],
        command=['<PYTHON>','-m',WORKER,entry['protocol'],entry['run'],str(entry['seed'])]+command[9:],unix=time.time()))


def monitor_rows(rows, baseline):
    import numpy as np
    names=[k for k in baseline['metrics'] if k.startswith('r3_sp_')]
    current={k:sum((np.asarray(r['exposure'][k],float) for r in rows),np.zeros_like(baseline['metrics'][k],dtype=float)) for k in names}
    b={k:np.asarray(v,float) for k,v in baseline['metrics'].items()}
    result={};trigger=False
    for seat in (0,1):
        n=sum(current['r3_sp_episodes'][:,seat]);n0=sum(b['r3_sp_episodes'][:,seat])
        enough=min(n,n0)>=64
        rate=lambda x,key:float(sum(x[key][:,seat])/sum(x['r3_sp_episodes'][:,seat])) if sum(x['r3_sp_episodes'][:,seat]) else None
        draw,draw0=rate(current,'r3_sp_draws'),rate(b,'r3_sp_draws')
        passive,passive0=rate(current,'r3_sp_passive'),rate(b,'r3_sp_passive')
        delivery=float(sum(current['r3_sp_active_deliveries'][:,seat]));delivery0=float(sum(b['r3_sp_active_deliveries'][:,seat]))
        flags=dict(draw_15pp=enough and draw>draw0+.15,passive_increase=enough and passive>passive0,
                   delivery_halved=enough and delivery0>0 and delivery<=delivery0*.5)
        trigger|=any(flags.values())
        result['P'+str(seat)]=dict(games=n,parent_games=n0,draw_rate=draw,parent_draw_rate=draw0,
            passive_rate=passive,parent_passive_rate=passive0,deliveries=delivery,parent_deliveries=delivery0,
            equal_environment_step_budget=True,flags=flags)
        for key in ('own_zero_bomb','both_zero_bomb','custody_loops','unresolved'):
            metric='r3_sp_'+key
            result['P'+str(seat)][key]=float(sum(current[metric][:,seat]))
        result['P'+str(seat)]['by_pool']={
            pool:{key.removeprefix('r3_sp_'):float(value[index,seat])
                  for key,value in current.items() if not key.startswith('r3_sp_fixed_')}
            for index,pool in enumerate(('current','ring','fixed'))}
    fixed={}
    for identity in range(3):
        fixed[str(identity)]={}
        for seat in (0,1):
            n=current['r3_sp_fixed_games'][identity,seat];w=current['r3_sp_fixed_wins'][identity,seat];d=current['r3_sp_fixed_draws'][identity,seat]
            unknown=current['r3_sp_fixed_unresolved'][identity,seat]
            resolved=n-unknown
            score=(w+.5*d)/resolved if resolved else None
            fixed[str(identity)]['P'+str(seat)]=dict(games=float(n),wins=float(w),draws=float(d),score=score,
                unresolved=float(unknown),resolved_games=float(resolved),
                elo_difference=float(400*np.log10(np.clip(score,1e-6,1-1e-6)/(1-np.clip(score,1e-6,1-1e-6)))) if resolved else None)
    return dict(status='ROLLBACK' if trigger else 'OBSERVING',seats=result,fixed=fixed,
        scope='train natural full-episode categorical; fixed actor immutable per environment; not independent dev/holdout',
        excluded='genuine Bank, clear arena, ledger tails',monitor_denominator_minimum=64,unix=time.time())


def write_report(registration, rows):
    import numpy as np
    entry=registration['entry'];monitor=read(OUT/'SELFPLAY_MONITOR.json',{})
    current=rows[-1] if rows else {}
    lines=['# R3-sp 自对弈递进（20261031）','',
        '原R3两路保持30%自对弈作对照；新arm首阶段55%自对弈、45% bot，固定640更新。奖励、规则、网络、LR和batch不改。',
        '父点 `'+registration['parent_source']['checkpoint']+'`；policy SHA `'+registration['parent_source']['policy_sha256']+'`。',
        '冻结协议与原始证据：`'+str(OUT/'PROTOCOL.json')+'`、`PARENT_GATE.json`、`REGISTRATION.json`。',
        'current=每更新的当前Actor；ring=4份历史Actor、每16更新替换一份；fixed=3份冻结历史强点，按环境编号静态均分。没有PFSP采样。',
        '当前run `'+entry['run']+'`；最新真实更新 '+str(current.get('new_update'))+'；数值finite='+str(current.get('finite'))+'。',
        '首正式更新以 `FIRST_FORMAL_UPDATE.json` 为准，launch/进程存在不算训练；全部原子保存含Actor/Critic/Adam/env/aux/RNG/ring。',
        '', '|对手池|实际环境tick|实际新局|', '|---|---:|---:|']
    if rows:
        allrows=tail_rows(OUT/'formal'/entry['run']/'updates.jsonl',700)
        ticks=sum((np.asarray(r['exposure']['pool_ticks']) for r in allrows),np.zeros(6))
        games=sum((np.asarray(r['exposure']['pool_new_episodes']) for r in allrows),np.zeros(6))
        for i,name in enumerate(('rule3','jax_hunter','jax_hunter_hard','current','ring','fixed')):
            lines.append(f'|{name}|{ticks[i]:.0f}|{games[i]:.0f}|')
    lines+=['','自对弈监控：同为20×262144 train步窗口，按P0/P1及current/ring/fixed分母；仅自然完整局，Bank、清场场景、终局账本尾段不混入。',
        '固定对手Elo是自然训练局 score=胜+半平 对应的分差，不冒充独立保留集Elo联赛或统计显著性。',
        '循环计数为“观察到击杀己方来源持包者→同源本家库存真实增加”的来源级序列，未保存唯一包ID，不能证明行为意图或同一枚包。',
        '被动局=双方都没有进入过敌笼，不等同已证明合谋；按用户规则阻止进一步加比例，当前健康55%预算继续。',
        '', '```json',json.dumps(monitor,ensure_ascii=False,indent=2),'```','',
        '|已完成bot holdout|对手/席位|局数|W/D/L|胜负差|父点差|', '|---|---|---:|---|---:|---:|']
    for f in sorted((OUT/'evaluation').glob('*/COMPLETE.json')):
        value=read(f);gate=read(f.parent/'BOT_GATE.json',{})
        for condition,seats in value['metrics'].items():
            for seat,v in seats.items():
                delta=gate.get('comparisons',{}).get(condition+'_'+seat,{}).get('delta','父点')
                lines.append(f"|{f.parent.name}|{condition}/{seat}|{v['games']}|{v['wins']}/{v['draws']}/{v['losses']}|{v['win_minus_loss']}|{delta}|")
    lines+=['','缺失数据：尚无COMPLETE的评估不记为通过；每个新点与父点同128局/条件、各席64，生产JS greedy、老师关闭、2400tick。',
        'fixed胜率/Elo曲线各40更新窗口保存在 `runs/r3_selfplay_ramp_20261031/monitor_points/`；逐更新原始exposure可完整重算，各席各pool的零泡/循环/交付/未知终局不混分母。',
        '不可用：没有PFSP池占比；没有唯一包ID级循环归因；训练on-policy categorical与生产JS greedy分开。',
        '护栏：任一hard/coop席位比父点WL低超过6局，安全原子停新arm，原R3 30%继续；平局+15pp/被动率升/交付减半只冻结更高比例。',
        '至少640更新后、bot不降且fixed成绩上升，且无被动化阻塞，才可冻结下一阶段70%（bot≥30%）；未达标不悄改比例。']
    if Path('runs/r3_selfplay_ramp_20261031/RECLAIM_1820_AUTHORIZATION.json').exists():
        lines=[line.replace('原R3两路保持30%自对弈作对照','原R3-a/b冻结u4002作对照').replace('原R3 30%继续','原R3-a/b保持冻结') for line in lines]
    suffix='_seed02' if OUT.name=='seed02' else ''
    Path('reports/r3_selfplay_ramp'+suffix+'_20261031.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'OWNER.lock')
    while True:
        if paused():atomic_json(OUT/'STATUS.json',dict(status='USER_PAUSED',unix=time.time()));time.sleep(10);continue
        if (OUT/'OWNER_BLOCKED.json').exists():return
        from scripts.r3_resume2110_20261031 import consume as consume2110
        if consume2110(sys.modules[__name__]):time.sleep(5);continue
        from scripts.r3_sp_reclaim_1820_20261031 import consume_resume
        if consume_resume(OUT):time.sleep(5);continue
        if consume_dedicated_capacity():time.sleep(5);continue
        gate=evaluate()
        if gate['status']!='PASS':
            atomic_json(OUT/'STATUS.json',dict(gate,training_started=False))
            if gate['status']=='NOT_ELIGIBLE':
                write_parent_rejection(gate)
                with Path('docs/training_experiment_ledger.md').open('a') as ledger:
                    ledger.write('\nR3-SP-RAMP最终父门槛未通过，新arm未启动；完整数据与逐条件差距见 `runs/r3_selfplay_ramp_20261031/PARENT_GATE.json` 及 `reports/r3_selfplay_ramp_20261031.md`。原R3对照继续，不选次优点绕门槛。\n')
                notify('PARENT_NOT_ELIGIBLE','R3-sp未启动：本侧新完整复核的最好R3父点未满足全部hard/coop hard双席门槛，具体差距已落盘PARENT_GATE.json；原R3继续。');return
            time.sleep(20);continue
        acceptance=read(OUT/'CPU_RUNTIME_ACCEPTANCE.json',{})
        assert acceptance.get('status')=='PASS','CPU passive action/reward parity not accepted'
        assert all(sha(Path(f))==h for f,h in acceptance['source_sha256'].items()),'passive runtime changed after acceptance'
        control_slots=read('runs/r3_items_route_20261030/R3_SLOTS.json',{})
        controls={arm:bool(entry.get('spawn_side_version')=='own-side-native-v1' and
            read(Path('runs/r3_items_route_20261030/formal')/entry['run']/'FIRST_FORMAL_UPDATE.json',{}).get('status')=='PASS')
            for arm,entry in control_slots.items()}
        if set(controls)!={'a','b'} or not all(controls.values()):
            atomic_json(OUT/'STATUS.json',dict(status='WAIT_R3_OWNSIDE_CONTROL_FORMAL_ACCEPTANCE',controls=controls,training_started=False,unix=time.time()))
            time.sleep(20);continue
        registration=register(gate)
        if registration is None:
            atomic_json(OUT/'STATUS.json',dict(status='WAIT_GPU_CAPACITY',reason='no card with48GiB free; reference training never interrupted',unix=time.time()));time.sleep(20);continue
        if OUT==Path('runs/r3_selfplay_ramp_20261031') and (OUT/'SECOND_SEED_CONDITIONAL_1610.json').exists():
            from scripts.r3_sp_second_seed_20261031 import safe_advance
            safe_advance()
        entry=registration['entry'];folder=OUT/'formal'/entry['run'];p=read(entry['protocol']);start=p['chunk_start_update']
        if not list(processes(EVALUATOR)) and not (OUT/'EVAL_BLOCKED.json').exists():
            with (OUT/'EVAL_OWNER.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-m',EVALUATOR],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        first=folder/f'u{start+3:06d}/COMMITTED.json'
        if first.exists() and not (folder/'FIRST_FORMAL_UPDATE.json').exists():
            cm=read(first);m=cm['metrics'];restored=read(folder/'RESTORE.json')
            if m.get('user_paused'):
                m=next(r for r in tail_rows(folder/'updates.jsonl') if r['new_update']==start+3)
            assert m['finite'] and min(m['update_l2'].values())>0 and m['actual_environment_steps_this_update']==262144
            assert sha(first.parent/'state.pkl')==cm['state_sha256'] and sha(first.parent/'policy.pt')==cm['policy_sha256']
            assert all(restored.get(k) for k in ('next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
            atomic_json(folder/'FIRST_FORMAL_UPDATE.json',dict(status='PASS',source=cm,restore=restored,env_per_second=262144/m['cycle_seconds'],unix=time.time()))
            notify('FIRST_FORMAL','R3-sp55首正式u'+str(cm['update'])+'：262144真实步、Actor/Critic非零、finite、原子hash/完整恢复PASS；原R3继续。',receipt=str(folder/'FIRST_FORMAL_UPDATE.json'))
        baseline=read(OUT/'MONITOR_BASELINE.json');rows=[]
        if (folder/'updates.jsonl').exists():rows=tail_rows(folder/'updates.jsonl')
        if baseline and len(rows)>=20:
            monitor=monitor_rows(rows[-20:],baseline);atomic_json(OUT/'SELFPLAY_MONITOR.json',dict(monitor,last_update=rows[-1]['new_update']))
            # Persist every observed forty-update window for the fixed Elo
            # curve. Raw per-update counters remain the recomputation source.
            for index,row in enumerate(rows):
                point=OUT/'monitor_points'/f"u{row['new_update']:06d}.json"
                if row['new_update']%40==0 and index>=19 and not point.exists():
                    atomic_json(point,dict(monitor_rows(rows[index-19:index+1],baseline),
                        last_update=row['new_update'],window_updates=20,
                        baseline_sha256=sha(OUT/'MONITOR_BASELINE.json')))
            if monitor['status']=='ROLLBACK' and not (OUT/'ESCALATION_BLOCKED.json').exists():
                atomic_json(OUT/'ESCALATION_BLOCKED.json',dict(status='BLOCK_ESCALATION_ONLY',reason='selfplay monitor',evidence=monitor,unix=time.time()))
                notify('SELFPLAY_ESCALATION_BLOCK','R3-sp55出现被动化/平局/交付预警，已冻结进一步提高比例；继续完成当前有界55%预算，原R3不变。')
        # Protect references during the diagnostic baseline and cold smoke too,
        # rather than waiting for eight counted SP updates to exist.
        resource_guard()
        if rows and (rows[-1]['new_update']%40==0 or (folder/'COMPLETE.json').exists()):write_report(registration,rows)
        for f in (OUT/'evaluation').glob('*/BOT_GATE.json'):
            from scripts.r3_resume2110_20261031 import superseded_bot_gate
            if superseded_bot_gate(sys.modules[__name__],f):continue
            authorization=read(OUT/'RECLAIM_1820_AUTHORIZATION.json',{})
            if str(f) in authorization.get('superseded_bot_gates',{}):continue
            if read(f)['status']=='ROLLBACK' and not (OUT/'RESOURCE_PAUSE.json').exists():
                atomic_json(OUT/'RESOURCE_PAUSE.json',dict(status='PAUSE_R3_SP_ONLY_AT_ATOMIC_UPDATE',reason='bot W-L below parent by more than6',evidence=str(f),unix=time.time()))
                notify('BOT_ROLLBACK','R3-sp55的bot holdout比父点回落超过6局，已请求仅新arm原子暂停；原R3 30%继续。',receipt=str(f))
        if (OUT/'RESOURCE_PAUSE.json').exists():
            running=list(processes(WORKER,entry['run']))
            state='ROLLBACK_REQUESTED_WAIT_SAFE_SAVE'
            if not running:
                checkpoints=sorted(folder.glob('u*/COMMITTED.json'))
                if checkpoints:
                    cm=read(checkpoints[-1]);cp=checkpoints[-1].parent
                    assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
                    assert (folder/f"NEXT_STEP_u{cm['update']:06d}.json").exists()
                    assert cm['metrics'].get('user_paused') or (folder/'COMPLETE.json').exists(),'stop gate was not acknowledged by worker'
                    atomic_json(OUT/'ROLLBACK_SAFE_SAVE.json',dict(status='PASS',source=cm,
                        original_R3_30pct_remains_active=True,training_workers_signalled=False,unix=time.time()))
                    state='ROLLED_BACK_TO_ORIGINAL_R3_30PCT'
            atomic_json(OUT/'STATUS.json',dict(status=state,unix=time.time()))
            time.sleep(10);continue
        if (folder/'COMPLETE.json').exists():
            done=read(folder/'COMPLETE.json');assert done['update']==start+640 and sha(Path(done['checkpoint'])/'state.pkl')==done['state_sha256']
            atomic_json(OUT/'STATUS.json',dict(status='FIRST640_COMPLETE_WAIT_PAIRED_BOT_AND_FIXED_CONFIRMATION',source=done,stage70_started=False,unix=time.time()))
            notify('FIRST640_COMPLETE','R3-sp55固定640更新已原子完成，等待异步bot及fixed曲线验收；未满足条件不提高70%，原R3对照状态不变。',source=done)
            time.sleep(20);continue
        if list(processes(WORKER,entry['run'])):
            atomic_json(OUT/'STATUS.json',dict(status='WORKER_PRESENT_CHECK_FORMAL_RECEIPT',run=entry['run'],last_update=rows[-1]['new_update'] if rows else None,unix=time.time()));time.sleep(10);continue
        phase='SMOKE' if not (folder/'SMOKE.json').exists() else 'RESTORE' if not (folder/'RESTORE.json').exists() else 'FORMAL'
        last=read(folder/'LAUNCH.json',{})
        if last.get('phase')==phase:
            if time.time()-last['unix']<30:time.sleep(5);continue
            raise RuntimeError('worker exited without phase acceptance; inspect '+str(folder/f'{phase}.log'))
        launch(entry,phase)
        atomic_json(OUT/'STATUS.json',dict(status='LAUNCHED_NOT_ACCEPTED',phase=phase,run=entry['run'],unix=time.time()));time.sleep(5)


def watchdog():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'WATCHDOG.lock')
    while True:
        owners=[p for p in processes(MODULE) if b'--watchdog' not in (p/'cmdline').read_bytes().split(b'\0')]
        if not owners and not paused() and not (OUT/'OWNER_BLOCKED.json').exists() and read(OUT/'PARENT_GATE.json',{}).get('status')!='NOT_ELIGIBLE':
            with (OUT/'OWNER.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-m',MODULE],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(OUT/'WATCHDOG_RESTART.json',dict(status='OWNER_RELOAD_LIVE_WORKERS_ADOPTED',unix=time.time()))
        time.sleep(20)


if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--watchdog',action='store_true');args=args.parse_args()
    if args.watchdog:watchdog()
    else:
        try:main()
        except Exception as error:
            atomic_json(OUT/'OWNER_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()))
            notify('TECHNICAL_BLOCKED','R3-sp owner技术阻塞，证据OWNER_BLOCKED.json及OWNER.log；原R3/R2/R1不受信号。');raise
