"""Authorized resource handoff and independent, resumable JS recheck."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import queue
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,lock,sha

OUT=Path('runs/r2a_yield_r3b_verify_20261031')

def read(p,default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default

def notify(key,message):
    p=OUT/'notifications'/f'{key}.json'
    if p.exists():return
    r=subprocess.run(['jdme-notify','send',message],capture_output=True)
    atomic_json(p,dict(message=message,returncode=r.returncode,unix=time.time()))

def handoff():
    from scripts import r3_sp_capacity_20261031 as capacity
    from scripts.bun_native_v2_owner import processes
    guard=lock(OUT/'HANDOFF_OWNER.lock')
    capacity.OUT=OUT
    while not (OUT/'HANDOFF_COMPLETE.json').exists():
        capacity.advance_handoff(read(OUT/'HANDOFF_PROTOCOL.json'),read(OUT/'HANDOFF_DECISION.json'))
        request=read(capacity.SP/'DEDICATED_CAPACITY_RESUME.json',{})
        if request.get('status')=='CONSUMED' and (capacity.SP/'RESOURCE_PAUSE.json').exists():
            if not list(processes('scripts.r3_selfplay_train_20261031')):
                safe=read(capacity.SP/'ROLLBACK_SAFE_SAVE.json',{})
                if safe.get('status')=='PASS' and safe.get('unix',0)>request['unix']:
                    atomic_json(OUT/'HANDOFF_BLOCKED.json',dict(status='BLOCKED_ORIGINAL_RESOURCE_GUARD',
                        safe_save=safe,exclusive_gpu=3,formal_updates_accepted=0,
                        no_automatic_retry=True,reason='original reference throughput gate remains binding',unix=time.time()))
                    notify('SP_SAFE_PAUSE','R3-sp独占GPU3尝试已按原吞吐护栏完整安全保存停止，正式更新0；不自动重试、不恢复R2-a/c，R3-b新seed复核继续。')
                    return
        atomic_json(OUT/'HANDOFF_STATUS.json',dict(status='WAIT_COMPLETE_R2_A_U5281' if not
            (capacity.R2/'capacity_pauses/a.json').exists() else 'SP_OWNER_CONSUMING_EXCLUSIVE_GPU3',unix=time.time()))
        time.sleep(10)
    notify('SP_FIRST_FORMAL','R3-sp已独占GPU3，u3040完整恢复、smoke/fresh restore及首正式更新通过；收据runs/r2a_yield_r3b_verify_20261031/HANDOFF_COMPLETE.json，原640预算继续。')

def extra_metrics(episodes,seat):
    rows=[e for e in episodes if e['seat']==seat]
    draws=[e for e in rows if e['winner'] in (-1,None)]
    return dict(draw_games=len(draws),draw_causal_kill_difference=sum(
        sum(k['attacker']==seat for k in e['kills'])-sum(k['attacker']==1-seat for k in e['kills']) for e in draws),
        carrying_own_glue_contacts=sum(c['type']==2 and c['owner']==seat and c['player']==seat and c['carried']>=0
            for e in rows for c in e['r3']['contacts']))

def evaluate():
    from scripts import r3_selfplay_eval_20261031 as ev
    ev.OUT=OUT
    guard=lock(OUT/'EVAL_OWNER.lock');p=read(OUT/'PROTOCOL.json')
    # Strict idle admission: never borrow GPU2 or preempt its existing queue.
    if not (OUT/'EVAL_ADMISSION.json').exists():
        while True:
            native=read('runs/bomb_threat_v1/NATIVE_STATUS.json',{})
            if native.get('status','').startswith('IDLE'):
                atomic_json(OUT/'EVAL_ADMISSION.json',dict(status='PASS_GPU2_OWNER_IDLE',native=native,
                    implementation='production JS CPU only, no GPU memory allocation',unix=time.time()));break
            atomic_json(OUT/'EVAL_STATUS.json',dict(status='WAIT_GPU2_EVALUATION_OWNER_IDLE',native=native,
                no_GPU_allocation=True,no_training_or_existing_evaluation_preemption=True,unix=time.time()))
            time.sleep(10)
    for key,source in p['models'].items():
        if (OUT/'evaluation'/key/'COMPLETE.json').exists():continue
        req=dict(key=key,source=source);model=ev.export(req)
        cores=queue.Queue()
        for core in range(188,192):cores.put(core)
        def collect(condition,start):
            core=cores.get()
            try:ev.collect(req,condition,start,model,core)
            finally:cores.put(core)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(collect,c,start) for i,(c,start) in enumerate(
                (c,start) for c in ('hunter_hard','coop_hard') for start in range(0,128,8))]
            for i,f in enumerate(jobs):
                f.result();atomic_json(OUT/'EVAL_STATUS.json',dict(status='FULL_GAME_EVALUATION',model=key,
                    accepted_shards_in_model=i+1,total_shards_in_model=32,unix=time.time()))
        # ev.finish expects the immutable parent before candidates.
        assert ev.finish(req)
    result={};files={}
    for key in p['models']:
        d=read(OUT/'evaluation'/key/'COMPLETE.json');files[str(OUT/'evaluation'/key/'COMPLETE.json')]=sha(OUT/'evaluation'/key/'COMPLETE.json')
        assert all(sha(Path(f))==h for f,h in d['files'].items())
        result[key]={}
        for c in ('hunter_hard','coop_hard'):
            es=[]
            for start in range(0,128,8):es+=read(OUT/'evaluation'/key/c/f'shard_{start:03d}.json.outcome.json')['episodes']
            assert len(es)==128 and len({(e['seed'],e['seat']) for e in es})==128
            result[key][c]={}
            for s in (0,1):
                m=d['metrics'][c][f'P{s}'];n=m['games'];x=extra_metrics(es,s)
                result[key][c][f'P{s}']=dict({k:m[k] for k in ('games','wins','draws','losses','win_minus_loss','causal_kills','bombs','zero_bomb','self_deaths','carrying_own_banana_contacts','glue_self_contacts')},
                    **x,bombs_per_game=m['bombs']/n,carrying_own_banana_per_game=m['carrying_own_banana_contacts']/n,
                    carrying_own_glue_per_game=x['carrying_own_glue_contacts']/n,all_phase_own_glue_per_game=m['glue_self_contacts']/n)
    atomic_json(OUT/'VERIFY_COMPLETE.json',dict(status='COMPLETE',games=768,results=result,files=files,
        protocol_sha256=sha(OUT/'PROTOCOL.json'),teachers_off=True,publication=False,unix=time.time()))
    atomic_json(OUT/'EVAL_STATUS.json',dict(status='COMPLETE',full_games=768,unix=time.time()))
    report();notify('R3B_VERIFY_COMPLETE','R3-b u3600新seed复核768完整生产JS局完成；parent/R3-a同清单，逐席位WDL/放泡/击杀/持包蕉胶见reports/r2a_yield_r3b_verify_20261031.md；未发布。')

def report():
    lines=['# R2-a让卡与R3-b新seed复核（20261031）','',
        '授权 `.hermes/STEER_r2a_yield_r3b_verify_20261031_1345.md` 覆盖此前第二条R2暂停差距门槛；仅资源让卡，不改奖励/超参/seed/预算。',
        'R2-a原完整chunk终点u5281原子保存后暂停，c原暂停请求保留；GPU3只给R3-sp，原u3040完整恢复→smoke→fresh restore→640更新。',
        'BT slot04已通过既有逐slot控制器正式退休，u4240完整state SHA核验并pin，禁止回退/后继；slot03保持BLOCK。收据 `runs/r2a_yield_r3b_verify_20261031/BT04_RETIRED.json`。']
    pause=read('runs/bun_native_v2_20261030/capacity_pauses/a.json')
    if pause:lines+=['','R2-a暂停验收：`'+pause['source']['checkpoint']+'`；state SHA `'+pause['source']['state_sha256']+'`，policy SHA `'+pause['source']['policy_sha256']+'`。']
    first=read('runs/r3_selfplay_ramp_20261031/formal/r3_sp55_c000000/FIRST_FORMAL_UPDATE.json')
    if first:lines+=['','R3-sp GPU3首正式u'+str(first['source']['update'])+'；state SHA `'+first['source']['state_sha256']+'`；policy SHA `'+first['source']['policy_sha256']+'`，env/s '+str(first['env_per_second'])+'。']
    else:lines+=['','R3-sp首正式更新：待验收，launch/进程存在不算完成。']
    resource=read(OUT/'SP_DEDICATED_RESOURCE_GUARD.json')
    if resource:
        m=resource['guard_snapshot']['comparisons']['runs/bomb_threat_v1:5']
        lines+=['','R3-sp独占尝试触发原吞吐护栏：BT slot05同run暖态基线env/s '
            +str(m['baseline']['env_per_second'])+'→'+str(m['current']['env_per_second'])+
            '，比例'+str(m['ratio'])+'，三个不重叠5更新窗口下降>5%；仅请求SP安全保存暂停，不改变门槛、不停止原线。是时间相关，不能据此宣称完整因果。资源快照 `SP_DEDICATED_RESOURCE_GUARD.json`。']
    blocked=read(OUT/'HANDOFF_BLOCKED.json')
    if blocked:lines+=['','R3-sp正式更新0，原吞吐闸门阻塞、安全保存收据已验收；暂无首正式SHA。GPU3当前空闲原因是原合同护栏不允许自动重试，不用无关任务占卡、不自动恢复R2-a。']
    diag=read(OUT/'R2_B_DIAGNOSIS.json')
    if diag:lines+=['','R2-b：'+diag['conclusion']+' 原始证据与限制见 `runs/r2a_yield_r3b_verify_20261031/R2_B_DIAGNOSIS.json`。']
    d=read(OUT/'VERIFY_COMPLETE.json')
    if d:
        lines+=['','新seed2026334500起，hard/协作hard各128局、每席64局；三模型共768局，固定本侧map806，生产JS greedy、老师关闭；不代表跨地图泛化。',
            '', '|模型|条件|席位|胜/平/负|因果击杀|泡/局|零泡局|平局击杀差|自伤|持包自踩蕉(次/局)|持包自踩胶(次/局)|全阶段自踩胶|',
            '|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|']
        for model,cs in d['results'].items():
            for c,seats in cs.items():
                for s,m in seats.items():
                    lines.append(f"|{model}|{c}|{s}|{m['wins']}/{m['draws']}/{m['losses']}|{m['causal_kills']}|{m['bombs_per_game']:.3f}|{m['zero_bomb']}|{m['draw_causal_kill_difference']}|{m['self_deaths']}|{m['carrying_own_banana_contacts']}/{m['carrying_own_banana_per_game']:.3f}|{m['carrying_own_glue_contacts']}/{m['carrying_own_glue_per_game']:.3f}|{m['glue_self_contacts']}|")
        lines+=['','香蕉/胶次数为真实消耗接触事件，不能直接当净收益；平局击杀差只累加平局里的己方因果击杀减敌方因果击杀；零分母不补造。']
    else:lines+=['','新seed复核状态：'+str(read(OUT/'EVAL_STATUS.json',{}).get('status','REGISTERED'))+'。GPU2 owner任务间空闲准入实栈见`EVAL_ADMISSION.json`；CPU生产JS评估不申请GPU，训练不阻塞。']
    p=Path('reports/r2a_yield_r3b_verify_20261031.md');tmp=p.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(p)

def supervise():
    from scripts.bun_native_v2_owner import processes
    guard=lock(OUT/'SUPERVISOR.lock')
    while True:
        report()
        for role,receipt in (('--handoff',OUT/'HANDOFF_COMPLETE.json'),('--evaluate',OUT/'VERIFY_COMPLETE.json')):
            if receipt.exists() or (OUT/(role[2:].upper()+'_BLOCKED.json')).exists():continue
            live=[p for p in processes('scripts.r2a_yield_r3b_verify_20261031') if role.encode() in (p/'cmdline').read_bytes().split(b'\0')]
            if not live:
                with (OUT/(role[2:].upper()+'.log')).open('ab') as log:
                    subprocess.Popen(['taskset','-c','188-191',sys.executable,'-m','scripts.r2a_yield_r3b_verify_20261031',role],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        if (OUT/'HANDOFF_COMPLETE.json').exists() and (OUT/'VERIFY_COMPLETE.json').exists():return
        time.sleep(10)

if __name__=='__main__':
    role='handoff' if '--handoff' in sys.argv else 'evaluate' if '--evaluate' in sys.argv else 'supervisor'
    try:{'handoff':handoff,'evaluate':evaluate,'supervisor':supervise}[role]()
    except Exception as e:
        atomic_json(OUT/(role.upper()+'_BLOCKED.json'),dict(status='BLOCKED',error_type=type(e).__name__,reason=str(e),unix=time.time()))
        notify(role+'_BLOCKED','R2-a交卡/R3-b复核技术阻塞：'+role+' '+type(e).__name__+'，证据runs/r2a_yield_r3b_verify_20261031/；训练未发信号。')
        raise
