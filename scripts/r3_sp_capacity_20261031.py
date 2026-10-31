"""Matched CPU comparison and durable resource handoff; never signals trainers."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

OUT = Path('runs/r3_sp_capacity_20261031')
SP = Path('runs/r3_selfplay_ramp_20261031')
R2 = Path('runs/bun_native_v2_20261030')


def read(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def export(key, source):
    dest = OUT/'evaluation'/key/'model'
    receipt = dest/'IDENTITY.json'
    if receipt.exists():
        d = read(receipt)
        assert sha(Path(d['model_json'])) == d['model_sha256']
        assert d['source']['policy_sha256'] == source['policy_sha256']
        return d
    cp = Path(source['checkpoint'])
    assert sha(cp/'policy.pt') == source['policy_sha256']
    dest.mkdir(parents=True, exist_ok=True)
    code = ('from pathlib import Path;from scripts.je08_bun_authority import export;'
            'import sys;export(Path(sys.argv[1]),Path(sys.argv[2]).resolve())')
    with (dest/'EXPORT.log').open('ab') as log:
        subprocess.run(['taskset','-c','184',sys.executable,'-c',code,
                        str(cp/'policy.pt'),str(dest)], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    model = dest/'model.json'
    assert read(model)['meta']['checkpoint_sha256'] == source['policy_sha256']
    d = dict(model_json=str(model),model_sha256=sha(model),source=source)
    atomic_json(receipt,d)
    return d


def select(protocol, metrics):
    parent = metrics['parent']
    scores = {}
    for key, m in metrics.items():
        scores[key] = dict(P1_sum=sum(v['P1']['win_minus_loss'] for v in m.values()),
            conditions={c:{s:{k:v[k] for k in ('games','wins','draws','losses','win_minus_loss','self_deaths')}
                              for s,v in seats.items()} for c,seats in m.items()})
    order = sorted(protocol['models'],key=lambda k:(scores[k]['P1_sum'],list(protocol['models']).index(k)))
    for key in order:
        scores[key]['deficit_against_R3_a'] = scores['parent']['P1_sum']-scores[key]['P1_sum']
    chosen = [order[0]]
    # GPU7 is protected/shared with BT slot07. A second R2 may be paused only
    # under the explicitly authorized >20 deficit condition, never BT slot07.
    if protocol['models'][chosen[0]]['entry']['gpu'] == 7:
        for key in order[1:]:
            if (scores[chosen[0]]['deficit_against_R3_a']>20 and
                scores[key]['deficit_against_R3_a']>20 and
                protocol['models'][key]['entry']['gpu'] in (3,4)):
                chosen.append(key)
                break
    exclusive = next((protocol['models'][key]['entry']['gpu'] for key in chosen
                      if protocol['models'][key]['entry']['gpu'] in (3,4)),None)
    return dict(status='SELECTED' if exclusive is not None else 'BLOCKED_NO_AUTHORIZED_EXCLUSIVE_CARD',
                scores=scores,pause_models=chosen,exclusive_gpu=exclusive,
                primary=protocol['primary'],protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time())


def write_pause_request(protocol, decision):
    assert decision['status'] in ('SELECTED','BLOCKED_NO_AUTHORIZED_EXCLUSIVE_CARD')
    entries={}
    for key in decision['pause_models']:
        v=protocol['models'][key]
        entries[key.removeprefix('r2_')]=dict(run=v['entry']['run'],
            target_update=v['target_chunk_end'],gpu=v['entry']['gpu'],
            comparison_checkpoint=v['source']['checkpoint'])
    request=dict(
        status='AUTHORIZED',authorization=protocol['authorization'],arms=entries,
        selection=str(OUT/'SELECTION.json'),selection_sha256=sha(OUT/'SELECTION.json'),
        no_worker_signals=True,no_checkpoint_deletion=True,unix=time.time())
    path=R2/'controls/SP_CAPACITY_PAUSE_AFTER_CHUNK.json'
    existing=read(path)
    if existing:
        assert all(existing[k]==request[k] for k in ('status','arms','selection','selection_sha256'))
    else:atomic_json(path,request)
    atomic_json(OUT/'STATUS.json',dict(status='WAIT_COMPLETE_R2_CHUNK_BOUNDARY' if decision['status']=='SELECTED'
        else 'BLOCKED_EXCLUSIVE_GPU_CAPACITY_WEAKEST_R2_PAUSE_QUEUED',
        selected=decision['pause_models'],exclusive_gpu=decision['exclusive_gpu'],
        pause_request=str(R2/'controls/SP_CAPACITY_PAUSE_AFTER_CHUNK.json'),unix=time.time()))


def record_capacity_block(protocol, decision):
    """Honor the weakest-arm pause without inventing a second-arm mandate."""
    assert decision['status']=='BLOCKED_NO_AUTHORIZED_EXCLUSIVE_CARD'
    write_pause_request(protocol,decision)
    blocked=OUT/'BLOCKED.json'
    if blocked.exists():return
    reason=('weakest R2 shares GPU7 with protected BT slot07; no exclusive GPU3/4 R2 '
            'also trails the parent by more than20; second R2 pause is not authorized')
    atomic_json(blocked,dict(status='BLOCKED_EXCLUSIVE_GPU_CAPACITY',reason=reason,
        selection=str(OUT/'SELECTION.json'),selection_sha256=sha(OUT/'SELECTION.json'),
        scores=decision['scores'],weakest_R2_complete_chunk_pause_queued=True,
        R3_sp_formal_updates=0,protected_BT_slots_untouched=True,training_workers_signalled=False,unix=time.time()))
    subprocess.run(['jdme-notify','send','R3-sp整卡交接受授权门槛阻塞：最弱R2与受保护BT07共GPU7；'
        'R2-a/b落后父点均未超过20，不能擅停第二条。已排最弱R2完整chunk暂停，SP正式更新仍0；'
        '完整比较见reports/r3_sp_dedicated_capacity_20261031.md。'],capture_output=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    guard=lock(OUT/'OWNER.lock')
    protocol=read(OUT/'PROTOCOL.json')
    # Evaluation and selection are immutable once accepted. A controller
    # restart must not rewrite their SHA while the R2 controller consumes it.
    decision=read(OUT/'SELECTION.json')
    if decision:
        assert decision['protocol_sha256']==sha(OUT/'PROTOCOL.json')
        if decision['status']!='SELECTED':
            record_capacity_block(protocol,decision)
            return
        write_pause_request(protocol,decision)
        while not (OUT/'HANDOFF_COMPLETE.json').exists():
            advance_handoff(protocol,decision)
            time.sleep(10)
        return
    from scripts import r3_selfplay_eval_20261031 as evaluator
    evaluator.OUT=OUT
    parent_path=Path(protocol['parent']['reused_complete'])
    assert sha(parent_path)==protocol['parent']['complete_sha256']
    parent=read(parent_path)
    for filename,h in parent['files'].items():assert sha(Path(filename))==h
    atomic_json(OUT/'evaluation/parent/COMPLETE.json',dict(parent,
        reused_from=str(parent_path),reused_sha256=sha(parent_path)))
    cores=queue.Queue()
    scheduling=read(OUT/'CPU_SCHEDULING_AMENDMENT01.json',{})
    for core in scheduling.get('cores',list(range(184,188))):cores.put(core)
    requests=[]
    for key,model in protocol['models'].items():
        req=dict(key=key,source=model['source'])
        if not (OUT/'evaluation'/key/'COMPLETE.json').exists():
            requests.append((req,export(key,model['source'])))
    def job(req,model,condition,start):
        core=cores.get()
        try:evaluator.collect(req,condition,start,model,core)
        finally:cores.put(core)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,req,model,c,start) for req,model in requests
                 for c in ('hunter_hard','coop_hard') for start in range(0,128,8)]
        for i,future in enumerate(futures):
            future.result()
            atomic_json(OUT/'STATUS.json',dict(status='MATCHED_CPU_EVALUATION',
                completed_shards=i+1,total_shards=len(futures),games_per_shard=8,unix=time.time()))
    for req,_ in requests:assert evaluator.finish(req)
    metrics={key:read(OUT/'evaluation'/key/'COMPLETE.json')['metrics']
             for key in ('parent',*protocol['models'])}
    decision=select(protocol,metrics)
    atomic_json(OUT/'SELECTION.json',decision)
    if decision['status']!='SELECTED':
        record_capacity_block(protocol,decision)
        return
    write_pause_request(protocol,decision)
    subprocess.run(['jdme-notify','send','R3-sp独占卡比较完成：'+
        str({k:v['P1_sum'] for k,v in decision['scores'].items()})+
        '；选中暂停'+','.join(decision['pause_models'])+'，仅在完整chunk边界保存，GPU'+
        str(decision['exclusive_gpu'])+'随后只给SP；BT03/04不操作。'],capture_output=True)
    # The existing R2/SP owners consume the handoff. This CPU owner only
    # publishes a request after the frozen chunk receipt has been verified.
    while not (OUT/'HANDOFF_COMPLETE.json').exists():
        advance_handoff(protocol,decision)
        time.sleep(10)


def advance_handoff(protocol, decision):
    OUT.mkdir(parents=True,exist_ok=True)
    try:guard=lock(OUT/'HANDOFF.lock')
    except BlockingIOError:return
    _advance_handoff(protocol,decision)


def _advance_handoff(protocol, decision):
    """Wait for R2's verified terminal receipt; no direct training launches."""
    required=[R2/'capacity_pauses'/f'{key.removeprefix("r2_")}.json'
              for key in decision['pause_models']
              if protocol['models'][key]['entry']['gpu']==decision['exclusive_gpu']]
    assert required
    if not all(p.exists() and read(p).get('status')=='PAUSED_COMPLETE_CHUNK' for p in required):
        return
    evidence=[dict(path=str(p),sha256=sha(p),receipt=read(p)) for p in required]
    request=dict(status='AUTHORIZED',authorization=protocol['authorization'],
        exclusive_gpu=decision['exclusive_gpu'],paused_R2=evidence,
        source_checkpoint='runs/r3_selfplay_ramp_20261031/formal/r3_sp55_c000000/u003040',
        same_seed=True,same_budget_updates=640,sp_only=True,unix=time.time())
    if not (SP/'DEDICATED_CAPACITY_RESUME.json').exists():
        atomic_json(SP/'DEDICATED_CAPACITY_RESUME.json',request)
    first=SP/'formal/r3_sp55_c000000/FIRST_FORMAL_UPDATE.json'
    if first.exists():
        assert read(first)['status']=='PASS'
        atomic_json(OUT/'HANDOFF_COMPLETE.json',dict(status='FIRST_FORMAL_UPDATE_ACCEPTED',
            evidence=evidence,first_formal=str(first),first_formal_sha256=sha(first),unix=time.time()))
        atomic_json(OUT/'STATUS.json',dict(status='HANDOFF_COMPLETE_FIRST_FORMAL_ACCEPTED',
            selected=decision['pause_models'],exclusive_gpu=decision['exclusive_gpu'],
            first_formal=str(first),unix=time.time()))


def write_report():
    protocol=read(OUT/'PROTOCOL.json')
    decision=read(OUT/'SELECTION.json',{})
    status=read(OUT/'STATUS.json',{})
    lines=['# R3-sp 独占卡交接（20261031）','',
        '授权 `.hermes/STEER_capacity_free_gpu_for_sp_20261031_1140.md`；仅改变资源编排，不改奖励、超参、seed、640更新预算、网页或发布。',
        '冻结比较 `runs/r3_sp_capacity_20261031/PROTOCOL.json`；状态 `'+status.get('status','PENDING')+'`。',
        '生产JS greedy、老师关闭、bun-native-v2、本侧出生/复活，新holdout seed2026317200；hard与协作hard各128完整局、P0/P1各64。R3-a u3040已完成的同协议256局逐文件验SHA复用。',
        'P1两个条件WL合计差距主排序；不是把不同训练代数或不同训练来源当配对训练实验。R2-c尚未切本侧训练，但这里也在同一本侧清单评估。',
        '', '|模型|完整checkpoint|hard P1 WL|协作hard P1 WL|P1合计|两条件P1自伤|',
        '|---|---|---:|---:|---:|---:|']
    sources={'parent':protocol['parent']['source'],**{k:v['source'] for k,v in protocol['models'].items()}}
    for key,source in sources.items():
        completed=read(OUT/'evaluation'/key/'COMPLETE.json',{})
        metrics=completed.get('metrics',{})
        if metrics:
            a=metrics['hunter_hard']['P1'];b=metrics['coop_hard']['P1']
            values=[a['win_minus_loss'],b['win_minus_loss'],a['win_minus_loss']+b['win_minus_loss'],a['self_deaths']+b['self_deaths']]
        else:values=['PENDING']*4
        lines.append('|'+key+'|`'+source['checkpoint']+'`|'+'|'.join(map(str,values))+'|')
    if decision:
        lines+=['','选择：`'+str(decision['pause_models'])+'`；目标独占GPU `'+str(decision['exclusive_gpu'])+'`。',
                '逐条件双席W/D/L、样本数、自伤与差距保存在 `SELECTION.json`，不把缺失结果当通过。']
        if decision['status']!='SELECTED':
            lines+=['','R3-sp整卡交接阻塞：最弱线与受保护BT slot07共用GPU7；独占GPU3/4的R2-a/b均未落后父点超过20，无法按当前授权暂停第二条以释放整卡。仅为最弱线登记完整chunk末暂停；不移动、不暂停BT07或其他R2，不重试共卡。R3-sp正式更新仍0，暂无独占卡号、首正式SHA或交卡后吞吐验收。阻塞收据 `runs/r3_sp_capacity_20261031/BLOCKED.json`。']
    lines+=['','暂停与资源交接：只等选中线当前完整640更新chunk终点，完整Actor/Critic/Adam/env/aux/RNG/ring、state/policy SHA及next-step token通过且worker退出后才标PAUSED_COMPLETE_CHUNK；阻止其后继。GPU驱动无compute app后由原SP唯一owner接管，先u3040跨卡完整恢复，再计入预算smoke、u3042 fresh restore及首正式u3043。',
            '暂停收据 `runs/bun_native_v2_20261030/capacity_pauses/`；完整state保留，可按收据原协议/终点恢复。没有发训练信号、没有删除checkpoint，也没有恢复旧O5/储备。']
    for key in decision.get('pause_models',[]):
        v=protocol['models'][key];entry=v['entry'];folder=R2/'formal'/entry['run']
        from scripts.r3_selfplay_owner_20261031 import tail_rows
        rows=tail_rows(folder/'updates.jsonl',1)
        if rows:
            last=rows[-1];left=max(0,v['target_chunk_end']-last['new_update'])
            lines.append(f"\n{key} 当前u{last['new_update']}，完整chunk终点u{v['target_chunk_end']}，剩余{left}更新；按最新周期估算{left*last['cycle_seconds']/60:.1f}分钟，非完成承诺。")
        receipt=read(R2/'capacity_pauses'/f'{key.removeprefix("r2_")}.json',{})
        if receipt:lines.append('\n暂停实证：`'+receipt['source']['checkpoint']+'`，state SHA `'+receipt['source']['state_sha256']+'`，policy SHA `'+receipt['source']['policy_sha256']+'`。')
    first=read(SP/'formal/r3_sp55_c000000/FIRST_FORMAL_UPDATE.json',{})
    if first:
        cm=first['source'];lines+=['','首正式更新：u'+str(cm['update'])+'，finite、Actor/Critic非零，state SHA `'+cm['state_sha256']+'`，policy SHA `'+cm['policy_sha256']+'`，env/s '+str(first['env_per_second'])+'。']
    else:lines+=['','R3-sp首正式更新尚未验收；不把CPU比较、launch、跨卡恢复或无更新监控基线计作正式训练。']
    lines+=['','参照吞吐门槛保持原合同：暖态下降>5%跨三个独立5更新窗口，仅新SP原子暂停。资源守护原始前后值在 `runs/r3_selfplay_ramp_20261031/RESOURCE_GUARD.json`，首正式前不能宣布整卡吞吐验收完成。',
        '', '## BT slot03/04（仅结论，不执行）','',
        'slot03：继续保持BLOCK，保留末state，不自动回退或重启。依据编排方复算c37末段dev greedy P0 WL+16/P1+23、优于slot00，尚无依据按弱线淘汰；等编排方处理阻塞原因。',
        'slot04：当前继续保持BLOCK；建议保留末state后正式停掉，等编排方决定，本任务不实施。依据编排方复算c39近6点P0 WL+17降至+6、平局22/32；小样本趋势不等于单项机制因果证明。',
        '未修改BT slot00/01/03/04/05/06/07配置、原R3位置、网页模型或解码方式；未push/发布。']
    report=Path('reports/r3_sp_dedicated_capacity_20261031.md')
    temp=report.with_suffix('.md.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(report)


def watchdog():
    from scripts.bun_native_v2_owner import processes
    guard=lock(OUT/'WATCHDOG.lock')
    while not (OUT/'HANDOFF_COMPLETE.json').exists():
        isolate_cpu_comparison()
        write_report()
        decision=read(OUT/'SELECTION.json',{})
        if decision.get('status')=='SELECTED':advance_handoff(read(OUT/'PROTOCOL.json'),decision)
        owners=[p for p in processes('scripts.r3_sp_capacity_20261031')
                if b'--watchdog' not in (p/'cmdline').read_bytes().split(b'\0')]
        paused=any((root/name).exists() for root in (SP,R2,Path('runs/bomb_threat_v1'),Path('runs/iter_pbt_v1'))
                   for name in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))
        if not owners and not paused and not (OUT/'BLOCKED.json').exists():
            with (OUT/'OWNER.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-m','scripts.r3_sp_capacity_20261031'],
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(OUT/'WATCHDOG_RESTART.json',dict(status='CPU_OWNER_RESTARTED',
                completed_shards_reused=True,direct_training_launches=0,unix=time.time()))
        time.sleep(10)
    write_report()


def isolate_cpu_comparison():
    """Change only this evaluation's CPU placement; never touch trainers."""
    amendment=read(OUT/'CPU_SCHEDULING_AMENDMENT01.json',{})
    if amendment.get('status')!='AUTHORIZED_SCHEDULING_ONLY':return
    old=amendment['previous_cores'];new=amendment['cores']
    assert len(old)==len(new)==4
    changes=0
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            args=(process/'cmdline').read_bytes().split(b'\0')
            if b'scripts/r3_items_candidate_eval.js' not in args:continue
            if not any(a.startswith(str(OUT/'evaluation').encode()+b'/') for a in args):continue
            current=os.sched_getaffinity(int(process.name))
            if len(current)!=1 or next(iter(current)) not in old:continue
            target={new[old.index(next(iter(current)))]}
            for task in (process/'task').iterdir():
                try:os.sched_setaffinity(int(task.name),target)
                except ProcessLookupError:pass
            changes+=1
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    if changes:
        atomic_json(OUT/'CPU_PLACEMENT_ACCEPTANCE.json',dict(status='PASS_PLACEMENT_ONLY',
            evaluator_processes_relocated=changes,cores=new,workers=4,
            original_training_and_evaluators_untouched=True,
            protocol_sha256=sha(OUT/'PROTOCOL.json'),amendment_sha256=sha(OUT/'CPU_SCHEDULING_AMENDMENT01.json'),unix=time.time()))


if __name__=='__main__':
    try:
        if '--watchdog' in sys.argv:watchdog()
        else:main()
    except Exception as error:
        atomic_json(OUT/'BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,
            reason=str(error),training_signals_sent=False,unix=time.time()))
        subprocess.run(['jdme-notify','send','R3-sp整卡交接技术阻塞：'+type(error).__name__+
            '；runs/r3_sp_capacity_20261031/BLOCKED.json已落盘，未向训练worker发信号。'],capture_output=True)
        raise
