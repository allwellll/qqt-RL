"""Passive frozen-terminal A3SR seat/funnel verdict; never launches training."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';B=BASE/'iter2/a3sr'


def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default


def request(cp,arm,split):
    p=Path(cp);name=f'{p.parent.name}_{p.name}'+('_holdout' if split=='holdout' else '')
    q=BASE/'eval_requests'/f'{name}.json'
    if not q.exists():atomic_json(q,dict(name=name,checkpoint=cp,split=split,rule_version='stock-window-v1',
        reward_version='bomb-threat-iter2-A3',iter2_arm=arm,reason='A3SR predeclared native terminal comparison'))
    return name


def metrics(cp,split):
    p=Path(cp);name=f'{p.parent.name}_{p.name}'+('_holdout' if split=='holdout' else '')
    out=BASE/'eval'/name
    if not (out/'NATURAL_COMPLETE.json').exists():return None
    from scripts.bomb_threat_gate_report import recompute_natural
    marker,m=recompute_natural(out)
    assert marker['policy_sha256']==read(ROOT/cp/'COMMITTED.json')['policy_sha256']
    seats=m['greedy']['by_seat'];funnel={s:dict(window_pickup=0,same_chain_delivery=0,games=0) for s in ('P0','P1')}
    for p in sorted(out.glob('natural_greedy_[0-9][0-9][0-9].json')):
        rows=read(p)['results']['natural_full2400']['episodes']
        for e in rows:
            x=funnel[f"P{e['team']}"];x['games']+=1
            x['window_pickup']+=e['stock_pickup_dead']+e['stock_pickup_native_sugar']
            x['same_chain_delivery']+=e['window_stock_delivery']
    assert all(x['games']==32 for x in funnel.values())
    return dict(by_seat=seats,funnel=funnel,source_sha256=sha(out/'NATURAL_COMPLETE.json'))


def compare(candidate,control):
    c=candidate['by_seat'];r=control['by_seat'];d={s:c[s]['win_minus_loss']-r[s]['win_minus_loss'] for s in ('P0','P1')}
    flags=dict(P1_win_minus_loss_gain_at_least5=d['P1']>=5,P0_decline_at_most3=d['P0']>=-3,
        single_zero_not_increased=all(c[s]['zero_bomb_games']<=r[s]['zero_bomb_games'] for s in ('P0','P1')),
        self_harm_not_increased=all(c[s]['self_deaths']<=r[s]['self_deaths'] for s in ('P0','P1')),
        P1_window_pickup_not_decreased=candidate['funnel']['P1']['window_pickup']>=control['funnel']['P1']['window_pickup'],
        P1_same_chain_delivery_not_decreased=candidate['funnel']['P1']['same_chain_delivery']>=control['funnel']['P1']['same_chain_delivery'])
    return dict(status='PASS' if all(flags.values()) else 'FAIL',conditions=flags,seat_win_minus_loss_difference=d)


def refresh():
    terminal=read(B/'TERMINAL.json',{})
    if not terminal:return
    start_brick_observation()
    decision=B/'VERDICT.json'
    if read(decision,{}).get('status') in ('PASS','FAIL'):return
    frozen=read(B/'EVALUATION_MANIFEST.json');candidate=terminal['last_atomic']['checkpoint']
    values={};missing=[]
    for label,cp in [('candidate',candidate)]+[(k,v['checkpoint']) for k,v in frozen['controls'].items()]:
        values[label]={}
        for split in ('dev','holdout'):
            request(cp,'A3SR' if label=='candidate' else 'A3S',split)
            m=metrics(cp,split)
            if m is None:missing.append(label+'_'+split)
            else:values[label][split]=m
    comparisons={}
    for label in frozen['controls']:
        comparisons[label]={}
        for split in ('dev','holdout'):
            if split in values['candidate'] and split in values[label]:comparisons[label][split]=compare(values['candidate'][split],values[label][split])
    complete=not missing
    status='PENDING' if not complete else 'PASS' if all(x['status']=='PASS' for arm in comparisons.values() for x in arm.values()) else 'FAIL'
    result=dict(status=status,run=terminal['run'],terminal=candidate,metrics=values,comparisons=comparisons,missing=missing,
        scope='fixed terminal native evaluation; compare each frozen A3S control and each split; no peak selection',
        limitations=['one A3SR seed; controls historical A3S1280 vs A3SR640 and different source Actors; not matched causal estimate',
            'JS self_deaths is attributed self-bomb death, not every possible safety cost','9 brick-separated games descriptive only'],unix=time.time())
    atomic_json(decision,result)
    lines=['# A3SR 固定640终态评估（20261030）','',f'状态：{status}。未完成：{", ".join(missing) or "无"}。',
        '', '固定两路A3S1280历史对照，逐split/seat比较；不择峰。一个A3SR seed且训练预算/起点不同，不作因果提升声明。',
        '', '|对照|split|判定|P0 W-L变化|P1 W-L变化|','|---|---|---|---:|---:|']
    for arm,splits in comparisons.items():
        for split,v in splits.items():lines.append(f'|{arm}|{split}|{v["status"]}|{v["seat_win_minus_loss_difference"]["P0"]}|{v["seat_win_minus_loss_difference"]["P1"]}|')
    lines+=['','完整原始指标、逐座位库存窗口取包/同链交付、未知/缺数据见`runs/bomb_threat_v1/iter2/a3sr/VERDICT.json`。未晋升、未改变已发布模型。']
    (ROOT/'reports/bomb_threat_v1_a3sr_terminal_20261030.md').write_text('\n'.join(lines)+'\n')
    if complete:
        atomic_json(BASE/'notifications'/f'A3SR_VERDICT_{status}.json',result)
        from scripts.iter_pbt_owner import notify
        notify('A3SR_VERDICT',f'A3SR固定640 dev+holdout分seat/窗口取包交付判定{status}，历史两路A3S对照分别列出；新模型未晋升。')


def start_brick_observation():
    """CPU diagnostics follow the terminal; completed games are never repeated."""
    import os,sys,subprocess
    out=B/'brick_observation'
    if (out/'COMPLETE.json').exists():return
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if b'scripts.bomb_threat_a3sr_brick_observation' in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return
        except OSError:pass
    out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.log').open('ab',buffering=0) as stream:
        subprocess.Popen(['taskset','-c','116-119',sys.executable,'-m','scripts.bomb_threat_a3sr_brick_observation'],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
