"""08:40 CPU-only fixed-candidate production-JS fresh comparison; no training/promotion."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',QQT_STOCK_RULE='stock-window-v1')
import json,sys,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bomb_threat_round15 import read
from scripts import bomb_threat_fresh256 as F
BASE=Path('runs/bomb_threat_v1');OUT=BASE/'fresh256_a3sr_vs_u800_20261030'
REPORT=Path('reports/bomb_threat_v1_fresh256_a3sr_20261030.md')
INSTRUCTION='.hermes/TASK_fresh256_a3sr_20261030_0840.md'
ROLES=('a3sr_u640','a3sr_u1280')


def freeze():
    path=OUT/'PROTOCOL.json'
    if path.exists():return read(path)
    oldpath=BASE/'fresh256_slot01_u800/PROTOCOL.json';old=read(oldpath)
    assert all(sha(Path(f))==h for f,h in old['source_sha256'].items() if f.endswith('.js'))
    sources=dict(a3sr_u640=('bt_slot05_c000026/u000640','bt_slot05_c000026_u000640_holdout'),
                 a3sr_u1280=('bt_slot05_c000028/u001280','bt_slot05_c000028_u001280'))
    models={'u800':old['models']['candidate']}
    for role,(cpname,evalname) in sources.items():
        cp=BASE/'formal'/cpname;cm=read(cp/'COMMITTED.json')
        assert sha(cp/'policy.pt')==cm['policy_sha256']
        # Identity is the new chunk's source, not a later running checkpoint.
        if role=='a3sr_u1280':
            new=read(BASE/'protocols/bt_slot05_c000029.json')
            assert new['continuation_source']['checkpoint']==str(cp)
            assert new['continuation_source']['policy_sha256']==cm['policy_sha256']
        model=BASE/'eval'/evalname/'model/model.json'
        assert model.exists(),f'missing previously verified export: {role}'
        models[role]=dict(checkpoint=str(cp),policy_sha256=cm['policy_sha256'],model_json=str(model),model_sha256=sha(model))
    files=[f for f in old['source_sha256'] if f.endswith('.js')]+['scripts/bomb_threat_fresh256.py',
        'scripts/bomb_threat_a3sr_u800_compare.py']
    proto=dict(id='A3SR-U640-U1280-VS-U800-FRESH256-20261030',status='FROZEN',instruction=INSTRUCTION,
        instruction_sha256=sha(Path(INSTRUCTION)),reference_protocol=str(oldpath),reference_protocol_sha256=sha(oldpath),
        models=models,tiers=old['tiers'],source_sha256={f:sha(Path(f)) for f in files},
        candidates=list(ROLES),modes=['greedy'],games_per_tier=256,seats=dict(P0=128,P1=128),
        total_new_games=1536,baseline_games_reused=768,teacher=False,engine='production_JS',
        opponent='production_JS_Hunter',rule_version='stock-window-v1',horizon=2400,native_bricks=True,
        workers=4,CPU_affinity='116-119',GPU_use=False,no_promotion=True,no_publication=True,
        comparison='P1 hard/normal/easy absolute W-L and matched difference vsu800; no peak selection',
        limitations=['64 frozen birth geometries cyclic reuse, 256 environment/action seeds per tier',
                    'existing route metrics lack per-cargo terminal cause and window-instance matching; unknown causes stay unknown'],unix=time.time())
    atomic_json(OUT/'AMENDMENT01_IDENTITY_AND_METRICS.json',dict(status='FROZEN',instruction=INSTRUCTION,
        candidate2='c29 startu1280 maps exactly to slot05c28/u1280 continuation_source; later points excluded',
        same_seed_requirement='u800 seeds2026142900/2026143900/2026144900; old A3SR fresh seeds differ and are not reused',
        baseline='all768 old u800 argmax games reused without replay',
        funnel='window/stock-pickup/same-cargo delivery exact counters; return_started covers all pickup sources; per-cargo death/drop/timeout causes unavailable, reported unknown',unix=time.time()))
    atomic_json(path,proto)
    atomic_json(OUT/'TASK_CONSUMED.json',dict(status='FROZEN_READY',protocol_sha256=sha(path),unix=time.time()))
    return proto


def jobs():
    return [dict(tier=t,role=r,mode='greedy',start=i) for t in F.TIERS for i in range(0,256,8) for r in ROLES]


def path_for(job):return OUT/job['tier']/job['role']/f"greedy_{job['start']:03d}.json"


def collect(job,proto):
    p=path_for(job);p.parent.mkdir(parents=True,exist_ok=True);proof=Path(str(p)+'.ACCEPTED.json')
    if proof.exists():
        for f,h in read(proof)['files'].items():assert sha(Path(f))==h
        F.validate_shard(p,job,proto);return
    if not(p.exists() and Path(str(p)+'.outcome.json').exists()):
        env=dict(os.environ,BT_FRESH_TIER=job['tier'],JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='256',
            JE10_ROUTE_START=str(job['start']),JE10_ROUTE_END=str(job['start']+8),JE10_ROUTE_SEED_START=str(proto['tiers'][job['tier']]['seed']))
        with p.with_suffix('.log').open('ab') as log:
            subprocess.run(['node','scripts/bomb_threat_fresh256_eval.js','greedy',proto['models'][job['role']]['model_json'],str(p),
                'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',proto['tiers'][job['tier']]['spawn_manifest']],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    F.validate_shard(p,job,proto)
    atomic_json(proof,dict(job=job,protocol_sha256=sha(OUT/'PROTOCOL.json'),
        files={str(f):sha(f) for f in (p,Path(str(p)+'.outcome.json'))},unix=time.time()))


def load_role(tier,role,proto):
    from scripts.bomb_threat_eval import summarize
    rows=[];episodes=[];files={}
    for start in range(0,256,8):
        job=dict(tier=tier,role='candidate' if role=='u800' else role,mode='greedy',start=start)
        if role=='u800':
            p=BASE/'fresh256_slot01_u800'/tier/'candidate'/f'greedy_{start:03d}.json'
            pp=read(BASE/'fresh256_slot01_u800/PROTOCOL.json')
        else:
            p=path_for(job);pp=proto
            if not Path(str(p)+'.ACCEPTED.json').exists():continue
        rr=F.validate_shard(p,job,pp);ee=read(p)['results']['natural_full2400']['episodes']
        rows+=rr;episodes+=ee
        files.update({str(f):sha(f) for f in (p,Path(str(p)+'.outcome.json'))})
    if not rows:return dict(games=0,files=files)
    assert len({r['seed'] for r in rows})==len(rows)
    if len(rows)==256:assert {r['seed'] for r in rows}==set(range(proto['tiers'][tier]['seed'],proto['tiers'][tier]['seed']+256))
    value=summarize(rows)
    def funnel(ep):
        keys=('window_events','window_stock_pickup','return_started','window_stock_delivery',
            'stock_pickup_dead','stock_pickup_native_sugar','nonstock_pickup','nonstock_delivery','death','self_death','pickup')
        sums={k:sum(e.get(k,0) for e in ep) for k in keys}
        sums['window_stock_pickup_games']=sum(e.get('window_stock_pickup',0)>0 for e in ep)
        sums['window_stock_delivery_games']=sum(e.get('window_stock_delivery',0)>0 for e in ep)
        sums['window_seen_no_stock_pickup_games']=sum(e['window_events']>0 and e['window_stock_pickup']==0 for e in ep)
        sums['stock_pickup_no_same_chain_delivery_games']=sum(e['window_stock_pickup']>0 and e['window_stock_delivery']==0 for e in ep)
        sums['stock_pickup_no_delivery_and_any_death_games']=sum(e['window_stock_pickup']>0 and e['window_stock_delivery']==0 and e['death']>0 for e in ep)
        sums['observed_C_RETURN_ticks']=sum(e.get('stage_funnel',{}).get('C_RETURN',{}).get('target_ticks',0) for e in ep)
        sums['first_deviation_phase_counts']=dict(Counter(e['first_deviation']['phase'] for e in ep if e.get('first_deviation')))
        sums['window_specific_return_start']=None
        sums['per_cargo_death_drop_timeout_failure_reason']=None
        sums['limitations']='window event/episode counts are not matched window-instance denominators; any same-game death is observational, not identified cargo-loss cause; oracle deviation is a diagnostic label, not Q'
        return sums
    value['funnel']=funnel(episodes)
    for seat in (0,1):value['by_seat'][f'P{seat}']['funnel']=funnel([e for e in episodes if e['team']==seat])
    value['files']=files
    value['episode_pairing']=[dict(seed=r['seed'],seat=r['seat'],winner=r['winner']) for r in rows]
    return value


def publish(proto):
    values={t:{r:load_role(t,r,proto) for r in (*ROLES,'u800')} for t in F.TIERS}
    newgames=sum(values[t][r]['games'] for t in F.TIERS for r in ROLES)
    done=newgames==1536
    result=dict(status='COMPLETE' if done else 'RUNNING',completed_new_games=newgames,total_new_games=1536,
        reused_u800_games=768,metrics=values,protocol_sha256=sha(OUT/'PROTOCOL.json'),
        no_promotion=True,no_publication=True,unix=time.time())
    atomic_json(OUT/'RESULTS.json',result)
    text=['# A3SR fresh256 与线上u800同口径对照 — 20261030','',
        f"状态 **{result['status']}**，新增候选完整局{newgames}/1536；u800直接复用768局。训练不停，不改奖励/闸门/BEST/网页。",'',
        '固定候选：A3SR u640固定预算末点；slot05 c29源原子点c28/u1280。未选择u840/u880等峰值。两个候选使用u800原fresh seed清单：hard2026142900、normal2026143900、easy2026144900，各连续256；P0/P1各128，argmax、带砖、2400tick、stock-window-v1、生产JS Hunter、teacher off。原u640其他fresh结果使用不同seed，只保留历史，不冒充本次配对证据。','',
        f"冻结协议 `{OUT}/PROTOCOL.json`；身份及指标解释 `{OUT}/AMENDMENT01_IDENTITY_AND_METRICS.json`。原始文件与SHA、逐局seed/seat/winner对照在RESULTS.json。",'',
        '| 档位 | 模型 | 座位 | n | W/D/L | W−L | 因果击杀 | 泡/局 | 单方零泡 | 结构双零 | 自伤 |','|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for tier in F.TIERS:
        for role in ('u800',*ROLES):
            v=values[tier][role]
            if not v['games']:continue
            for seat,g in [('all',v),*v['by_seat'].items()]:
                text.append(f"|{tier}|{role}|{seat}|{g['games']}|{g['win']}/{g['draw']}/{g['loss']}|{g['win_minus_loss']:+d}|{g['causal_kills']}|{g['bombs_per_game']:.3f}|{g['zero_bomb_games']}|{g['structural_zero_bomb_games']}|{g['self_deaths']}|")
    text+=['','## P1网页场景对照','',
        '真人对局模型固定P1。未齐128例的座位行仅作进度，不得判断全面不差或胜负差至少+10；即使完整256也仅覆盖冻结地图与Hunter，不证明真人对抗泛化。','',
        '| 候选 | 档位 | 候选P1 W−L | u800 P1 W−L | 配对集合胜负差变化 | P1绝对≥+10 |','|---|---|---:|---:|---:|---|']
    for role in ROLES:
        for tier in F.TIERS:
            c=values[tier][role];b=values[tier]['u800']['by_seat']['P1']
            c=c.get('by_seat',{}).get('P1',{})
            full=c.get('games')==128
            text.append(f"|{role}|{tier}|{c.get('win_minus_loss','MISSING')}|{b['win_minus_loss']}|{c['win_minus_loss']-b['win_minus_loss'] if full else 'MISSING'}|{c['win_minus_loss']>=10 if full else 'MISSING'}|")
    text+=['','## 运包分环节与失败证据','',
        '窗口→窗口内库存取包→返程→同来源交付分别记录。窗口事件数与取包次数并非匹配的一一窗口实例，不把比值冒充成功概率。window_stock_delivery由原生取包因果账本绑定同次取得的包。return_started覆盖所有来源取包，无法从已有u800分片独立识别“窗口来源专属返程起点”；C_RETURN ticks是暴露，不是完成。失败按已观测环节缺口列出；同局死亡仅作共现，不能据此归因死后丢包；逐cargo死亡/掉包/超时等原因缺记录的部分明确unknown。','',
        '| 档位/模型/seat | 窗口事件 | 窗口库存取包 | 返程开始（全来源） | 返程ticks | 同链交付 | 有窗口未取包局 | 取包未同链交付局 | 后者同局有死亡 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for tier in F.TIERS:
        for role in ('u800',*ROLES):
            v=values[tier][role]
            if not v['games']:continue
            for seat,g in [('all',v),*v['by_seat'].items()]:
                f=g['funnel'];keys=('window_events','window_stock_pickup','return_started','observed_C_RETURN_ticks','window_stock_delivery','window_seen_no_stock_pickup_games','stock_pickup_no_same_chain_delivery_games','stock_pickup_no_delivery_and_any_death_games')
                text.append(f"|{tier}/{role}/{seat}|"+'|'.join(str(f[k]) for k in keys)+'|')
    text+=['','结论：'+('全部候选固定样本已完成；各档/座位逐项比较以表中数值为准，任何下降不以别的档位补偿。' if done else '尚未完成固定样本，当前不判断候选全面不差或P1至少+10。'),'',
        '复现 `.venv/bin/python -m scripts.bomb_threat_a3sr_u800_compare`；resume自动核验ACCEPTED与文件SHA，只补未完成8局分片。', '']
    REPORT.write_text('\n'.join(text))
    atomic_json(OUT/'EXECUTION_STATUS.json',dict(status=result['status'],completed_new_games=newgames,total_new_games=1536,
        report=str(REPORT),report_sha256=sha(REPORT),CPU_only=True,unix=time.time()))
    if done:
        atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',result_sha256=sha(OUT/'RESULTS.json'),
            protocol_sha256=sha(OUT/'PROTOCOL.json'),new_games=1536,baseline_reused=768,unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('A3SR_U800_MATCHED_FRESH256_COMPLETE','A3SR u640/固定u1280与线上u800同seed生产JS fresh256完成；1536候选新局、768基线复用，P1三档和运包环节报告已落盘；未晋升未发布。',report=str(REPORT))
    return result


def main():
    proto=freeze();guard=lock(OUT/'OWNER.lock')
    for f,h in proto['source_sha256'].items():assert sha(Path(f))==h,f
    for m in proto['models'].values():assert sha(Path(m['model_json']))==m['model_sha256']
    if (OUT/'COMPLETE.json').exists():return
    publish(proto)
    atomic_json(OUT/'LAUNCH.json',dict(status='CPU_RUNNING',workers=4,CPU_affinity=sorted(os.sched_getaffinity(0)),
        training_owners_created=0,protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(collect,j,proto):j for j in jobs()}
            count=0
            for future in as_completed(futures):
                future.result();count+=1
                atomic_json(OUT/'PROGRESS.json',dict(completed_shards=count,total_shards=192,last_job=futures[future],unix=time.time()))
                if count==1 or count%4==0:publish(proto)
        publish(proto)
    except Exception as error:
        atomic_json(OUT/'BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,
            reason=str(error),training_untouched=True,unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('A3SR_U800_MATCHED_FRESH256_BLOCKED',f'A3SR-u800同口径fresh256技术阻塞：{type(error).__name__}，完整分片保留，训练不停。')
        raise

if __name__=='__main__':main()
