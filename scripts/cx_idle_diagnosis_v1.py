"""Bounded, CPU-slot-coordinated offline stationary decomposition; no training imports."""
import concurrent.futures
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from scripts import cc_strong_eval_v1 as E
from scripts.model_launch import manifest, command, public_project
from scripts.population_run_state import atomic_json, sha

OUT=Path('runs/cc_train_20261105/cx_takeover_20261106/ftn_diagnosis/idle_v1')
REPORT=Path('reports/cx_ftn_idle_20261106.md')
LABELS=['L2_S05g_u019302','L2_FTN_u019702','L2_ATK_u020022','L2_SP_u019862','L2_S05h_u020822']

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'PROTOCOL.json'
    if path.exists():
        p=json.loads(path.read_text())
        assert p['observer_sha256']==sha('scripts/cx_idle_replay_v1.js'), 'observer changed: use new namespace'
        return p
    sources=[];rows=[]
    for label in LABELS:
        for suffix in ('','__P1'):
            f=E.OUT/'fountain_metrics_v1/pairs'/f'{label}__vs__G_frozen{suffix}'/'GAMES.jsonl'
            sources.append(dict(file=str(f),sha256=sha(f)))
            rs=[json.loads(s) for s in f.read_text().splitlines()]
            assert len(rs)==88 and all(r['r3_chain_check'] for r in rs)
            rows+=rs
    assert len(rows)==880
    p=dict(status='FROZEN',unix=time.time(),labels=LABELS,files=[r['file'] for r in rows],sources=sources,
           reviews={r['file']:r['review_sha256'] for r in rows},observer_sha256=sha('scripts/cx_idle_replay_v1.js'),
           simulator_sha256=sha('web/sim-bun-native-v2.js'),games=880,CPU_only=True,shared_slots=24,max_replay_workers=6,
           labels_frozen_at_start=True,classification='conservative current-existing-hazard certificate; not a safe expert')
    atomic_json(path,p);return p

def shard(i,files):
    dest=OUT/f'chunk_{i:03d}.jsonl'
    if dest.exists():
        rows=[json.loads(s) for s in dest.read_text().splitlines()]
        if len(rows)==len(files) and not any('error' in r for r in rows):return len(rows)
    with E.Slot() as cpu:
        cfg=OUT/f'input_{i:03d}.json';atomic_json(cfg,dict(files=files,output=str(dest)))
        generic=manifest(dict(script='scripts/cx_idle_replay_v1.js',args=[str(cfg)],env=E.CPU_ENV))
        with (OUT/f'log_{i:03d}.txt').open('a') as log:
            r=subprocess.run(['taskset','-c',str(cpu)]+command('evaluation',generic,node=True),cwd=public_project(),env=dict(os.environ,**E.CPU_ENV),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        assert r.returncode==0,(i,r.returncode)
    return len(files)

def aggregate(p):
    rows=[]
    for f in sorted(OUT.glob('chunk_*.jsonl')):rows += [json.loads(s) for s in f.read_text().splitlines()]
    assert len(rows)==880 and len({r['file'] for r in rows})==880
    groups=defaultdict(list); correlations=defaultdict(list);index={}
    for r in rows:
        assert r['review_sha256']==p['reviews'][r['file']]==sha(r['file']) and r['r3_chain_check'] and r['hash_checks_verified']>0
        mine=r['idle'][r['seat']]; groups[f"{r['label']}/{r['mode']}/P{r['seat']}"].append(r)
        result=json.loads((Path(r['file']).parent/'RESULT.json').read_text());pick=result['r3']['pickups'][r['seat']]
        fountain=r['players'][r['seat']]
        for attr in (0,1):correlations[f"{r['label']}/{r['mode']}/{'capacity' if attr==0 else 'blast'}/{'yes' if pick[attr] else 'no'}"].append(dict(pickups=pick[attr],streak=fountain['fountain_streak'],strict_streak=fountain['strict_fountain_streak'],ticks=r['ticks'],c=mine['categories']['c']))
        index[r['review_sha256']]=dict(file=r['file'],idle=r['idle'],hash_checks_verified=r['hash_checks_verified'])
    output={}
    for key,rs in sorted(groups.items()):
        ms=[r['idle'][r['seat']] for r in rs];sums={k:sum(m[k] for m in ms) for k in ('ticks','alive','actionable','stationary','stationary_alive','stationary_actionable','idle_action','capacity_ticks','blast_ticks')}
        hist={k:dict(sum((Counter(m[k]) for m in ms),Counter())) for k in ('categories','reasons','capacity_hist','blast_hist','c_capacity_hist','c_blast_hist')}
        output[key]=dict(games=len(rs),sums=sums,**hist,
            stationary_rate=sums['stationary']/sums['ticks'],alive_stationary_rate=sums['stationary_alive']/sums['alive'],actionable_stationary_rate=sums['stationary_actionable']/sums['actionable'],
            c_actionable_rate=hist['categories']['c']/sums['actionable'],
            longest={k:max(m[k] for m in ms) for k in ('max_stationary','max_stationary_alive','max_stationary_actionable','max_c_run')},
            mean_longest_actionable=sum(m['max_stationary_actionable'] for m in ms)/len(ms),
            fountain_streak_mean=sum(r['players'][r['seat']]['fountain_streak'] for r in rs)/len(rs),
            growth_amounts=[sum(m['pickups'][j] for m in ms) for j in range(3)],
            examples=sorted([dict(file=r['file'],seed=r['seed'],longest_c=r['idle'][r['seat']]['max_c_run'],examples=r['idle'][r['seat']]['examples']) for r in rs],key=lambda z:z['longest_c'],reverse=True)[:3])
    corr={k:dict(games=len(v),pickups=sum(x['pickups'] for x in v),mean_streak=sum(x['streak'] for x in v)/len(v),mean_strict_streak=sum(x['strict_streak'] for x in v)/len(v),mean_ticks=sum(x['ticks'] for x in v)/len(v),c_per_1000ticks=1000*sum(x['c'] for x in v)/sum(x['ticks'] for x in v)) for k,v in sorted(correlations.items())}
    result=dict(status='COMPLETE_880_HASH_VERIFIED',unix=time.time(),protocol_sha256=sha(OUT/'PROTOCOL.json'),games=880,groups=output,pickup_fountain_observational=corr)
    atomic_json(OUT/'SUMMARY.json',result);atomic_json(OUT/'REVIEW_IDLE_INDEX.json',index)
    lines=['# 发呆分解（20261106）','','五个冻结检查点各176局，共880局；既有固定评估seed，greedy/categorical、P0/P1各44局。所有录像重新运行生产模拟器，检查状态hash与胜负，交叉R3链指标。SP在冻结时最新完整点已到u019862，因此使用该点；后续新点不漂移进本批。','','静止=真实坐标L1变化<1e-8；全tick、存活和可行动（存活且未受控）分别作分母。动作idle与物理静止分开。最长段单位tick，分全/存活/可行动/c。a/b/c/d互斥且总和等于静止tick。','','保守优先级：死亡/受控→d；主动放泡或用道具→d；自有在场泡达容量→a；携包→d保护；特殊移动→d；b只认当前位置安全、推进路径均跨现有火线的等待；c需余泡>0、敌或砖攻击站位BFS可达、生产_steer存在严格缩短距离且整个移动扫掠避开现有危险。未知未来敌新泡不在证据内，c是当前危险下安全推进的保守下界，不能直接冒充native安全专家。火线按双方所有现有泡射线并集，忽略可炸砖/泡遮挡只保留墙，覆盖连锁后障碍消失；无法证明的情况进入d，不罚合法避险。','','## 分模式、分座统计','','| 点/模式/座 | 静止% 全/存活/可行动 | 最长段 全/存活/可行动/c | a/b/c/d占静止% | c占可行动% | 最长可行动段均值 |','|---|---|---|---|---|---|']
    for key,r in output.items():
        cats=r['categories'];den=r['sums']['stationary'];l=r['longest']
        lines.append(f"| {key} | {100*r['stationary_rate']:.2f}/{100*r['alive_stationary_rate']:.2f}/{100*r['actionable_stationary_rate']:.2f} | {'/'.join(str(l[k]) for k in ('max_stationary','max_stationary_alive','max_stationary_actionable','max_c_run'))} | {'/'.join(f'{100*cats[k]/den:.2f}' for k in 'abcd')} | {100*r['c_actionable_rate']:.2f} | {r['mean_longest_actionable']:.1f} |")
    lines+=['','## 静止时容量/火力（tick加权）','','| 点/模式/座 | 静止容量直方图 | 静止火力直方图 | c容量直方图 | c火力直方图 |','|---|---|---|---|---|']
    for key,r in output.items():lines.append('| '+key+' | '+' | '.join(json.dumps(r[k],sort_keys=True) for k in ('capacity_hist','blast_hist','c_capacity_hist','c_blast_hist'))+' |')
    lines+=['','## 吃道具与喷泉串：同局观察关联','','拾取次数来自原RESULT.r3.pickups（泡数/火力）；重放growth_amounts是属性增量，超级+5不等于5次拾取。按是否拾取分层，保留局长作为混杂提示；不作因果改善结论，未拾取组为空时不伪造零组。','','| 点/模式/属性/有无拾取 | 局数/拾取次数 | 宽/严格串均值 | 局长均值 | c/千tick |','|---|---|---|---|---|']
    for key,r in corr.items():lines.append(f"| {key} | {r['games']}/{r['pickups']} | {r['mean_streak']:.2f}/{r['mean_strict_streak']:.2f} | {r['mean_ticks']:.1f} | {r['c_per_1000ticks']:.2f} |")
    lines+=['','## 下一轮候选与防hack验收','','仅针对c的安全推进支持；保护movement、运包/道具，未标注只独立冻结源KL。不加放泡次数奖、存量tick奖或泛idle罚。诊断标签尚非训练专家，先对native selected安全与后状态做对拍。','','1. **冻结目标势函数**：在可核c状态选定敌/新砖安全威胁站位，使用有界Φ与γΦ(s′)−Φ(s)，目标在整段固定，终止Φ归零，整段按原γ折扣核账；危险/携包期间保留势函数状态、停付款不能任意重置免掉负回撤，目标切换也必须守恒。不按每个接近tick只付正项。防hack：A↔B往返、角落抖动、死亡重置、目标频繁切换和进入/离开c门循环，折扣总信用必须等于边界项；放泡无新威胁、空场和携包不得获净额外奖励。若门控破坏守恒则拒绝该实现。','','2. **一次安全威胁前沿事件**：仅c出发、native安全动作与后状态证实，让此前未进入安全威胁距离的真实敌/新砖首次进入；每episode对同一冻结敌阶段/砖ID最多一次，保留小总预算与延迟结算，死亡/受控待支付作废。敌自行走近不付（需自身可归因推进），敌移动/砖重生不得刷新同ID。防hack：反复进出距离阈值、同格绕圈、同tick/同批重报、敌静止无可达目标、空放泡、自己堵路后再解堵、携包路上偏航，全应0或不超过一次上限；用真实胜负/控制杀/交付/自伤验收。','','3. **一次净道具里程碑/离线安全示范**：优先复用原道具支付，若增加只对c起点可归因的首次真实泡容量/火力净增长，绑定出生道具身份与episode高水位，死亡掉落重拾不重复，满级溢出不付，超级道具算一次而非+5次。或不改奖励，用短离线示范对安全推进动作开支持，live行归一+稀疏单例权重封顶，冻结源KL。防hack：丢失重拾循环、重生刷点、假拾取、溢出、无关道具/携包冲突、稀疏正例从40→1时系数不得自动放大；暖身后native PPO、完整fresh与固定双座评测，评估无在线辅助。','','候选单因素、先smoke和完整restore再新namespace有界训练，不改现有四臂。当前四卡均健康且SP未完成固定预算，未抢占任何卡；等待其他臂安全预算边界或空闲卡，再按证据冻结单一候选。全量JSON/每局类别段与可回看c样例位于 '+str(OUT)+'。']
    REPORT.write_text('\n'.join(lines)+'\n')
    base=Path('reports/cx_ftn_diagnosis_20261106.md');text=base.read_text().split('\n<!-- IDLE_DECOMPOSITION -->')[0]
    base.write_text(text+'\n<!-- IDLE_DECOMPOSITION -->\n'+REPORT.read_text())
    return result

def main():
    p=freeze();atomic_json(OUT/'STATUS.json',dict(status='RUNNING',unix=time.time(),protocol_sha256=sha(OUT/'PROTOCOL.json'),games=880,completed=0))
    files=p['files'];done=0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            fut=[pool.submit(shard,i,files[k:k+11]) for i,k in enumerate(range(0,len(files),11))]
            for f in concurrent.futures.as_completed(fut):
                done+=f.result();atomic_json(OUT/'STATUS.json',dict(status='RUNNING',unix=time.time(),completed=done,games=880));print('VERIFIED',done,flush=True)
        r=aggregate(p);atomic_json(OUT/'STATUS.json',dict(status=r['status'],completed=880,games=880,unix=time.time()))
    except Exception as e:
        atomic_json(OUT/'STATUS.json',dict(status='FAILED_REUSABLE_SHARDS',error=f'{type(e).__name__}: {e}',completed=done,unix=time.time()));raise

if __name__=='__main__':main()
