"""Audit completed production-rule matches and hand optional specialists to web owner."""
import json,time,os,re
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.cx_items_candidates_v1 import ROOT,REPORT,BOTS,config,CPU

def paired_intervals(rows,label,mode,reference,kind):
    import numpy as np
    from scripts.corner_counterplay_stage_report_v1 import measures
    def choose(model,decision):
        return {r['seed']:r for r in rows if r['candidate']==model and
                r['mode']==('greedy' if decision=='argmax' else 'tau03') and
                r['target']==kind and r['split']=='holdout'}
    a=choose(reference,'tau0.3');b=choose(label,mode);assert set(a)==set(b)
    seeds=sorted(a);rng=np.random.default_rng(2026799099);ids=rng.integers(0,len(seeds),(10000,len(seeds)))
    values={}
    for metric in ('causal_kills','deliveries','self_deaths'):
        d=np.array([measures(b[s])[metric]-measures(a[s])[metric] for s in seeds],float)
        boot=d[ids].mean(-1);values[metric]=dict(delta_per_game=float(d.mean()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)))
    return dict(reference=reference+' tau0.3',paired_games=len(seeds),bootstrap_seed=2026799099,intervals=values,
        interpretation='descriptive paired fixed-seed bootstrap; selection and multi-metric exploration prevent global significance claim')

def finish(p):
    screen=json.loads((ROOT/'screen_SUMMARY.json').read_text())
    conf=json.loads((ROOT/'confirm_SUMMARY.json').read_text())
    web=json.loads((ROOT/'web_SUMMARY.json').read_text())
    inventory=[]
    for phase,value in [('screen',screen),('confirm',conf),('web',web)]:
        assert len(value['rows'])=={'screen':216,'confirm':320,'web':320}[phase]
        assert len({(r['candidate'],r['mode'],r['target'],r['split'],r['seed']) for r in value['rows']})==len(value['rows'])
        for key,stats in value['groups'].items():
            expected=8 if phase=='web' else (4 if '/strong_full/' in key else 8) if phase=='screen' else (8 if '/strong_full/' in key else 16)
            assert stats['games']==expected,(phase,key,stats['games'],expected)
        for r in value['rows']:
            assert r['ruleset']=='bun-native-v3-items' and r['simulator_sha256']==p['engine_sha256']
            assert r['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
            assert r['review_verified'] and r['extra_replay_hash_checks']>0
        for f in (ROOT/phase).glob('**/COMPLETE.json'):
            for r in json.loads(f.read_text())['results']:
                folder=f.parent/f"match-{r['seed']}-P{r['seat']}-{r['mode']}-{r['opponent_mode']}"
                assert json.loads((folder/'RESULT.json').read_text())==r
                assert sha(folder/'review.json')==r['review_sha256']
                rv=json.loads((folder/'review.json').read_text())
                assert rv['meta']['rule']==p['rule'] and rv['meta']['simulator_sha256']==p['engine_sha256']
                inventory.append(dict(phase=phase,result=str(folder/'RESULT.json'),review=str(folder/'review.json'),review_sha256=r['review_sha256']))
    atomic_json(ROOT/'ARTIFACT_AUDIT.json',dict(status='PASS_ALL_PRODUCTION_PHYSICS_RESULT_REPLAY_IDENTITIES',games=len(inventory),inventory=inventory))
    selection=json.loads((ROOT/'SELECTION.json').read_text())
    selected=selection['candidates']
    scores=selection['all_screen']
    representatives={
        '击杀':max(scores,key=lambda x:(x['kill'],x['net'])),
        '进攻流畅':min(scores,key=lambda x:x['stationary']),
        '喷泉':max(scores,key=lambda x:x['fountain']),
        '运包':max(scores,key=lambda x:(x['deliveries'],x['net'])),
    }
    online=p['references'][0]['label'];health=p['references'][1]['label']
    groups=conf['groups'];wg=web['groups'];recommend=[];checks=[]
    def c(label,mode,kind,split):return groups[f'{label}/{mode}/{kind}/{split}']
    def w(label,mode,bot):return wg[f'{label}/{mode}/{bot}/web']
    for item in selected:
        m=item['model'];label=m['label'];choices=[]
        for mode in ('argmax','tau0.3'):
            dh=c(label,mode,'duel','holdout');dd=c(label,mode,'duel','dev');fh=c(label,mode,'strong_full','holdout')
            refs={ref:dict(dh=c(ref,'tau0.3','duel','holdout'),dd=c(ref,'tau0.3','duel','dev'),fh=c(ref,'tau0.3','strong_full','holdout')) for ref in (online,health)}
            strengths=[];detail=[]
            for ref,rv in refs.items():
                # A selectable specialist needs repeated, measurable behavior, not global dominance.
                kill=dh['causal_kills']>rv['dh']['causal_kills'] and dd['causal_kills']>rv['dd']['causal_kills']
                killsweb=sum(w(label,mode,b)['causal_kills'] for b in BOTS)
                refkills=sum(w(ref,'tau0.3',b)['causal_kills'] for b in BOTS)
                transport=fh['deliveries']>rv['fh']['deliveries'] and sum(w(label,mode,b)['deliveries'] for b in BOTS)>sum(w(ref,'tau0.3',b)['deliveries'] for b in BOTS)
                fountain=fh['fountain_streak']>rv['fh']['fountain_streak'] and sum(w(label,mode,b)['fountain_streak'] for b in BOTS)>sum(w(ref,'tau0.3',b)['fountain_streak'] for b in BOTS)
                flow=fh['actionable_stationary_pct']<rv['fh']['actionable_stationary_pct'] and fh['mean_actionable_longest']<rv['fh']['mean_actionable_longest'] and sum(w(label,mode,b)['actionable_stationary_pct'] for b in BOTS)<sum(w(ref,'tau0.3',b)['actionable_stationary_pct'] for b in BOTS)
                if kill:strengths.append('独立dev/holdout人工短局真实因果击杀均高于'+ref+' τ0.3')
                if transport:strengths.append('独立完整局holdout及网页五对手同批交付均高于'+ref+' τ0.3')
                if fountain:strengths.append('独立完整局holdout及网页五对手有效喷泉串均高于'+ref+' τ0.3')
                if flow:strengths.append('独立完整局holdout静止占比/平均最长段更短，网页静止占比也低于'+ref+' τ0.3')
                detail.append(dict(reference=ref,kill_dev_holdout=kill,web_causal_kills=[killsweb,refkills],transport_holdout_and_web=transport,fountain_holdout_and_web=fountain,flow_holdout_and_web=flow))
            # Prefer directly established killing/transport; death remains descriptive.
            key=(any(x['kill_dev_holdout'] for x in detail),any(x['transport_holdout_and_web'] for x in detail),len(strengths),dh['causal_kills'],fh['deliveries'],fh['score'])
            choices.append(dict(mode=mode,strengths=strengths,checks=detail,key=key,duel_holdout=dh,duel_dev=dd,full_holdout=fh))
        best=max(choices,key=lambda x:x['key']);checks.append(dict(model=label,choices=choices))
        if not best['strengths']:continue
        mode=best['mode'];cp=Path(m['checkpoint']);cm=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'policy.pt')==m['policy_sha256']==cm['policy_sha256']
        assert sha(m['model_json'])==m['model_json_sha256']
        compare=c(online,'tau0.3','strong_full','holdout');fh=best['full_holdout'];weak=[]
        if fh['score']<compare['score']:weak.append('完整局holdout score低于线上参照，不能替换默认')
        if fh['deliveries']<compare['deliveries']:weak.append('完整局交付低于线上参照，运包能力有回退')
        if fh['self_deaths']>compare['self_deaths']:weak.append('完整局自死较多')
        if fh['actionable_stationary_pct']>compare['actionable_stationary_pct']:
            weak.append(f"完整局可行动静止占比高于线上参照：{fh['actionable_stationary_pct']:.2f}% vs {compare['actionable_stationary_pct']:.2f}%，不能称全局更流畅")
        if fh['max_actionable_stationary']>=100:weak.append('仍存在较长可行动静止段')
        own_web=[w(label,mode,b) for b in BOTS];online_web=[w(online,'tau0.3',b) for b in BOTS]
        own_self=sum(s['self_deaths'] for s in own_web);online_self=sum(s['self_deaths'] for s in online_web)
        if own_self>online_self:weak.append(f'网页40局自死{own_self}次，高于线上参照{online_self}次')
        if mode=='tau0.3' and sum(s['fountain_streak'] for s in own_web)<sum(w(label,'argmax',b)['fountain_streak'] for b in BOTS):
            weak.append('网页有效喷泉串弱于同checkpoint的argmax；采样胜负更好不能掩盖这一混合结果')
        for bot in BOTS:
            actual=w(label,mode,bot);reference=w(online,'tau0.3',bot)
            if actual['score']<reference['score']:
                weak.append(bot+' 网页快筛score低于线上参照：'+
                    str(actual['wins'])+'/'+str(actual['losses'])+'/'+str(actual['draws'])+' vs '+
                    str(reference['wins'])+'/'+str(reference['losses'])+'/'+str(reference['draws']))
        weak+=['只验证网页P1；每个网页对手每方式8局，独立短局dev/holdout各16局，样本有限，未证明全面或统计显著优势',
               '人工短局有任务隔离和属性/地形变化，导出固定完整局mode0；专项成绩不等于完整抢包胜率',
               '此处只推荐可选模型；训练持续变化，发布必须使用本收据固定checkpoint和SHA']
        exp=Path(m['model_json']).parent
        export_cfg=config(dict(module='scripts.r4_k4_export',args=[str(cp),str(exp)],env=CPU))
        replay_cfg=config(dict(module='scripts.cx_items_candidates_v1',args=['all'],env=CPU))
        record=dict(id='items-'+label.lower()+'-'+mode.replace('.',''),display_name=label+' '+mode,
            suggested_optional_label='近身击杀专项' if any(x['kill_dev_holdout'] for x in best['checks']) else '运包与行动风格专项',
            status='READY_OPTIONAL_SPECIALIST',checkpoint=str(cp.resolve()),policy_sha256=m['policy_sha256'],
            checkpoint_committed_sha256=sha(cp/'COMMITTED.json'),model_json=str(Path(m['model_json']).resolve()),
            model_json_sha256=m['model_json_sha256'],onnx=str((exp/'model.onnx').resolve()),onnx_sha256=sha(exp/'model.onnx'),
            export_receipt=str(exp/'K4_MODE0_IDENTITY.json'),export_entry='scripts.r4_k4_export.export',
            export_command='python_environment/bin/python -u model_evaluation.py '+export_cfg,
            reproduce_command='python_environment/bin/python -u model_evaluation.py '+replay_cfg,
            inference=dict(greedy=mode=='argmax',temperature=.3 if mode=='tau0.3' else 1.,seed=0x515154),
            rule=p['rule'],simulator_sha256=p['engine_sha256'],production_simulator=p['engine'],
            default_change=False,optional_publication_allowed=True,web_owner_publishes=True,
            strengths=best['strengths'],weaknesses=weak,evidence=[str(ROOT/'confirm_SUMMARY.json'),str(ROOT/'web_SUMMARY.json'),str(ROOT/'ARTIFACT_AUDIT.json'),str(REPORT)],
            metrics=dict(duel_dev=best['duel_dev'],duel_holdout=best['duel_holdout'],full_holdout=fh,
                web={b:w(label,mode,b) for b in BOTS}),comparison=best['checks'])
        record['paired_holdout']={ref:{kind:paired_intervals(conf['rows'],label,mode,ref,kind)
            for kind in ('duel','strong_full')} for ref in (online,health)}
        if record['paired_holdout'][online]['duel']['intervals']['causal_kills']['lower95']<=0:
            record['suggested_optional_label']='网页猎手与喷泉'
            record['weaknesses'].append('人工短局holdout因果杀的配对差95%区间含0，不宣称已建立近身击杀优势；本点的可选依据主要是网页猎手表现及重复喷泉信号')
        recommend.append(record)
    atomic_json(ROOT/'RECOMMENDATION_CHECKS.json',dict(status='COMPLETE',checks=checks))
    ready=dict(schema='qqt.web-candidate-ready/v1',status='READY' if recommend else 'NO_SUPPORTED_SPECIALIST_YET',
        created_unix=time.time(),rule=p['rule'],simulator_sha256=p['engine_sha256'],candidates=recommend,
        selection_scope='optional specialists; default untouched; no global superiority gate',
        online_reference=online+' tau0.3',healthy_reference=health,protocol=str(ROOT/'PROTOCOL.json'),protocol_sha256=sha(ROOT/'PROTOCOL.json'),
        evidence_report=str(REPORT),private_absolute_paths_intentional=True,training_continues=True,no_push=True,no_cc=True)
    ready['delivery_source_sha256']={f:sha(f) for f in ('scripts/cx_items_candidates_report_v1.py','scripts/cx_items_candidate_cpu_v3.py')}
    atomic_json('.hermes/WEB_CANDIDATE_READY.json',ready)
    ledger_path=Path('docs/training_experiment_ledger.md')
    if '### CX-WEB-ITEMS-CANDIDATES-20261107-V1' not in ledger_path.read_text():
        with ledger_path.open('a') as ledger:
            ledger.write('\n### CX-WEB-ITEMS-CANDIDATES-20261107-V1\n\nCPU-only production bun-native-v3-items selection; seven latest COMMITTED checkpoints frozen, argmax/tau0.3, fixed webpage P1 engine seat1, paired G_frozen tau0.3 and healthy u019302. Screen216, independent confirmation320, web five opponents320; all losses retained, source/model/engine/replay audited. Protocol `'+str(ROOT/'PROTOCOL.json')+'` SHA `'+sha(ROOT/'PROTOCOL.json')+'`; no training change, no push or default change. Result '+ready['status']+'; optional combinations '+', '.join(r['display_name'] for r in recommend)+'. Relative evidence report `'+str(REPORT)+'`; private web-owner contract `.hermes/WEB_CANDIDATE_READY.json`.\n')
    with REPORT.open('a') as f:
        f.write('\n## 七臂专项代表筛查\n\n|方向|筛查代表|因果杀合计|交付合计|喷泉串分组均值之和|可行动静止%分组之和|\n|---|---|---:|---:|---:|---:|\n')
        for role,s in representatives.items():
            f.write(f"|{role}|{s['model']['label']}|{s['kill']}|{s['deliveries']}|{s['fountain']:.3f}|{s['stationary']:.2f}|\n")
        f.write('\n筛查各组局数相同，上表仅用于候选排序；短局和完整局分组和不是生产线上统一得分。首批只对两个筛查候选追加独立dev/holdout与网页快筛，其他代表未被冒充独立通过。\n')
        f.write('\n## 可选交付结论\n\n所有逐局RESULT、生产物理SHA、模型SHA及独立重放hash已审计；argmax与τ0.3分别保留，失败不删。首批网页每对手每方式8局为快筛，不冒称历史64局已完成。独立专项dev与holdout各16局/方式，完整局holdout8局/方式；仅P1，有选择偏差及小样本限制。\n')
        f.write('\n指标口径：因果杀按原只读控制/死亡事件链统计，自死导致的胜场不冒充因果击杀。可行动静止=存活、未受控且真实坐标不变；c类为现有危险场下有安全推进证据的保守分类，不是全未来安全证明。喷泉串为有效跨tick批次、间隔≤24tick的最长串均值；严格串另保留在JSON的strict_fountain_streak，不按放泡量代替。人工专项300tick或首死亡终止、空手无运包，是任务设置而非网页完整局规则；网页和strong_full均为生产2400tick完整抢包。\n')
        for r in recommend:
            f.write('\n可选组合 **'+r['display_name']+'**；checkpoint `'+os.path.relpath(r['checkpoint'])+'`，policy SHA `'+r['policy_sha256']+'`。\n\n')
            for s in r['strengths']:f.write('- 长处：'+s+'。\n')
            for s in r['weaknesses']:f.write('- 局限：'+s+'。\n')
        if not recommend:f.write('\n当前两个点尚无独立重复专项长处，不伪造推荐。检查明细见 `'+str(ROOT/'RECOMMENDATION_CHECKS.json')+'`。\n')
        f.write('\n网页交付契约 `.hermes/WEB_CANDIDATE_READY.json` 为私有编排文件，按用户要求保留绝对checkpoint/导出路径；公开报告仅相对路径。网页CX负责发布，本owner不push、不修改默认。\n')
    text=REPORT.read_text()
    text=text.replace('阶段：实际导出完成，正在生产JS筛查；完成独立dev/holdout及网页快筛后才写ready候选。',
        '状态：216局七臂筛查、320局独立确认、320局五对手网页P1快筛全部完成；已完成856局逐项审计。交付契约状态 '+ready['status']+'。')
    # Queue scheduling restarts reused exact artifacts; keep one copy of each result table.
    sections=re.split(r'\n(?=## )',text);kept={};order=[]
    for section in sections:
        key=section.split('\n',1)[0]
        if key not in kept:order.append(key)
        kept[key]=section
    text='\n'.join(kept[key] for key in order)
    quick=['\n## 首批网页P1组合对照\n','每组合40局（五对手各8），仅为有限快筛；全部使用生产bun-native-v3-items。默认不改。\n',
        '|模型+τ|胜/负/平|交付|因果杀|自死|','|---|---|---:|---:|---:|']
    for label in (online,health)+tuple(r['model']['label'] for r in selected):
        stats=[w(label,'tau0.3',b) for b in BOTS]
        totals={k:sum(s[k] for s in stats) for k in ('wins','losses','draws','deliveries','causal_kills','self_deaths')}
        quick.append(f"|{label}+τ0.3|{totals['wins']}/{totals['losses']}/{totals['draws']}|{totals['deliveries']}|{totals['causal_kills']}|{totals['self_deaths']}|")
    # Replace an earlier compact table on idempotent receipt refresh.
    text=re.sub(r'\n## 首批网页P1组合对照\n.*?(?=\n## |\Z)','',text,flags=re.S)
    lines=text.split('\n',1);text=lines[0]+'\n'+'\n'.join(quick)+'\n'+lines[1]
    text+='\n最终实际评测调度为12个空闲物理CPU核36–47，共享逐核锁；纯调度回执 `runs/items_candidates_20261107/v1/CPU_SCHEDULING_V3.json`，八卡训练未更动。\n'
    REPORT.write_text(text)
    atomic_json(ROOT/'STATUS.json',dict(status='COMPLETE_READY' if recommend else 'COMPLETE_NO_SUPPORTED_SPECIALIST',candidates=len(recommend),games=len(inventory),report=str(REPORT),unix=time.time()))
    print(json.dumps(dict(status=ready['status'],candidates=[r['display_name'] for r in recommend])),flush=True)

if __name__=='__main__':finish(json.loads((ROOT/'PROTOCOL.json').read_text()))
