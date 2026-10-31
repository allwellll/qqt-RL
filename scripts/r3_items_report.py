"""Recomputable denominators for passive R3 full-game diagnostics."""
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha

OUT=Path('runs/r3_items_route_20261030')


def summarize(es,seat):
    rows=[e for e in es if e['seat']==seat];ms=[e['r3'] for e in rows]
    hist=[v for m in ms for v in m['live_hist'][seat]];batch=[v for m in ms for v in m['batch_hist'][seat]]
    contacts=[dict(c,episode_seed=e['seed'],followup_kill_observed=any(k['attacker']==seat and c['tick']<=k['tick']<=c['tick']+50 for k in e['kills']),
        followup_pickup_observed=any(v['type'] in ('stock_pickup','loose_pickup','contact_transfer') and v['player']==seat and c['tick']<=v['tick']<=c['tick']+50 for v in e['antifarm']['events']),
        full_50tick_followup=e['ticks']>=c['tick']+50)
        for e in rows for c in e['r3']['contacts'] if c['type']==2 and c['owner']==seat and c['player']!=seat]
    slides=[v for m in ms for v in m['slides'] if v['player']==seat]
    alive=sum(m['alive_ticks'][seat] for m in ms);chains=[v for m in ms for v in m['chains'] if v['player']==seat]
    avg=lambda key:float(np.mean([m[key][seat] for m in ms])) if ms else None
    return dict(games=len(rows),wins=sum(e['winner']==seat for e in rows),losses=sum(e['winner']==1-seat for e in rows),
        draws=sum(e['winner'] in (-1,None) for e in rows),pickup_events=np.sum([m['pickups'][seat] for m in ms],axis=0).tolist(),
        pickup_type_order=['bomb','blast','speed','banana','glue'],terminal_attributes=[m['attributes'][seat] for m in ms],
        release_counts=np.sum([m['releases'][seat] for m in ms],axis=0).tolist(),release_type_order=['banana','glue'],
        enemy_glue_contacts=len(contacts),critical_glue_contacts=sum(c['critical_path'] for c in contacts),glue_events=contacts,
        glue_subsequent_kill=sum(c['followup_kill_observed'] for c in contacts),glue_subsequent_pickup=sum(c['followup_pickup_observed'] for c in contacts),
        glue_incomplete_followup=sum(not c['full_50tick_followup'] for c in contacts),
        glue_self_contacts=sum(c['type']==2 and c['owner']==seat and c['player']==seat for m in ms for c in m['contacts']),
        carrying_own_banana_contacts=sum(c['type']==1 and c['owner']==seat and c['player']==seat and c['carried']>=0 for m in ms for c in m['contacts']),
        banana_slides=slides,banana_completed=sum(v.get('end_reason')=='completed' for v in slides),
        banana_estimated_saved_ticks=sum(v.get('saved_ticks_estimate') or 0 for v in slides if v.get('end_reason')=='completed'),
        banana_blast_within30=sum(v.get('blast_within30',False) for v in slides),
        banana_unfinished=sum(m['unfinished_slides'][seat] for m in ms),
        routes=[dict(initial=m['route_initial'],final=m['route_final']) for m in ms],own_causal_shortest_path_bricks=sum(m['path_bricks'][seat] for m in ms),
        alive_ticks=alive,zero_bomb_ticks=sum(m['zero_bomb_ticks'][seat] for m in ms),
        zero_bomb_alive_fraction=sum(m['zero_bomb_ticks'][seat] for m in ms)/alive if alive else None,
        live_bomb_mean=float(np.mean(hist)) if hist else None,live_bomb_p90=float(np.percentile(hist,90)) if hist else None,live_bomb_max=max(hist,default=None),
        live_batch_mean=float(np.mean(batch)) if batch else None,live_batch_max=max(batch,default=None),
        chain_events=len(chains),chain_batch_counts=[v['batch_count'] for v in chains],chains=chains,
        self_deaths=sum(e['self_deaths'][seat] for e in rows),
        missing_data=['counterfactual benefit of glue','same-action no-banana branch','true opponent intention'],
        recomputable=['per-game raw contacts/slides/chain edges','placement batch histories','route endpoints','all seeds and outcomes'])


def report(p):
    result={};files={}
    for model in p['models']:
        result[model]={}
        for c in p['conditions']:
            es=[]
            for f in sorted((OUT/'baseline'/model/c).glob('shard_*.json.outcome.json')):
                es+=json.loads(f.read_text())['episodes'];files[str(f)]=sha(f)
            assert len(es)==128 and len({(e['seed'],e['seat']) for e in es})==128
            result[model][c]={f'P{s}':summarize(es,s) for s in (0,1)}
    doc=dict(status='COMPLETE',games=768,results=result,files=files,protocol_sha256=sha(OUT/'BASELINE_PROTOCOL.json'),unix=time.time())
    atomic_json(OUT/'BASELINE_COMPLETE.json',doc)
    lines=['# R3 道具、两笼通路与连泡基线（20261030）','',
      '768完整生产JS自然局；历史指定slot00 u2400与R2-a冻结点，各三档×两席位×64局，greedy，老师关闭。全部原始分片和SHA见 `runs/r3_items_route_20261030/BASELINE_COMPLETE.json`。',
      '', '|模型|对手|席位|局数|胜/平/负|五类拾取|蕉/胶释放|持包自踩蕉|胶敌触发|泡mean/P90/max|批次mean/max|0泡存活占比|真连锁|',
      '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for m,cs in result.items():
        for c,seats in cs.items():
            for s,v in seats.items():lines.append(f"|{m}|{c}|{s}|{v['games']}|{v['wins']}/{v['draws']}/{v['losses']}|{v['pickup_events']}|{v['release_counts']}|{v['carrying_own_banana_contacts']}|{v['enemy_glue_contacts']}|{v['live_bomb_mean']:.2f}/{v['live_bomb_p90']:.0f}/{v['live_bomb_max']}|{v['live_batch_mean']:.2f}/{v['live_batch_max']}|{v['zero_bomb_alive_fraction']:.3f}|{v['chain_events']}|")
    lines+=['','完整终局三属性、初终BFS/最少砖数、胶触发状态及50tick结果、香蕉完整滑行和30tick危险、因果路径砖及连锁批次数均保存在逐席位JSON，0分母为null。',
      '', '限制：同一map806，出生几何重复不等于独立地图；胶后续击杀/拾取是观察性事件，不能称“因此”；追击和去包仅几何代理；香蕉省tick是同路线原生持包速度估算，不是独立动作分支因果效应；终局不足窗口明确censored。只Sim.step，不代表真人frameStep。']
    Path('reports/r3_items_route_chain_20261030.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'NOTIFY.log').open('ab') as log:
        subprocess.run(['jdme-notify','send','R3基线768完整JS局已完成，历史u2400与R2-a各三档两席位64局；道具/香蕉/胶/两笼通路/真实连锁分母已落盘。见reports/r3_items_route_chain_20261030.md；R1/R2继续，无网页规则切换。'],stdout=log,stderr=subprocess.STDOUT,check=False)
