"""Recomputable descriptive audit, exact original trajectories; no training changes."""
import json,math,shutil,time,statistics
from pathlib import Path
from collections import Counter,defaultdict
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/p0_jitter_u800');R=Path('runs/bomb_threat_v1/replay_best_u800')
CATS=['原地振荡','持包停滞','正常游走，只是概率低','其他']
def read(p):return json.loads(p.read_text())
def mean(a):return statistics.mean(a) if a else None
def segments(rows,key):
    out=[];start=None
    for i in range(len(rows)+1):
        yes=i<len(rows) and rows[i][key]
        if yes and start is None:start=i
        if start is not None and not yes:out.append((start,i));start=None
    return out
def merge_windows(windows,key,label):
    out=[]
    for w in windows:
        if not w[key]:continue
        if out and out[-1]['end']==w['start']:out[-1]['end']=w['end']
        else:out.append(dict(start=w['start'],end=w['end'],label=label))
    return out
def describe(d):
    rows=d['rows'];ww=d['windows'];osc={i for w in ww if w['raw_oscillation'] for i in range(w['start'],w['end'])};stall={i for w in ww if w['raw_stall'] for i in range(w['start'],w['end'])}
    carries=[]
    for start,end in segments(rows,'carry'):
        rr=rows[start:end];a=rr[0];z=rr[-1];gd=[r['home_static'] for r in rr if r['home_static']>=0]
        carry=dict(start=start,end=end,ticks=end-start,start_pos=a['pos'],end_pos=z['after'],start_home_static=a['home_static'],end_home_static=z['after_home_static'],min_home_static=min(gd) if gd else None,
            home_center_improvement=a['home_center_distance']-z['after_home_center_distance'],move_counts=dict(Counter(str(r['move']) for r in rr)),bomb_mask_legal_ticks=sum(r['bomb_legal'] for r in rr),bomb_masked_probability_mean=mean([r['bomb_probability'] for r in rr]),
            home_dynamic_unreachable_ticks=sum(r['home_dynamic']<0 for r in rr),bomb_only_disconnected_ticks=sum(r['home_dynamic']<0<=r['home_static'] for r in rr),brick_connection_if_ignored_ticks=sum(r['home_static']<0<=r['home_ignore_bricks_bombs'] for r in rr),
            reachable_dynamic_ticks=sum(r['home_dynamic']>=0 for r in rr),dynamic_distance_histogram=dict(Counter(r['home_dynamic'] for r in rr)),region_ticks=dict(Counter(r['region'] for r in rr)),raw_oscillation_ticks=sum(i in osc for i in range(start,end)),raw_stall_ticks=sum(i in stall for i in range(start,end)),positions_by_cell=dict(Counter(str(r['cell']) for r in rr)),
            termination='carry_continues_at_timeout' if end==len(rows) and z['next_carry'] else 'death_or_control' if not z['next_controlled'] else 'carry_changed_after_step')
        carries.append(carry)
    flips=[];rawflips=[];field_constant=0;field_comparisons=0
    def lr(a,b):return {a,b}=={2,3}
    for i in sorted(osc):
        if i==0 or i-1 not in osc:continue
        a,z=rows[i-1],rows[i]
        if a['cell']==z['cell']:
            field_comparisons+=1;field_constant+=(a['phi_home_geometry_proxy']==z['phi_home_geometry_proxy'] and a['phi_enemy_geometry_proxy']==z['phi_enemy_geometry_proxy'])
        if lr(a['move_top'],z['move_top']):
            flips.append(dict(tick=i,from_move=a['move_top'],to_move=z['move_top'],same_cell=a['cell']==z['cell'],near_grid_boundary=min(a['grid_boundary_distance'],z['grid_boundary_distance'])<=.1,
                raw_lr_flip=lr(a['raw_move_top'],z['raw_move_top']),mask_changed=a['move_mask']!=z['move_mask'],pos_before=a['pos'],pos_after=z['pos'],cell_before=a['cell'],cell_after=z['cell'],
                left_right_gap_before=a['move_logits'][2]-a['move_logits'][3],left_right_gap_after=z['move_logits'][2]-z['move_logits'][3],
                home_geometry_proxy_before=a['phi_home_geometry_proxy'],home_geometry_proxy_after=z['phi_home_geometry_proxy'],enemy_geometry_proxy_before=a['phi_enemy_geometry_proxy'],enemy_geometry_proxy_after=z['phi_enemy_geometry_proxy']))
        if lr(a['raw_move_top'],z['raw_move_top']):rawflips.append(i)
    physics=Counter();obstructed=[]
    for r in rows:
        if not r['controlled'] or r['move']==4:continue
        z=r['physics']
        if z:
            physics['instrumented_normal_movement_ticks']+=1
            if z['requested_axis']<.95*z['dist']:
                obstructed.append(r['tick']);physics['requested_axis_limited_ticks']+=1
                for k in ['wall_limited','brick_limited','bomb_limited']:physics[k]+=bool(z[k])
                physics['limited_unattributed']+=not any(z[k] for k in ['wall_limited','brick_limited','bomb_limited'])
        elif r['forced_status']:physics['forced_status_not_normal_probe']+=1
        else:physics['not_instrumented']+=1
        physics['zero_actual_displacement_ticks']+=r['displacement']<=1e-7
    intervals=d['reversals']['intervals'];of=[rows[i] for i in sorted(osc)]
    cells={}
    for r in of:
        k=str(r['cell']);z=cells.setdefault(k,dict(ticks=0,y_min=r['pos'][0],y_max=r['pos'][0],x_min=r['pos'][1],x_max=r['pos'][1],home_proxy_values=set(),enemy_proxy_values=set(),move_tops=Counter()))
        z['ticks']+=1;z['y_min']=min(z['y_min'],r['pos'][0]);z['y_max']=max(z['y_max'],r['pos'][0]);z['x_min']=min(z['x_min'],r['pos'][1]);z['x_max']=max(z['x_max'],r['pos'][1]);z['home_proxy_values'].add(r['phi_home_geometry_proxy']);z['enemy_proxy_values'].add(r['phi_enemy_geometry_proxy']);z['move_tops'][r['move_top']]+=1
    for z in cells.values():
        for k in ['home_proxy_values','enemy_proxy_values']:z[k]=sorted(z[k],key=lambda v:float('-inf') if v is None else v)
    legal=[r for r in rows if r['bomb_legal']]
    return dict(index=d['index'],seed=d['seed'],group=d['group'],ticks=d['ticks'],tail_ticks=d['tail_ticks'],category_ticks={c:sum(50 for w in ww if w['category']==c) for c in CATS},
        raw_oscillation_ticks=len(osc),raw_stall_ticks=len(stall),raw_overlap_ticks=len(osc&stall),oscillation_region_ticks=dict(Counter(r['region'] for r in of)),oscillation_enemy_euclidean_mean=mean([r['enemy_euclidean'] for r in of]),oscillation_enemy_grid_mean=mean([r['enemy_grid_distance'] for r in of if r['enemy_grid_distance']>=0]),
        direct_lr_reversals=len(d['reversals']['events']),reversal_interval_median=statistics.median(intervals) if intervals else None,reversal_interval_mean=mean(intervals),reversal_interval_histogram=dict(Counter(intervals)),reversal_blocked_ticks=sum(i in set(obstructed) for i in d['reversals']['events']),physics=dict(physics),
        logits=dict(oscillation_masked_lr_flips=len(flips),raw_lr_flips=len(rawflips),masked_lr_same_cell=sum(f['same_cell'] for f in flips),masked_lr_cross_cell=sum(not f['same_cell'] for f in flips),masked_lr_near_grid_boundary=sum(f['near_grid_boundary'] for f in flips),masked_lr_with_raw_flip=sum(f['raw_lr_flip'] for f in flips),masked_lr_mask_changed=sum(f['mask_changed'] for f in flips),same_cell_geometry_proxy_comparisons=field_comparisons,same_cell_geometry_proxy_unchanged=field_constant,events=flips,cell_summaries=cells),
        carry_segments=carries,carry_ticks=sum(r['carry'] for r in rows),bomb_legal_ticks=len(legal),bomb_probability_legal_mean=mean([r['bomb_probability'] for r in legal]),
        annotations=sorted(merge_windows(ww,'raw_oscillation','原地振荡')+merge_windows(ww,'raw_stall','持包停滞'),key=lambda z:(z['start'],z['label'])))
def main():
    cfg=read(B/'PROTOCOL.json');complete=read(B/'COLLECTION_COMPLETE.json')
    for p,h in complete['files'].items():assert sha(Path(p))==h
    ds=[read(B/f"game_{g['index']:03d}.json") for g in cfg['games']];rs=[describe(d) for d in ds]
    assert sum(r['ticks'] for r in rs)==complete['ticks']
    for r in rs:assert sum(r['category_ticks'].values())+r['tail_ticks']==r['ticks']
    groups={}
    for group in ['low','high']:
        rr=[r for r in rs if r['group']==group]
        groups[group]=dict(games=len(rr),ticks=sum(r['ticks'] for r in rr),analyzed_full_window_ticks=sum(sum(r['category_ticks'].values()) for r in rr),tail_ticks=sum(r['tail_ticks'] for r in rr),
            categories={c:dict(games=sum(r['category_ticks'][c]>0 for r in rr),ticks=sum(r['category_ticks'][c] for r in rr)) for c in CATS},
            raw_oscillation_ticks=sum(r['raw_oscillation_ticks'] for r in rr),raw_stall_ticks=sum(r['raw_stall_ticks'] for r in rr),overlap_ticks=sum(r['raw_overlap_ticks'] for r in rr),carry_ticks=sum(r['carry_ticks'] for r in rr),
            direct_lr_reversals=sum(r['direct_lr_reversals'] for r in rr),oscillation_region_ticks=dict(sum((Counter(r['oscillation_region_ticks']) for r in rr),Counter())),physics=dict(sum((Counter(r['physics']) for r in rr),Counter())),
            logits={k:sum(r['logits'][k] for r in rr) for k in ['oscillation_masked_lr_flips','raw_lr_flips','masked_lr_same_cell','masked_lr_cross_cell','masked_lr_near_grid_boundary','masked_lr_with_raw_flip','masked_lr_mask_changed','same_cell_geometry_proxy_comparisons','same_cell_geometry_proxy_unchanged']})
    reps={c:min((r['index'] for r in rs if r['category_ticks'][c]),default=None) for c in CATS}
    atomic_json(B/'RESULTS.json',dict(status='RECOMPUTED',groups=groups,games=rs,representatives=reps,source_manifest_sha256=sha(B/'PROTOCOL.json'),unix=time.time()))
    # Preserve original21 entries, append fixed earliest representatives (duplicates disclosed).
    if not (R/'MANIFEST_before_p0_jitter.json').exists():shutil.copyfile(R/'MANIFEST.json',R/'MANIFEST_before_p0_jitter.json')
    manifest=read(R/'MANIFEST_before_p0_jitter.json');original_ids={g['index'] for g in manifest['games']}
    for c,i in reps.items():
        if i is None:continue
        g=dict(next(g for g in cfg['games'] if g['index']==i));r=next(r for r in rs if r['index']==i)
        g['rule']='P0振荡/持包停滞诊断·'+c+('（重复聚焦）' if i in original_ids else '');g['annotations']=r['annotations']
        source=Path(cfg['source'])/f'game_{i:03d}';dest=R/f'game_{i:03d}';dest.mkdir(exist_ok=True)
        for name in ['review.json','MATCH.json']:
            if (dest/name).exists():
                a,z=read(dest/name),read(source/name)
                if name=='review.json':
                    # Separately recorded identical games have distinct timestamps/identity.
                    for key in ['actions','checks','summary']:assert a[key]==z[key]
                    for key in ['seed','learner_seat','checkpoint_sha256','model_sha256']:assert a['meta'][key]==z['meta'][key]
                else:assert a['actual']==z['actual'] and a['status']==z['status']=='MATCH'
            else:shutil.copyfile(source/name,dest/name)
        manifest['games'].append(g)
    manifest['p0_jitter_protocol']=str(B/'PROTOCOL.json');manifest['entries']=len(manifest['games']);manifest['independent_games']=len({g['index'] for g in manifest['games']});atomic_json(R/'MANIFEST.json',manifest)
    text=['# P0振荡与真实持包停滞审计（20261029）','',
        '执行`.hermes/TASK_p0_jitter_carry_stall_20261029_2230.md`。仅CPU，训练/奖励/原动作不改。预冻结15局P0低泡+16局P0高泡；来源为同一u800候选生产JS困难Hunter、argmax、老师关闭的完整自然局；无人工预携包。候选未晋升。',
        '',f"policy SHA `{cfg['model']['policy_sha256']}`；model JSON SHA `{cfg['model']['model_sha256']}`。31局共{complete['ticks']}实际tick，所有Review全动作/状态hash/终局验证通过；原状态补算移动及ability argmax均零不一致，CPU ONNX与JS首forward误差最大{max(d['forward_parity_max_error'] for d in ds):.6g}。没有重新学生rollout或介入动作。",'',
        '主要观察：低泡组原始振荡31,450/36,000tick=87.36%，高泡组21,050/34,598=60.84%；低泡并非单纯方向计数相等，而是正路程下净位移极低。低泡振荡内14,733次合法LR top1翻转，其中14,545次同时是未mask原logit LR top1翻转；4,574次发生在同一格，6,519次至少一端距格边界≤0.1。不能全归于mask或只归于跨格。index14/78真实持包全程2232/2095tick，动态及去泡静态返程图均不通，去砖后连通；持包同时伴随振荡，并非只停在敌笼附近。真实静态砖图不能证明永久无路或安全可炸通；没有证据支持直接加权。', '',
        '预注册`runs/bomb_threat_v1/p0_jitter_u800/PROTOCOL.json`。50tick不重叠窗口，完整前后50步均活且可行动；正路程且净位移/路程<0.1为振荡，零位移不冒充振荡。持包停滞为完整窗口真实携敌包、己笼中心净推进<0.1格且格点BFS未降到起点以下。独占分类优先级：持包停滞→振荡→正常游走概率低→其他；局数为各类出现局数，可跨类，tick互斥。正常游走定义路程≥1格、ratio≥0.1、至少一个合法放泡tick且合法P(bomb)均值<0.1，不代表移动目标合理。最后不足50tick不分类。另报振荡/停滞原始掩码与交集。', '',
        '|组|实际tick/分类tick/尾tick|振荡 局/tick|持包停滞 局/tick|游走低概率 局/tick|其他 局/tick|原始振荡/停滞/重叠tick|左右直接反转|', '|---|---|---|---|---|---|---|---|']
    for k,g in groups.items():
        cells=[f"{g['categories'][c]['games']}/{g['categories'][c]['ticks']}" for c in CATS]
        text.append('|'+ '|'.join([k,f"{g['ticks']}/{g['analyzed_full_window_ticks']}/{g['tail_ticks']}",*cells,f"{g['raw_oscillation_ticks']}/{g['raw_stall_ticks']}/{g['overlap_ticks']}",str(g['direct_lr_reversals'])])+'|')
    text+=['','振荡位置先按敌3×3笼中心曼哈顿≤3，再按敌实际出生点≤2，余下其他；不是己半场或敌半场替代。敌距同时报连续欧氏与瞬时可达格距；未通不填0。','', '|组|振荡位置tick|移动受限/其中墙/砖/泡|振荡内LR翻转/原logit翻转|同格/跨格/边界≤0.1|mask变化/同时原logit翻转|同格势proxy不变/比较次数|','|---|---|---|---|---|---|---|']
    for k,g in groups.items():
        p=g['physics'];l=g['logits'];text.append(f"|{k}|{g['oscillation_region_ticks']}|{p.get('requested_axis_limited_ticks',0)}/{p.get('wall_limited',0)}/{p.get('brick_limited',0)}/{p.get('bomb_limited',0)}|{l['oscillation_masked_lr_flips']}/{l['raw_lr_flips']}|{l['masked_lr_same_cell']}/{l['masked_lr_cross_cell']}/{l['masked_lr_near_grid_boundary']}|{l['masked_lr_mask_changed']}/{l['masked_lr_with_raw_flip']}|{l['same_cell_geometry_proxy_unchanged']}/{l['same_cell_geometry_proxy_comparisons']}|")
    text+=['','碰撞为原Review step中只读拦截真实`_steer`，以相同连续起点、同dist的`_tryMove`分别探测wall、wall+brick、真实blocked（含泡）。列为请求轴推进<95%dist，不等于最后坐标完全不动；侧滑/强制滑行另列。该Sim.step没有角色身体互阻，玩家身体碰撞阻挡“不适用”；推箱目标被玩家占据属于推箱约束，不能混作身体碰撞。','',
        '距离场只重算JS几何proxy：φhome=−到己3×3笼footprint的格BFS，φenemy=−到敌格BFS，在同一物理状态投影的格内常值；跨tick敌位置/砖/泡变化可令同格取值改变，并非永远不变。真实训练φ/奖励未记录，不能据此证明势函数错误或权重不足。logits为原模型原状态重算，不是Q。原始/合法top1、mask变化、同格/跨格及边界≤0.1分别列；同格或边界翻转仅为观察，敌状态、危险场和其他输入同时可能变化，不能单因果归于格点分辨率。每格连续坐标范围、proxy取值集合和逐翻转logit差保存RESULTS。','',
        '|index/组|tick|独占分类tick（振/停/游/其）|原始振荡tick|敌距欧氏/格均值|左右反转/间隔中位|携包tick|','|---|---|---|---|---|---|---|']
    for r in sorted(rs,key=lambda r:r['index']):
        text.append(f"|{r['index']}/{r['group']}|{r['ticks']}|{'/'.join(str(r['category_ticks'][c]) for c in CATS)}|{r['raw_oscillation_ticks']}|{r['oscillation_enemy_euclidean_mean']}/{r['oscillation_enemy_grid_mean']}|{r['direct_lr_reversals']}/{r['reversal_interval_median']}|{r['carry_ticks']}|")
    text+=['','真实持包区段逐段列出，连续位置完整保存在每局JSON；动作顺序0上/1下/2左/3右/4停。','', '|index/组/区段[start,end)|tick|位置起→止|己笼静态格距起/末/最小；连续净推进|移动计数|mask合法/Pbomb|动态可达/仅泡阻断/去砖可连tick|振荡/停滞tick|结束|','|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rs,key=lambda r:r['index']):
        for z in r['carry_segments']:
            text.append(f"|{r['index']}/{r['group']}/[{z['start']},{z['end']})|{z['ticks']}|{z['start_pos']}→{z['end_pos']}|{z['start_home_static']}/{z['end_home_static']}/{z['min_home_static']}；{z['home_center_improvement']:.4f}|{z['move_counts']}|{z['bomb_mask_legal_ticks']}/{z['bomb_masked_probability_mean']}|{z['reachable_dynamic_ticks']}/{z['bomb_only_disconnected_ticks']}/{z['brick_connection_if_ignored_ticks']}|{z['raw_oscillation_ticks']}/{z['raw_stall_ticks']}|{z['termination']}|")
    text+=['','动态BFS每tick含未爆泡；另列去泡静态、去砖保留泡、同时去砖去泡。不通只表示该时刻网格连接不通，不表示永久无路；泡会爆、砖可炸、玩家会响应，安全返程还需时序危险搜索。可达也不保证安全。持包本身原生不允许放泡，mask/P=0应与空手合法机会分母分开，不能称持包阶段“不会放泡”。完整carry区段含非整50窗口，因此carrytick可高于分类停滞tick。本表carry要求活且可行动，高泡5864tick与上一报告alive-only5898tick相差34tick原生受控携包，分母不同而非丢样本。', '',
        '几何测量修正01：生产Sim.step把crate作为可走过拾取，推箱目标不能占crate不等于角色移动不能通过crate；本任务BFS已去掉crate假阻挡，训练和原始低泡TRACE不改。旧测量保存在geometry_v1，修正无新forward、无新rollout；高泡停滞由旧8局3300tick修正为4局3100tick。`BARRIER_SNAPSHOTS.json`保存21个真实carry起点/终段的wall/brick/fuse与去砖路径见证、所穿砖格；仅为忽略阻挡的最短长度路径，不是最少炸砖或可执行安全方案。', '',
        f"四类按最早index追加录像：{reps}。原21入口保持，重复聚焦明确标签；中文可点击时间轴标原始振荡/停滞区段。链接保持原OSS对象，浏览器/上传验收见RECEIPT。", '',
        '复现：`node --test tests/bomb_threat_p0_jitter_test.js tests/bomb_threat_low_bomb_test.js tests/bomb_threat_iter2_actions_test.js`；`CUDA_VISIBLE_DEVICES= .venv/bin/python -m scripts.bomb_threat_p0_jitter_owner`（锁与逐局续接）；`.venv/bin/python -m scripts.bomb_threat_p0_jitter_report`；`.venv/bin/python -m scripts.build_bomb_threat_best_replay`。原始及SHA见COLLECTION_COMPLETE/RESULTS，每局含全部before/after坐标、50tick窗口、反转tick和间隔、逐tickmove logits/mask、己笼四种BFS及邻格dynamic场。','']
    Path('reports/bomb_threat_v1_p0_jitter_20261029.md').write_text('\n'.join(text))
    print(json.dumps(dict(groups=groups,representatives=reps),ensure_ascii=False))
if __name__=='__main__':main()
