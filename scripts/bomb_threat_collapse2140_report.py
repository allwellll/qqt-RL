"""Read-only 40-update training and raw dev-seat diagnosis for amendment03."""
import json,time,statistics
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_train_monitor import read_updates,native_component_schema
from scripts.bomb_threat_slot04_guard2140 import BASE,OUT,ROOT,dev_greedy,refresh

def read(p):return json.loads(p.read_text())
def stats(rows):
    e=[r['exposure'] for r in rows];parts=[v['native_parts'] for v in e]
    sums=lambda k:sum(v.get(k,0) for v in e)
    den=sums('absolute_total_reward');ticks=sums('policy_diagnostic_ticks');n=len(rows)
    schema=native_component_schema(len(parts[0]))['indices'];tot={k:sum(a[i] for a in parts) for k,i in schema.items()}
    comp={k:tot[k] for k in ['global_safety','settled_kill','transport_capture','transport_shape','control_entry_reward','control_residual_reward','control_clawback','route_shaped_reward']}
    seats=[sum(v['seat_ticks'][i] for v in e) for i in [0,1]];live=[sum(v['seat_live_ticks'][i] for v in e) for i in [0,1]]
    return dict(updates=[r['new_update'] for r in rows],count=n,complete=n==40,
        policy_entropy=sums('policy_entropy_sum')/ticks,bomb_probability=sums('bomb_probability_sum')/ticks,diagnostic_live_ticks=ticks,
        approx_kl=statistics.mean(r['loss'][4] for r in rows),clip_frac=statistics.mean(r['loss'][5] for r in rows),
        seat_ticks=seats,seat_fractions=[v/sum(seats) for v in seats],seat_live_ticks=live,seat_live_fractions=[v/sum(live) for v in live],
        absolute_reward=den,total_reward=sums('total_reward'),shaping_paid=sums('chain_reward'),shaping_abs_fraction=sums('chain_reward')/den,
        threat_events=sums('threat_events'),topology_events=sums('topology_events'),shaping_candidates=sums('chain_candidates'),settled=sums('chain_settled'),
        components=comp,component_abs_fractions={k:v/den for k,v in comp.items()},
        learner_deaths=sums('learner_deaths'),bombs_placed=sums('learner_bombs_placed'),
        missing=['threat/topology/chain separate paid amounts','native win/loss reward separated from lesson_task','learner training self_death vs enemy_death disaggregation'],
        env_per_second=sum(r['actual_environment_steps_this_update'] for r in rows)/sum(r['cycle_seconds'] for r in rows),finite=all(r['finite'] for r in rows))

def main():
    windows=[];inputs={};restores={}
    for s in [4,7]:
        for c in [20,21]:
            run=f'bt_slot{s:02d}_c{c:06d}';out=BASE/'formal'/run;p=out/'updates.jsonl';rows=read_updates(p);inputs[str(p.relative_to(ROOT))]=dict(sha256=sha(p),last_update=rows[-1]['new_update'],note='live snapshot; subsequent appends expected')
            for end in range((rows[0]['new_update']+39)//40*40,rows[-1]['new_update']+1,40):
                group=[r for r in rows if end-40<r['new_update']<=end]
                if len(group)==40:windows.append(dict(slot=s,chunk=c,end_update=end,**stats(group)))
            if c==21:
                restores[str(s)]=dict(restore=read(out/'RESTORE.json'),smoke=read(out/'SMOKE.json')['passed'],source=read(BASE/'protocols'/(run+'.json'))['continuation_source'],first642=read(out/'u000642/COMMITTED.json'))
    dev=[]
    # Only complete raw64 argmax evidence; categorical need not finish first.
    for s in [1,4,7]:
        for out in sorted((BASE/'eval').glob(f'bt_slot{s:02d}_c*_u*')):
            if out.name.endswith('_holdout'):continue
            name,u=out.name.rsplit('_u',1)
            if not u.isdigit() or int(u)%40:continue
            cfgp=BASE/'protocols'/(name+'.json')
            if not cfgp.exists():continue
            cfg=read(cfgp);chunk=cfg.get('chunk',-1)
            if (s in [4,7] and chunk not in [20,21]) or (s==1 and chunk<19):continue
            d=dev_greedy(out)
            if d:dev.append(dict(slot=s,chunk=chunk,**d))
    guard=refresh();native=read(BASE/'NATIVE_STATUS.json');done=list((BASE/'eval').glob('*/native_hunter256_v1/NATIVE_HUNTER_COMPLETE.json'))
    latest=sorted(done,key=lambda p:p.stat().st_mtime,reverse=True)[:3]
    result=dict(status='PASSIVE_DIAGNOSIS_COMPLETE_GUARD_ACTIVE',windows=windows,dev_points=dev,restores=restores,guard=guard,native_status=native,native_completed_markers=len(done),latest_native=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in latest],input_snapshots=inputs,unix=time.time())
    atomic_json(OUT/'DIAGNOSIS.json',result)
    text=['# ITER2 第10轮：slot04塌缩诊断与amendment03（20261029）','',
        '执行`.hermes/STEER_slot04_collapse_20261029_2140.md`。诊断CPU只读；7路训练和GPU2原生回补继续，未向训练worker发信号。编排方将slot04 u640之后趋势定性为拖延型消极塌缩；以下以原始训练行和完整dev argmax分片复算，区分行为现象与根因证据。','',
        '## 每40更新训练诊断','',
        '每行完整40个正式更新，非单点；熵/放泡概率来自masked live collector（熵为move+ability总熵，概率分母为所有live ticks，非仅合法放泡机会）；KL/clip为优化minibatch指标的更新均值。seat为全tick比例，live比例另存JSON；奖励分母为Σtick |实际总reward|，不是各分项绝对值之和。各分项有正负和重叠诊断项，不应相加至100%。','',
        '|slot/chunk|末u|熵|approxKL|clip|P(bomb)|P0/P1 tick%|合并塑形实付%|transport交付%|global safety%|死亡数|',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for w in windows:
        f=w['component_abs_fractions'];text.append(f"|{w['slot']:02d}/c{w['chunk']}|{w['end_update']}|{w['policy_entropy']:.4f}|{w['approx_kl']:.5f}|{w['clip_frac']:.4f}|{w['bomb_probability']:.4f}|{100*w['seat_fractions'][0]:.1f}/{100*w['seat_fractions'][1]:.1f}|{100*w['shaping_abs_fraction']:.3f}|{100*f['transport_capture']:.2f}|{100*f['global_safety']:.3f}|{w['learner_deaths']}|")
    text+=['','c20与c21完整窗口均值（每行40更新，c21只含已完整的窗口，不将尾部残窗充整）：','', '|slot/chunk|窗口数|总熵|P(bomb)%|合并塑形实付%|global safety%|平均死亡/40更新|','|---|---:|---:|---:|---:|---:|---:|']
    for slot in (4,7):
        for chunk in (20,21):
            ww=[w for w in windows if w['slot']==slot and w['chunk']==chunk]
            avg=lambda key:statistics.mean(w[key] for w in ww)
            safe=statistics.mean(w['component_abs_fractions']['global_safety'] for w in ww)
            text.append(f"|{slot}/c{chunk}|{len(ww)}|{avg('policy_entropy'):.5f}|{100*avg('bomb_probability'):.3f}|{100*avg('shaping_abs_fraction'):.3f}|{100*safe:.3f}|{avg('learner_deaths'):.1f}|")
    text+=['','这些日志未显示slot04全局熵塌缩或负奖励/合并塑形占比上升；但训练混合暴露的均值不能排除特定JS argmax状态的消极策略，也不能独立识别奖励因果。','', '分项补充（金额而非事件数；正负均保留）：','', '|slot/c/u|合并威胁+拓扑+连锁|威胁事件|拓扑事件|transport shape|transport交付|原生结算kill|control entry|control residual|control clawback|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for w in windows:
        c=w['components'];text.append(f"|{w['slot']}/c{w['chunk']}/u{w['end_update']}|{w['shaping_paid']:.3f}|{w['threat_events']:.0f}|{w['topology_events']:.0f}|{c['transport_shape']:.2f}|{c['transport_capture']:.2f}|{c['settled_kill']:.2f}|{c['control_entry_reward']:.2f}|{c['control_residual_reward']:.2f}|{c['control_clawback']:.2f}|")
    text+=['','**未记录**：威胁/拓扑/连锁分别实付；原生胜负奖励从lesson_task中独立拆分；训练死亡中自伤/被敌炸的分别计数。`chain_reward`历史名称实际是三项合并结算，不可直接称单独威胁占比，不以事件数×系数冒充实付（去重、pending、cap和作废会改变结算）。global_safety是已记录的聚合安全惩罚，不等同自伤。','', '## u640→u642恢复','']
    for s,v in restores.items():
        cm=v['first642'];delta=cm['metrics']['update_l2'];rr=v['restore'];text.append(f"- slot{s}：smoke={v['smoke']}；fresh进程next-step/Adam/env+aux+RNG/ring={all(rr[k] for k in ['passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'])}；u642 Actor/Critic L2 {delta['actor']:.6f}/{delta['critic']:.6f}，finite={cm['metrics']['finite']}；源u640 state SHA `{v['source']['state_sha256']}`。源边界与恢复验收原件见`extension1280/BOUNDARY640_ACCEPTANCE.json`及各c21 `RESTORE.json`。")
    text+=['','两路恢复机制一致且无已记录损坏；不能据此证明所有潜在训练机制没有问题。c21协议initialization历史文本仍写policy-only，但实际continuation_source和worker分支明确完整状态接续，本报告按实际路径解释。','', '## dev argmax逐seat完整局','', '每点各seat32局；zero为己方0泡且对手实际放泡>0的单方零泡，双方0泡另列；所有点先验证seed集合、model→policy SHA绑定与逐局终局，categorical未齐不阻挡完整argmax诊断。','', '|slot/c|u|P0胜/平/负|P0 W-L|P0泡/局|P0单方零泡|P0双方零泡|P0击杀|P1胜/平/负|P1 W-L|P1泡/局|P1单方零泡|P1击杀|','|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for d in sorted(dev,key=lambda r:(r['slot'],r['update'])):
        a=d['by_seat']['P0'];b=d['by_seat']['P1'];text.append(f"|{d['slot']:02d}/c{d['chunk']}|{d['update']}|{a['win']}/{a['draw']}/{a['loss']}|{a['win_minus_loss']}|{a['bombs_per_game']:.2f}|{a['zero_bomb_games']}|{a['structural_zero_bomb_games']}|{a['causal_kills']}|{b['win']}/{b['draw']}/{b['loss']}|{b['win_minus_loss']}|{b['bombs_per_game']:.2f}|{b['zero_bomb_games']}|{b['causal_kills']}|")
    text+=['','## 护栏与因果边界','',f"当前护栏 **{guard['status']}**；从u880起严格连续3个40倍数dev点，P0胜-负≤0且两seat单方零泡合计≥12，才登记切换。缺点不跳过、不视为满足；已完成argmax64即被动消费，无须等categorical。"]
    for g in guard['points']:text.append(f"- u{g['update']}：P0 W-L={g['P0']['win_minus_loss']}，单方零泡合计={g['unilateral_zero_total']}，满足={g['qualifies']}。")
    text+=['','触发后仅slot04下一个完整320更新chunk原子边界切P0 A3，Actor/Critic/Adam/env/aux/RNG/ring及seed2026112904完整保留，不重新初始化或择峰。为使完整state与P0新目标一致，在途P1局保留原物理座位/账本自然结束，然后新局按原P0 reset；瞬时P1暴露继续记录。这一迁移解释写入amendment03，不把P1 aux硬解释成P0，也不瞬移携包。slot07继续固定1280严格dev+holdout判据，u840较好不提前推广。slot01只继续20:40红色观察，不依据本轮P1回落改配方。','',
        'slot04与slot07全tick seat比例均严格50/50，故“slot04实际少采某seat”无证据支持；同配方slot07未出现同期相同dev退化，不能将seat样本减半认定为充分原因。恢复证据未发现差异。训练categorical混合对手/Bank暴露与JS argmax完整局不同：训练熵或概率稳定不排除argmax局部状态消极策略，不能由相关趋势推导唯一因果。继续按冻结预算和预声明护栏判断，不自行调奖励。','',
        '## GPU2与可复算证据','',f"GPU2唯一native owner：{native.get('status')}，pending={native.get('pending')}，完整native marker数={len(done)}；512/768整局批扫描、256局/动作模式，辅助JAX不代替生产JS闸门。"]
    for p in latest:text.append(f'- `{p.relative_to(ROOT)}`，SHA `{sha(p)}`。')
    text+=['','原始窗口、分项、输入SHA、各seat逐局分片SHA和恢复原件摘录：`runs/bomb_threat_v1/a3s/amendment03/DIAGNOSIS.json`；护栏：`PROTOCOL.json`/`GUARD_STATUS.json`；复算：`CUDA_VISIBLE_DEVICES= .venv/bin/python -m scripts.bomb_threat_collapse2140_report`。未push/发布/改网页/models.json/BEST，未删除state或其他产物。']
    (ROOT/'reports/bomb_threat_v1_iter2_round10_20261029.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=result['status'],windows=len(windows),dev_points=len(dev),guard=guard['status'],native_pending=native.get('pending'))))

if __name__=='__main__':main()
