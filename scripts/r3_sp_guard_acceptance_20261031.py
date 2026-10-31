"""Verify the resumed SP first formal update without controlling training."""
import json
import hashlib
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts import r3_selfplay_owner_20261031 as owner


def main():
    base = owner.OUT
    entry = owner.read(base/'REGISTRATION.json')['entry']
    folder = base/'formal'/entry['run']
    first = owner.read(folder/'FIRST_FORMAL_UPDATE.json')
    if not first:
        print(json.dumps(dict(status='PENDING_FIRST_FORMAL',owner=owner.read(base/'STATUS.json'))))
        return
    request = owner.read(base/'DEDICATED_CAPACITY_RESUME.json')
    assert request['status'] == 'CONSUMED' and request['attempt'] == 'gpu3_guard1600'
    consumed = owner.read(base/'GUARD1600_TASK_CONSUMED.json')
    assert sha(Path(entry['protocol'])) == consumed['protocol_sha256']
    relocated = owner.read(base/'DEDICATED_GPU_RESTORE_ACCEPTANCE.json')
    restored = owner.read(folder/'RESTORE.json')
    for receipt,update in ((relocated,3040),(restored,3042)):
        assert receipt['checkpoint'].endswith(f'/u{update:06d}')
        assert all(receipt[k] for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
    assert first['status'] == 'PASS' and first['source']['update'] == 3043
    cp = Path(first['source']['checkpoint'])
    assert sha(cp/'state.pkl') == first['source']['state_sha256']
    assert sha(cp/'policy.pt') == first['source']['policy_sha256']
    rows = owner.tail_rows(folder/'updates.jsonl',700)
    row = next(row for row in rows if row['new_update'] == 3043)
    assert row['finite'] and min(row['update_l2'].values()) > 0
    assert row['actual_environment_steps_this_update'] == 262144
    assert owner.read(base/'MONITOR_BASELINE.json')['starting_state_unchanged']
    guard_bytes = (base/'RESOURCE_GUARD.json').read_bytes()
    guard = json.loads(guard_bytes)
    keys = ['runs/bomb_threat_v1:'+str(s) for s in (0,1,5,6,7)]
    keys += ['runs/bun_native_v2_20261030:'+s for s in ('b','c')]
    keys += ['runs/r3_items_route_20261030:'+s for s in ('a','b')]
    references = {key:guard.get('comparisons',{}).get(key,dict(status='WAIT_FIVE_WARM_UPDATES_OR_BASELINE',
                    minimum_ratio=owner.reference_minimum_ratio(key))) for key in keys}
    result = dict(status='PASS_FIRST_FORMAL_NOT_640_COMPLETE',gpu=entry['gpu'],seed=entry['seed'],
                  source=first['source'],env_per_second=first['env_per_second'],
                  first_cycle_includes_cold_compile=True,reference_ratios=references,
                  resource_guard_sha256=hashlib.sha256(guard_bytes).hexdigest(),
                  resource_guard_snapshot=guard,
                  restored_u3040=relocated,restored_u3042=restored,
                  protocol_sha256=consumed['protocol_sha256'],
                  last_observed_update=rows[-1]['new_update'],budget_end_update=3680,
                  audit_path=str(base/'audit/guard1600/OWN_GLUE_AUDIT.json'),
                  audit_sha256=sha(base/'audit/guard1600/OWN_GLUE_AUDIT.json'),unix=time.time())
    atomic_json(base/'GUARD1600_FIRST_FORMAL_ACCEPTANCE.json',result)
    if consumed['status'] != 'FIRST_FORMAL_ACCEPTED_BOUNDED_RUN_CONTINUES':
        with Path('docs/training_experiment_ledger.md').open('a') as ledger:
            ledger.write('\nR3-SP-GUARD03 GPU3首正式u3043已验收：state SHA `'+first['source']['state_sha256']+
                         '`，policy SHA `'+first['source']['policy_sha256']+'`，finite、Actor/Critic非零，262144真实步；u3040与u3042完整fresh restore均PASS。'+
                         '参照比例和原始快照见 `runs/r3_selfplay_ramp_20261031/GUARD1600_FIRST_FORMAL_ACCEPTANCE.json`；固定640预算由唯一owner继续，未宣称预算完成。\n')
    consumed.update(status='FIRST_FORMAL_ACCEPTED_BOUNDED_RUN_CONTINUES',acceptance='runs/r3_selfplay_ramp_20261031/GUARD1600_FIRST_FORMAL_ACCEPTANCE.json',
                    acceptance_sha256=sha(base/'GUARD1600_FIRST_FORMAL_ACCEPTANCE.json'),unix=time.time())
    atomic_json(base/'GUARD1600_TASK_CONSUMED.json',consumed)
    report = Path('reports/r3_sp_guard_amend_20261031.md')
    text = report.read_text().split('\n<!-- GUARD1600_FORMAL -->')[0]
    lines = ['','<!-- GUARD1600_FORMAL -->','','## 正式更新验收','',
             f"GPU{entry['gpu']}独占，u3040完整恢复与u3042 fresh restore均PASS；首正式u3043 finite、Actor/Critic非零：{row['update_l2']}，262144真实on-policy步。",
             f"state SHA `{first['source']['state_sha256']}`；policy SHA `{first['source']['policy_sha256']}`。",
             f"首周期env/s {first['env_per_second']:.3f}（含新进程冷编译，非暖态稳定吞吐）；最新已观察u{rows[-1]['new_update']}，固定终点u3680。",'',
             '|参照线|同阶段基线env/s|当前5更新总步/总墙钟env/s|比例|最低比例|连续越界窗|',
             '|---|---:|---:|---:|---:|---:|']
    for key,r in references.items():
        if 'ratio' in r:
            lines.append(f"|{key}|{r['baseline']['env_per_second']:.2f}|{r['current']['env_per_second']:.2f}|{r['ratio']:.4f}|{r['minimum_ratio']:.2f}|{r['strikes']}|")
        else:lines.append(f"|{key}|待暖态基线|待5更新|未可比|{r['minimum_ratio']:.2f}|不预判|")
    lines += ['','吞吐比例是时间窗相关观测；同5更新总步/总cycle墙钟与原pre-SP锚的相同奇偶阶段比较，修正交替调度中位数偏差；参照run变更后才建立新基线。收据 `runs/r3_selfplay_ramp_20261031/GUARD1600_FIRST_FORMAL_ACCEPTANCE.json`；640预算尚未完成，由唯一owner及watchdog继续。']
    report.write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('status','gpu','env_per_second','last_observed_update','reference_ratios')},ensure_ascii=False))


if __name__ == '__main__':
    main()
