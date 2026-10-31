"""Register revised b/c only after same-engine six-case acceptance."""
import copy
import json
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bun_native_v2_owner import OUT,read,processes,MODULE

NEW='scripts.bun_native_v2_custody_train'
SOURCES=['scripts/bun_native_v2_custody_reward.py','scripts/bun_native_v2_custody_collector.py',
         'scripts/bun_native_v2_custody_train.py']


def register():
    accepted=read(OUT/'custody_v2_tests/ACCEPTANCE.json')
    assert accepted['status']=='PASS' and not accepted['failures']
    assert read(OUT/'CUSTODY_FOCUSED_TESTS.json')['status']=='PASS'
    reference=read(OUT/'custody_v2_tests/ORDINARY_old.json');current=read(OUT/'custody_v2_tests/ORDINARY_new.json')
    assert reference['status']==current['status']=='PASS' and reference['records']==current['records']
    for f,h in accepted['source_sha256'].items():assert sha(Path(f))==h
    slots=read(OUT/'R2_SLOTS.json');old=copy.deepcopy(slots)
    if all(slots[a].get('r2_reward_version')=='custody-graded-conservative-v2' for a in ('b','c')):return
    registration=read(OUT/'R2_REGISTRATION.json');source=registration['common_parent']
    assert source['update']==2721
    cp=Path(source['checkpoint'])
    assert sha(cp/'state.pkl')==source['state_sha256'] and sha(cp/'policy.pt')==source['policy_sha256']
    for arm in ('b','c'):
        oldentry=old[arm];assert oldentry['suspended']=='SUPERSEDED_BY_CUSTODY_AMENDMENT'
        assert not list(processes(MODULE,oldentry['run']))
        p=read(oldentry['protocol']);name=f'r2_{arm}_custodyv2_c000000'
        recipe=p['arms'][oldentry['run']]
        p.update(arms={name:recipe},continuation_source=source,chunk_start_update=source['update'],
            r2_reward_version='custody-graded-conservative-v2',r2_same_version=False,r2_chunk=0,worker_module=NEW,
            custody_amendment=str(OUT/'AMENDMENT05_CUSTODY_GRADED_CONSERVATIVE.json'),
            custody_amendment_sha256=sha(OUT/'AMENDMENT05_CUSTODY_GRADED_CONSERVATIVE.json'),
            custody_tests=str(OUT/'custody_v2_tests/ACCEPTANCE.json'))
        p['source_sha256'].update({f:sha(Path(f)) for f in SOURCES})
        target=OUT/'protocols'/f'{name}.json';assert not target.exists()
        atomic_json(target,p)
        entry=dict(oldentry,run=name,protocol=str(target),worker_module=NEW,chunk=0,
            run_prefix=f'r2_{arm}_custodyv2',r2_reward_version=p['r2_reward_version'])
        entry.pop('suspended');slots[arm]=entry
    atomic_json(OUT/'CUSTODY_V2_REGISTRATION.json',dict(status='REGISTERED',old_slots=old,new_slots=slots,
        common_parent=source,technical_acceptance=str(OUT/'custody_v2_tests/ACCEPTANCE.json'),
        new_environment_steps_per_arm=640*262144,old_updates_not_reused=True,unix=time.time()))
    atomic_json(OUT/'R2_SLOTS.json',slots)
    with Path('docs/training_experiment_ledger.md').open('a') as f:
        f.write('\n## BUN-NATIVE-V2-CUSTODY-REV2 — 20261030\n\n'
            '用户修订2及严格守恒选择：分级归属 Φ′−Φ，κ=1；防守型击杀/控制重复信用0、己源回收0，独立终局+20/0/−20。'
            '旧b/c安全保存后封存；a及R1不变。共同父仍u2721，完整state/Adam父与原b/c seeds保留，迁移重建自然env/RNG，'
            '每臂640更新=167772160新增真实环境步，env2048×128、minibatch8192、epoch1、LR7.5e−5、entropy.005、DAgger/KL0。'
            '逐臂协议与source SHA见 `runs/bun_native_v2_20261030/CUSTODY_V2_REGISTRATION.json`；六条同JS/JAX事件轨迹验收 '
            '`runs/bun_native_v2_20261030/custody_v2_tests/ACCEPTANCE.json`。这是用户要求的联合防刷奖励修订，不能归因于单一系数；'
            '严格原始归属闭环守恒不代表折扣PPO目标不变。CPU评估异步，持久唯一owner接续640chunk；网页规则不切换。\n')


if __name__=='__main__':register()
