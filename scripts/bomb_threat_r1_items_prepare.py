"""Durable revised-R1 CPU gates; GPU launch stays in existing sole owner."""
import os,json,time,subprocess,sys
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2')


def live(module):
    for p in Path('/proc').iterdir():
        try:
            if module.encode() in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def main():
    guard=lock(OUT/'PREPARE.lock');children={};began=time.time()
    specs=[('BANK_FROZEN.json','scripts.bomb_threat_r1_items_bank','120-127'),
        ('BANK_WITH_HISTORY.json','scripts.bomb_threat_r1_items_bank_history','120-127'),
        ('RESET_AUDIT.json','scripts.bomb_threat_r1_items_reset_audit','120-127'),
        ('MICRO.json','scripts.bomb_threat_r1_items_micro','128-135'),
        ('WIRING.json','scripts.bomb_threat_r1_items_wiring','136-143')]
    while True:
        for marker,module,cpus in specs:
            if (OUT/marker).exists():
                assert json.loads((OUT/marker).read_text())['status']=='PASS';continue
            if marker!='BANK_FROZEN.json' and not (OUT/'BANK_FROZEN.json').exists():continue
            if marker not in ('BANK_FROZEN.json','BANK_WITH_HISTORY.json') and not (OUT/'BANK_WITH_HISTORY.json').exists():continue
            if module in children:
                p,f=children[module]
                if p.poll() is None:continue
                f.close()
                if p.returncode:
                    receipt=dict(status='BLOCKED',module=module,exit_code=p.returncode,log=str(OUT/(module.split('.')[-1]+'.log')),training_unaffected=True,unix=time.time())
                    atomic_json(OUT/'PREPARE_BLOCKED.json',receipt)
                    from scripts.iter_pbt_owner import notify
                    notify('R1_ITEMS_PREPARE_BLOCKED_'+module,'R1道具原生迁移技术前置失败；新arm未放行，旧对照训练继续，详见items_v2/PREPARE_BLOCKED.json。');return
                continue
            if live(module):continue
            f=(OUT/(module.split('.')[-1]+'.log')).open('ab',buffering=0)
            p=subprocess.Popen(['taskset','-c',cpus,sys.executable,'-u','-m',module],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children[module]=(p,f)
        markers=[x[0] for x in specs]+['PARITY_TEST_ACCEPTANCE.json','TEST_ACCEPTANCE.json','PARENT_EXPORT.json']
        states={f:(OUT/f).exists() for f in markers}
        atomic_json(OUT/'PREPARE_STATUS.json',dict(status='RUNNING',gates=states,wall_seconds=time.time()-began,training_worker_launches=0,unix=time.time()))
        if all(states.values()):
            assert all(json.loads((OUT/f).read_text())['status']=='PASS' for f in markers)
            files=['jax_bomb/je05_core_env_r1_items.py','jax_bomb/je07_bun_env_r1_items.py','jax_bomb/je07_native_items.py',
                'scripts/bomb_threat_r1_item_progress.py','scripts/bomb_threat_r1_progress_reward.py','scripts/bomb_threat_r1_progress_collector.py','scripts/bomb_threat_r1_progress_train.py','scripts/bomb_threat_r1_items_runtime.py','scripts/bomb_threat_r1_reward.py','scripts/bomb_threat_r1_train.py','scripts/bomb_threat_r1_collector.py','jax_bomb/bun_safety.py','scripts/bomb_threat_r1_items_bank_history.py']
            atomic_json(OUT/'SOURCE_SEAL.json',dict(status='PASS',files={f:sha(Path(f)) for f in files},unix=time.time()))
            markers+=['SOURCE_SEAL.json','AMENDMENT03.json','AMENDMENT04_BANK_HISTORY.json']
            atomic_json(OUT/'READY_FOR_BOUNDARY.json',dict(status='PASS',prerequisites={f:sha(OUT/f) for f in markers},next_step='sole bomb_threat_owner safe boundary -> new namespace GPU smoke -> fresh restore -> formal640',unix=time.time()))
            from scripts.iter_pbt_owner import notify
            notify('R1_ITEMS_CPU_PASS','R1新版道具原生迁移：逐tick JS对照、真实双座位Bank重放、100k reset和微测试均通过；唯一owner自动安全接管，先GPU smoke/恢复再正式。');return
        time.sleep(10)

if __name__=='__main__':main()
