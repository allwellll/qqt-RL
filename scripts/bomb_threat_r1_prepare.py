"""Durable CPU-only R1 prerequisites; never launches/signals GPU workers."""
import os,json,time,subprocess,sys
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030')


def live(module):
    for p in Path('/proc').iterdir():
        try:
            if module.encode() in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def main():
    guard=lock(OUT/'PREPARE.lock');began=time.time();children={}
    specs=[('BANK_FROZEN.json','scripts.bomb_threat_r1_bank','120-127','bank_prepare.log'),
        ('RESET_AUDIT.json','scripts.bomb_threat_r1_reset_audit','120-127','reset_audit.log'),
        ('MICRO.json','scripts.bomb_threat_r1_micro','128-135','micro.log')]
    while True:
        for marker,module,cpu,log in specs:
            if (OUT/marker).exists():
                if json.loads((OUT/marker).read_text()).get('status')!='PASS':raise RuntimeError('failed prerequisite '+marker)
                continue
            if marker!='BANK_FROZEN.json' and not (OUT/'BANK_FROZEN.json').exists():continue
            if module in children:
                p,f=children[module]
                if p.poll() is None:continue
                f.close()
                if p.returncode:
                    atomic_json(OUT/'PREPARE_BLOCKED.json',dict(status='BLOCKED',module=module,exit_code=p.returncode,log=str(OUT/log),training_unaffected=True,unix=time.time()))
                    from scripts.iter_pbt_owner import notify
                    notify('R1_PREPARE_BLOCKED_'+module,'R1技术前置失败，详见PREPARE_BLOCKED收据；原训练未停止，不放行新奖励。');return
                continue
            if live(module):continue
            f=(OUT/log).open('ab',buffering=0)
            p=subprocess.Popen(['taskset','-c',cpu,sys.executable,'-u','-m',module],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children[module]=(p,f)
        states={m:(OUT/m).exists() for m,_,_,_ in specs}
        atomic_json(OUT/'PREPARE_STATUS.json',dict(status='RUNNING',gates=states,wall_seconds=time.time()-began,training_worker_launches=0,unix=time.time()))
        if all(states.values()):
            tests=OUT/'TEST_ACCEPTANCE.json'
            if not tests.exists():time.sleep(5);continue
            assert json.loads(tests.read_text())['status']=='PASS'
            atomic_json(OUT/'READY_FOR_BOUNDARY.json',dict(status='PASS',prerequisites={m:sha(OUT/m) for m in [*states,'TEST_ACCEPTANCE.json']},
                next_step='sole existing bomb_threat_owner consumes at complete chunk boundary; per-arm smoke then fresh restore then formal',unix=time.time()))
            from scripts.iter_pbt_owner import notify
            notify('R1_CPU_GATES_PASS','R1真实P1 Bank完整重放、100000次实际reset、P1返笼微测试和focused回归通过；唯一owner将在各槽完整chunk边界切换，训练未收信号。')
            return
        time.sleep(10)

if __name__=='__main__':main()
