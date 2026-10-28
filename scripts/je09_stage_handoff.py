"""Single H2->H3 handoff supervisor; no capability score controls launch."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
H2 = ROOT/'runs/je09_bun_20261016/stage_h2_item_coverage_pfsp8_v15'
H3 = ROOT/'runs/je09_bun_20261016/stage_h3_natural_combat_coverage_pfsp8_v17'

def write(name, data):
    from scripts.population_run_state import atomic_json
    atomic_json(H3/'state'/name,data)

def main():
    from scripts.population_run_state import lock
    owner = lock(H3/'handoff.lock')
    complete=json.loads((H2/'CAMPAIGN_COMPLETE.json').read_text())
    trainer=json.loads((H2/'TRAINING_COMPLETE.json').read_text())
    assert complete['completed'] and complete['actual_generation']==32
    assert trainer.get('completed', trainer.get('passed', False)) and trainer.get('generation', 32)==32
    for file in [H2/'gpu_worker.lock',H2/'formal/bun392_bridge_pfsp8/worker.lock']:
        released=lock(file)
        released.close()
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(memory.splitlines())==8 and all(int(x.strip())<1024 for x in memory.splitlines()), 'GPU release required'
    write('HANDOFF.json', {'from_namespace': str(H2.relative_to(ROOT)), 'to_namespace': str(H3.relative_to(ROOT)),
        'source_generation': 32, 'verified_h2_complete': True, 'gpu_release_required': True,
        'capability_gate': False, 'status': 'verified; launching H3'})
    if not (H3/'PREPARED.json').exists():
        subprocess.run([sys.executable,'scripts/je09_stage_h3.py','prepare'], cwd=ROOT, check=True)
    write('HANDOFF.json', {'from_namespace': str(H2.relative_to(ROOT)), 'to_namespace': str(H3.relative_to(ROOT)),
        'source_generation': 32, 'verified_h2_complete': True, 'capability_gate': False,
        'status': 'prepared; starting fixed-budget H3'})
    result=subprocess.run([sys.executable,'scripts/je09_stage_h3.py','run'], cwd=ROOT)
    terminal=H3/'CAMPAIGN_COMPLETE.json'
    write('HANDOFF_COMPLETE.json', {'status': 'H3 controller returned', 'h3_namespace': str(H3.relative_to(ROOT)),
        'returncode': result.returncode, 'campaign_complete': terminal.exists(), 'fixed_budget_generations': 32})
    print('JE09 H3 terminal; requesting same-context continuation',flush=True)
    thread=os.environ.get('CODEX_THREAD_ID') or os.environ.get('CODEX_SESSION_ID')
    if thread:
        prompt=('自动接力：H3 v17 已到终态，请读取 runs/je09_bun_20261016/stage_h3_natural_combat_coverage_pfsp8_v17/'
                'state/HANDOFF_COMPLETE.json、原始events及已完成旧协议评估。在同一workspace和实验分支继续当前JE09任务。'
                '先确认技术/硬安全故障；若正常固定32代完成，快速汇总、按完整漏斗选父，建立下一独立单变量至少32代实验并实际8卡开训，'
                '不因两个短点能力早停，CPU旧协议异步不阻塞GPU；不新增网页协议、不push/main/release、不覆盖或删除旧run。'
                '技术失败先修复而不是宣称完成。同步更新当天报告/总账、本地提交。')
        write('AUTO_CONTINUATION.json',dict(requested=True,same_context=True,mechanism='codex exec resume exact inherited thread',
            scope='one continuation turn; no automatic parameter change',training_exit=result.returncode))
        with (H3/'auto_continuation.log').open('a') as log:
            child=subprocess.run(['codex','exec','resume',thread,prompt],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write('AUTO_CONTINUATION_EXIT.json',dict(returncode=child.returncode,same_context=True))
    else:
        write('AUTO_CONTINUATION.json',dict(requested=False,reason='no inherited thread identifier',requires_attention=True))

if __name__ == '__main__': main()
