"""Single CPU watchdog for the bounded 16:20 evaluation and final analysis."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import json
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

OUT = Path('runs/r3_ownside_recheck_1620_20261031')
MODULE = 'scripts.r3_ownside_recheck_1620_20261031'


def read(p):
    return json.loads(Path(p).read_text())


def alive():
    result = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = (d/'cmdline').read_bytes().split(b'\0')
            if MODULE.encode() in args and (d/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                result.append(d)
        except OSError:
            pass
    return result


def notify(message, receipt):
    r = subprocess.run(['jdme-notify', 'send', message], capture_output=True, text=True)
    atomic_json(OUT/receipt, dict(message=message, returncode=r.returncode, unix=time.time()))


def main():
    guard = lock(OUT/'WATCHDOG.lock')
    while True:
        if (OUT/'BLOCKED.json').exists():
            notify('R3 16:20 CPU复核技术阻塞；详见runs/r3_ownside_recheck_1620_20261031/BLOCKED.json。训练未停、不发布。', 'BLOCKED_NOTIFIED.json')
            return
        if (OUT/'COMPLETE.json').exists():
            doc = read(OUT/'COMPLETE.json')
            assert sha(OUT/'RESULTS.json') == doc['results_sha256']
            with (OUT/'ANALYSIS.log').open('ab') as f:
                r = subprocess.run(['taskset', '-c', '146-147', '.venv/bin/python', '-m',
                                    'scripts.r3_ownside_recheck_analysis_20261031'], stdout=f, stderr=subprocess.STDOUT)
            if r.returncode:
                atomic_json(OUT/'ANALYSIS_BLOCKED.json', dict(status='BLOCKED', reason='see ANALYSIS.log', unix=time.time()))
                notify('R3 16:20全部局已完成，附加配对分析技术阻塞，见ANALYSIS.log；训练继续、不发布。', 'ANALYSIS_BLOCKED_NOTIFIED.json')
                return
            doc = read(OUT/'COMPLETE.json')
            report = Path('reports/r3_ownside_recheck_1620_20261031.md')
            assert sha(report) == doc['report_sha256']
            assert sha(OUT/'COMPARISON_ANALYSIS.json') == doc['comparison_analysis_sha256']
            atomic_json(OUT/'TASK_FINAL_ACCEPTANCE.json', dict(status='COMPLETE', games=doc['games'],
                selection=doc['selection'], results_sha256=doc['results_sha256'],
                analysis_sha256=doc['comparison_analysis_sha256'], report_sha256=doc['report_sha256'],
                no_publish=True, training_untouched=True, unix=time.time()))
            notify(f"R3配对复核与等N holdout分析全部完成，建议={doc['selection']['winner']}；2816完整局、逐档胜负差/击杀率及CI已落盘。等待编排确认、不发布，训练继续。", 'FINAL_NOTIFIED.json')
            return
        processes = alive()
        assert len(processes) <= 1, 'duplicate evaluator owner'
        if not processes:
            with (OUT/'OWNER.log').open('ab') as f:
                subprocess.Popen(['taskset', '-c', '112-113,146-163', '.venv/bin/python', '-m', MODULE],
                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(OUT/'WATCHDOG_RESTART.json', dict(status='CPU_OWNER_RESTARTED_ADOPT_EXISTING_SHARDS', unix=time.time()))
        atomic_json(OUT/'WATCHDOG_STATUS.json', dict(status='MONITORING_CPU_EVALUATION', owners=len(processes), unix=time.time()))
        time.sleep(10)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise SystemExit('existing CPU watchdog')
