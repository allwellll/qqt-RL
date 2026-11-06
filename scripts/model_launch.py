"""Generic public process names with private source/argument mappings."""
import json
import os
import subprocess
import sys
from pathlib import Path

PRIVATE = Path('local_config')
def public_project():
    # Optional caller-selected alias; never depend on a machine's sibling tree.
    project = Path(__file__).resolve().parents[1]
    alias = Path(os.environ.get('QQT_PROJECT_ROOT', project))
    if alias.resolve() != project:
        raise ValueError('QQT_PROJECT_ROOT must resolve to this repository')
    return alias

def public_python():
    return os.environ.get('PYTHON', sys.executable)

def manifest(value):
    # 11:55 fix: one subdirectory per day (local_config/YYYYMMDD/), so the flat 0000-9999 namespace can no
    # longer fill up (it did on 2026-11-06 and blocked a supervisor relaunch). Existing flat configs keep working.
    import time
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    day = PRIVATE / time.strftime('%Y%m%d'); day.mkdir(mode=0o700, exist_ok=True)
    for index in range(100000):
        name = day / f'parameters_{index:05d}.json'
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
        return str(name)
    raise RuntimeError('configuration namespace full')

def command(role, config, node=False):
    assert role in ('training', 'evaluation', 'diagnostics')
    return (['node', 'model_evaluation.js', config] if node else
            [public_python(), '-u', f'model_{role}.py', config])

def visible_processes():
    processes = {}
    roots = set()
    for filename in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = filename.read_bytes().decode().split('\0')[:-1]
            parent = int(next(line.split()[1] for line in (filename.parent/'status').read_text().splitlines() if line.startswith('PPid:')))
            pid = int(filename.parent.name)
            processes[pid] = (parent, argv)
            if (filename.parent/'cwd').resolve() == Path.cwd() and any(x in argv for x in ('model_training.py','model_evaluation.py','model_diagnostics.py','model_evaluation.js','r.py','a.js')):
                roots.add(pid)
        except (OSError, UnicodeError, StopIteration):
            continue
    for _ in range(12):
        roots.update(pid for pid,(parent,argv) in processes.items() if parent in roots)
    rows = []
    for pid in sorted(roots):
        args = subprocess.run(['ps','-o','args=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
        if args:
            rows.append(dict(args=args, legacy=any(x in processes[pid][1] for x in ('r.py','a.js'))))
    return rows
