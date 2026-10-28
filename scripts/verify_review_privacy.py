"""Scan tracked and staged files only; never traverse ignored training artifacts."""
import getpass
import json
import re
import subprocess
from pathlib import Path


def git_files(*args):
    return {p for p in subprocess.check_output(['git', *args, '-z']).decode().split('\0') if p}


def main():
    tracked = git_files('ls-files')
    staged = git_files('diff', '--cached', '--name-only', '--diff-filter=ACMR')
    files = sorted(tracked | staged)
    patterns = {
        'private_identity': re.compile(re.escape(getpass.getuser()), re.I),
        'private_absolute_path': re.compile(r'/(?:mnt/(?:jpfs|afs)|home|Users|root)/[^\s\"<>]+'),
        'private_ipv4': re.compile(r'\b(?:10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})\b'),
        'literal_machine_pid': re.compile(r'(?i)\b(?:process_id|process_pid|os_pid|pid)\s*[=:]\s*[0-9]{3,}\b'),
    }
    findings, exclusions = [], []
    binary = {'.png','.jpg','.jpeg','.gif','.webp','.woff','.woff2','.zip','.onnx','.npz','.ico','.wasm'}
    for name in files:
        file = Path(name)
        if not file.is_file():
            continue
        if file.is_symlink():
            findings.append(dict(file=name, kind='published_symlink'))
            continue
        if file.suffix.lower() in binary:
            continue
        value = file.read_text(errors='replace')
        for kind, pattern in patterns.items():
            for match in pattern.finditer(value):
                row = dict(file=name, kind=kind, line=value.count('\n',0,match.start())+1)
                # One historical Markdown metric heading contains slash home/capture;
                # it is a slash-separated funnel, not a local filesystem path.
                if (name == 'reports/je09_bun_20261017/route_pbrs_training_20261017.md'
                    and kind == 'private_absolute_path' and match.group() == '/'+'home/capture'
                    and value[max(0,match.start()-7):match.start()] == '25%/50%'):
                    exclusions.append(dict(**row, reason='historical funnel column heading'))
                    continue
                findings.append(row)
    ignored = subprocess.run(['git','check-ignore','-q','AGENTS.md']).returncode == 0
    result = dict(passed=not findings and ignored and 'AGENTS.md' not in tracked,
                  scope='git ls-files -z union staged ACMR files; ignored runs/checkpoints/task orchestration excluded',
                  files=len(files), staged_files=len(staged), checks=list(patterns),
                  findings=findings, reviewed_false_positives=exclusions,
                  player_id_note='pid 0/1 in game code are player indices, not machine process IDs',
                  local_agents_ignored=ignored,local_agents_tracked='AGENTS.md' in tracked)
    Path('reports/review_best_d_20261028/privacy.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Privacy', 'GREEN' if result['passed'] else 'RED',len(files),'files',len(findings),'findings')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
