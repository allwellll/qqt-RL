"""Scan tracked files plus explicitly included public artifacts, not local runs.

Set PRIVATE_DENY_TOKENS to comma-separated private identifiers. Tokens never
appear in findings. This scans current content; it does not rewrite Git history.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess

PID = re.compile(r'(?i)(?:\bPID[\s:_=-]*\d{4,}|\b(?:worker|owner|evaluator)\d{4,}|'
                 r'"[a-z_]*pid"\s*:\s*\d{4,}|pid\d{4,})')
IP = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])|'
                r'(?i:\b(?:fc|fd|fe80)[0-9a-f:]+:[0-9a-f:]+)')
NETWORKS = [ipaddress.IPv4Network(n) for n in
            [(0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16), (0xA9FE0000, 16)]]
NETWORKS += [ipaddress.IPv6Network(n) for n in
             [(0xFC << 120, 7), (0xFE80 << 112, 10)]]


def findings(content, tokens):
    result = []
    if any(token and token.casefold() in content.casefold() for token in tokens):
        result.append('private identifier')
    if PID.search(content):
        result.append('machine PID literal')
    for match in IP.finditer(content):
        try:
            ip = ipaddress.ip_address(match.group())
        except ValueError:
            continue
        if any(ip.version == net.version and ip in net for net in NETWORKS):
            result.append('private IP literal'); break
    return result


def scan(root, tokens, include=()):
    root = Path(root)
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    result = []
    for name in sorted(set(filter(None, tracked)) | set(include)):
        p = Path(name) if Path(name).is_absolute() else root / name
        data = p.read_bytes().decode('utf-8', errors='replace')
        # Packed fp32 base64 can coincidentally spell PID-like tokens; those are
        # numbers, not machine metadata. Still scan the entire file for private
        # identifiers, and scan every metadata/tensor-index field for IP/PID.
        machine_text = data
        if p.suffix == '.json':
            try:
                doc = json.loads(data)
                if (isinstance(doc, dict) and isinstance(doc.get('tensors'), dict)
                        and doc.get('meta', {}).get('arch') in ('transformer', 'mlp4', 'mlp', 'cnn')):
                    machine_text = json.dumps({k:v for k,v in doc.items() if k != 'flat'})
            except (ValueError, TypeError):
                pass
        issues = findings(name + '\n' + machine_text, [])
        if any(t and t.casefold() in (name+'\n'+data).casefold() for t in tokens):
            issues.insert(0, 'private identifier')
        if issues:
            # Do not print matched text or external/local path strings.
            label = name if not Path(name).is_absolute() else '<EXTERNAL_ARTIFACT>'
            if any(t and t.casefold() in label.casefold() for t in tokens):
                label = '<PRIVATE_FILE>'
            result.append(dict(file=label, issues=issues))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--include', action='append', default=[])
    a = p.parse_args()
    tokens = list(filter(None, os.environ.get('PRIVATE_DENY_TOKENS', '').split(',')))
    if not tokens:
        p.error('PRIVATE_DENY_TOKENS must declare identifiers for a release scan')
    failures = scan(a.root, tokens, a.include)
    for row in failures:
        print(row)
    print(f'privacy scan: {len(failures)} findings (git ls-files + explicit artifacts)')
    raise SystemExit(bool(failures))
