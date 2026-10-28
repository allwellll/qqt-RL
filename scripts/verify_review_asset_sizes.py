"""Bound every indexed file and every new branch blob to 10 decimal MB."""
import json
import subprocess
from pathlib import Path

LIMIT = 10_000_000
BASE = '7128071'
MODEL = 'web/review-evidence/best-d-20261028/model.json'


def objects(rows):
    if not rows:
        return []
    output = subprocess.check_output(
        ['git', 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
        input=''.join(oid+'\n' for oid, name in rows).encode()).decode().splitlines()
    result = []
    for (oid, name), line in zip(rows, output, strict=True):
        actual, kind, size = line.split()
        assert actual == oid
        if kind == 'blob':
            result.append(dict(file=name, bytes=int(size)))
    return result


def main():
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).decode().split('\0')
    indexed = []
    for entry in filter(None, entries):
        meta, name = entry.split('\t', 1)
        mode, oid, stage = meta.split()
        assert stage == '0', 'unmerged index'
        indexed.append((oid, name))
    files = objects(indexed)
    assert MODEL not in {name for oid, name in indexed}, 'D model must remain outside Git'
    history = subprocess.check_output(['git', 'rev-list', '--objects', BASE+'..HEAD']).decode().splitlines()
    rows = [(parts[0], parts[1] if len(parts)>1 else '') for line in history if (parts:=line.split(' ', 1))]
    new_blobs = objects(rows)
    assert MODEL not in {row['file'] for row in new_blobs}, 'D model entered new branch history'
    large = [row for row in files+new_blobs if row['bytes']>LIMIT]
    result = dict(passed=not large,limit_bytes=LIMIT,indexed_files=len(files),
                  maximum=max(files,key=lambda row:row['bytes']),oversized=large,
                  new_branch_blobs=len(new_blobs),new_branch_maximum=max(new_blobs,key=lambda row:row['bytes']) if new_blobs else None,
                  history_base=BASE,history_checked_through=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  model_tracked=False,model_in_new_branch_history=False,
                  scope='All indexed files plus reachable blobs added after baseline; old comparison branch excluded.')
    Path('reports/review_best_d_20261028/asset-sizes.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Asset sizes', 'GREEN' if result['passed'] else 'RED',result['maximum'])
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
