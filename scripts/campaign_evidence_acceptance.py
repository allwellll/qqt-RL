"""Accept registered scientific evidence only after real datasets exist."""
import json
from pathlib import Path
import sys
from scripts.population_run_state import atomic_json,sha


def main():
    path=Path(sys.argv[1]);p=json.loads(path.read_text());out=Path(p['namespace']);evidence={}
    for name in p['required_evidence']:
        f=out/name
        value=json.loads(f.read_text())
        if value.get('scientific_acceptance') is not True:
            raise ValueError('Missing scientific acceptance; audit-only outputs are insufficient')
        if not value.get('raw_sources'):
            raise ValueError('No recomputable raw source evidence')
        for source in value['raw_sources']:
            if sha(source['path'])!=source['sha256']:raise ValueError('Raw evidence hash mismatch')
        evidence[name]=sha(f)
    atomic_json(out/'COMPLETE.json',dict(passed=True,protocol_sha256=sha(path),evidence=evidence))


if __name__=='__main__':main()
