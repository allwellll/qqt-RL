"""Bind frozen FUNNEL condition before its module-level fullmatch import."""
import json
import os
from pathlib import Path
import runpy
import sys

def bootstrap(protocol,condition):
    root=Path(__file__).resolve().parents[1]
    value=json.loads(Path(protocol).read_text());spec=value['conditions'][condition]
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
        FULLMATCH_OUTPUT=str(root/value['namespace']/condition),
        FULLMATCH_GAMES_PER_MODE=str(spec['games_per_mode']),
        FULLMATCH_SEED_START=str(spec['seed_start']),FULLMATCH_MODELS='')
    return dict(condition=condition,output=value['namespace']+'/'+condition,
        games_per_mode=spec['games_per_mode'],seed_start=spec['seed_start'])

def main():
    bootstrap(sys.argv[1],sys.argv[2])
    runpy.run_module('scripts.campaign_funnel_eval',run_name='__main__')

if __name__=='__main__':main()
