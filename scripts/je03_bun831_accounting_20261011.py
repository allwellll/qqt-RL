"""Versioned telemetry correction, proven against saved Adam counters.

The sealed training kernel stays unchanged. Top25 reduces minibatch count for
Actor AND uniform Critic subsamples; raw log update counters overlooked this.
"""
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import je03_bun831_20261011 as s
from scripts.population_run_state import atomic_json,sha,lock


def actual_updates(envs,steps,minibatch,epochs,frac):
    total=envs*steps;half=max(minibatch//2,1)
    keep=min(max((int(total*frac)//half)*half,half),total) if 0<frac<1 else total
    assert keep%half==0
    return epochs*keep//half


def once():
    p=s.read(s.RUN/'plan.json');out=s.RUN/'accounting_v1';out.mkdir(exist_ok=True)
    corrected={};counter_proofs={}
    for a in s.ARMS:
        updates=actual_updates(p['envs_per_device'],p['steps'],p['minibatch_per_device'],p['epochs'],p['recipes'][a]['adv_top_frac'])
        events=s.RUN/'formal'/a/'events.jsonl'
        if not events.exists():continue
        lines=events.read_text().splitlines();rows=[]
        for line in lines:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            rows.append(dict(row,raw_optimizer_updates=row['optimizer_updates'],
                raw_optimizer_updates_per_second=row['optimizer_updates_per_second'],
                optimizer_updates=updates,optimizer_updates_per_second=updates/row['seconds'],
                accounting_version='v1 checkpoint-verified; original log retained'))
        target=out/f'events_{a}.jsonl';temp=target.with_suffix('.tmp')
        with temp.open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
            f.flush();os.fsync(f.fileno())
        temp.replace(target);corrected[a]=dict(actual_updates_per_generation=updates,
            fresh_ticks_per_generation=p['ticks_per_generation'],corrected_events=str(target),last=rows[-1] if rows else None)
        # Cache verification of actual COMMITTED states. This never loads GPU
        # tensors and never writes to the training checkpoint namespace.
        for cp in sorted((s.RUN/'formal'/a).glob('gen*/COMMITTED.json')):
            g=s.read(cp)['generation'];proof=out/f'ADAM_COUNT_{a}_g{g:06d}.json'
            if proof.exists():counter_proofs[proof.name]=sha(proof);continue
            if g not in [0,32,64,128,256,512,1024,2048,4096]:continue
            s.verify_checkpoint(cp.parent,sha(s.RUN/'plan.json'))
            with (cp.parent/'state.pkl').open('rb') as f:state=pickle.load(f)
            counts=np.asarray(state['live'][1][0].count)
            assert (counts==g*updates).all(),(a,g,counts,updates)
            atomic_json(proof,dict(passed=True,generation=g,arm=a,adam_counts=counts.tolist(),
                expected=g*updates,updates_per_generation=updates,committed_sha256=sha(cp),unix=time.time()))
            counter_proofs[proof.name]=sha(proof)
    doc=dict(passed=True,plan_sha256=sha(s.RUN/'plan.json'),kernel_unchanged=True,
        correction='base128 updates/gen; recipe32. Critic drawn uniformly from all rows but only n_keep25% each epoch, not exhaustive all-row learning.',
        raw_logs_preserved=True,arms=corrected,counter_proofs=counter_proofs,unix=time.time())
    atomic_json(out/'VERIFIED_ACCOUNTING.json',doc)
    return doc


def main():
    out=s.RUN/'accounting_v1';out.mkdir(exist_ok=True);own=lock(out/'owner.lock')
    atomic_json(out/'OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',unix=time.time()))
    while True:once();time.sleep(15)


if __name__=='__main__':main()
