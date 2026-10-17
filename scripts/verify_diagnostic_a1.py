#!/usr/bin/env python3
"""Independent sampled A1 check against the original, immutable CPU checkpoint."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

import diagnostic_advantage as A
import diagnostic_reward as R
import jax
import numpy as np
from jax_bomb import jax_net


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--critic-meta',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('files',nargs='+',type=Path);args=ap.parse_args()
    meta=json.loads(args.critic_meta.read_text());receipt=json.loads((args.checkpoint/'COMMITTED.json').read_text())
    assert receipt['state_sha256']==meta['state_sha256'] and receipt['generation']==32
    digest=hashlib.sha256()
    with (args.checkpoint/'state.pkl').open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    assert digest.hexdigest()==receipt['state_sha256']
    plan_file=args.checkpoint.parents[2]/'plan.json'
    assert hashlib.sha256(plan_file.read_bytes()).hexdigest()==meta['plan_sha256']
    assert json.loads(plan_file.read_text())['gamma']==meta['gamma']
    assert hashlib.sha256(Path(jax_net.__file__).read_bytes()).hexdigest()==meta['network_sha256']
    with (args.checkpoint/'state.pkl').open('rb') as f:state=pickle.load(f)
    critic=jax.tree.map(lambda a:np.asarray(a)[0],state['live']['params']['critic'])
    forward=jax.jit(lambda obs,gv:jax_net.net_forward(critic,'transformer',obs,gv)[2])
    assert jax.default_backend()=='cpu'
    evidence=[]
    for file in args.files:
        doc=json.loads(file.read_text());fixtures=pickle.loads(file.with_suffix('.ledger-fixtures.pkl').read_bytes())
        ticks=[f['tick'] for f in fixtures];by_tick={f['tick']:f for f in fixtures}
        proc=subprocess.Popen(['node','scripts/diagnostic_counterfactual.js',str(file),','.join(map(str,ticks))],stdout=subprocess.PIPE,text=True,env=dict(os.environ,REPLAY_STAGE='h8'))
        json.loads(proc.stdout.readline())
        computed={};largest=0.;digest=hashlib.sha256()
        scene={'native':8,'open':0,'carry_home':2}[doc['meta']['scene']]
        for line in proc.stdout:
            packet=json.loads(line);tick=packet['tick'];f=by_tick[tick];cs=packet['candidates'];n=len(cs)
            afters=[];infos=[]
            before=R.convert(f['before_raw'],f['linger'],scene)
            for c in cs:
                e=c['entry'];e['prior_legal']=f['prior_legal'];cause=copy.deepcopy(f['causes'])
                infos.append(R.observed(f['before_raw'],e,cause));afters.append(R.convert(e['state'],A.advance_linger(f['linger'],e),scene))
            ix=list(range(n))+[n-1]*(15-n)
            after=jax.tree.map(lambda *x:np.stack(x),*[afters[i] for i in ix]);info=jax.tree.map(lambda *x:np.stack(x),*[infos[i] for i in ix])
            _,rewards,_=A.branches(f['carry'],before,after,info);rewards=np.asarray(rewards)[:,0,0]
            import base64
            obs=np.stack([np.frombuffer(base64.b64decode(cs[i]['obs']),np.float32).reshape(24,13,15) for i in ix])
            gv=np.stack([np.frombuffer(base64.b64decode(cs[i]['global']),np.float32) for i in ix])
            values=np.asarray(forward(obs,gv))
            candidates=[dict(action=c['action'],probability=c['probability'],reward=float(rewards[i]),value=float(values[i]),terminal=c['terminal']) for i,c in enumerate(cs)]
            result=A.calculate(candidates,packet['taken'],meta['gamma']);computed[tick]=result[2]
            idx=doc['advantages']['ticks'].index(tick)
            for key,value in zip(('taken_q','mean_q','advantage'),result):
                err=abs(value-doc['advantages'][key][idx]*A.SCALE);largest=max(largest,err);assert err<=A.SCALE*.50001,(tick,key,err)
            if tick-1 in computed:
                expected=doc['advantages']['delta'][idx]
                assert expected is not None
                err=abs(result[2]-computed[tick-1]-expected*A.SCALE);largest=max(largest,err);assert err<=A.SCALE*.50001
            digest.update(json.dumps(candidates,sort_keys=True).encode())
        assert proc.wait()==0 and set(computed)==set(ticks)
        e=dict(replay_id=doc['meta']['id'],ticks=ticks,independent_counterfactual_ticks=len(ticks),original_checkpoint_critic=True,cloned_rng=True,quantization_error_max=largest,summary_sha256=digest.hexdigest())
        evidence.append(e);print(json.dumps(e),flush=True)
    result=dict(schema='qqt.a1-independent-verification/v1',cpu_only=True,checkpoint_sha256=meta['checkpoint_sha256'],critic_sha256=meta['critic_sha256'],state_sha256=meta['state_sha256'],sampler='fixed PCG64(seed xor 0xA100), eight random ticks plus first/terminal and their predecessors',evidence=evidence,passed=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
