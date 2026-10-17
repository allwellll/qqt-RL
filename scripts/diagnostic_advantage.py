#!/usr/bin/env python3
"""Strict policy-weighted A1 using production Sim branches and sealed ledgers."""
import argparse
import base64
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import diagnostic_reward as R
import jax
import jax.numpy as jnp
import numpy as np
from jax_bomb import jax_net

SCALE=1e-6
SCHEMA='qqt.policy-weighted-a1/v1'


def advance_linger(linger,entry):
    result=np.maximum(linger-1,0)
    for source in entry['explosions']:
        if source['owner'] in (0,1):
            for axis,key in enumerate(('horzCovered','vertCovered')):
                result[source['owner'],axis][np.array(source[key],bool).reshape(13,15)]=2
    return result


@jax.jit
def branches(carry,before,after,info):
    def evaluate(a,i):
        data=jax.tree.map(lambda x:x[None],(before,a,i))
        new,(reward,parts,own)=R.evaluate_chunk(carry,data)
        return new,reward[0],parts[0]
    return jax.vmap(evaluate)(after,info)


def calculate(candidates,taken,gamma):
    legal=[c for c in candidates if c.get('legal',True)]
    mass=sum(c['probability'] for c in legal)
    assert mass>0
    q=[c['reward']+gamma*(0 if c['terminal'] else c['value']) for c in legal]
    mean=sum(c['probability']/mass*v for c,v in zip(legal,q))
    selected=next(i for i,c in enumerate(legal) if c['action']==taken)
    return q[selected],mean,q[selected]-mean


def run(file,critic_file,verify=False,limit=None):
    doc=json.loads(file.read_text());meta=json.loads(critic_file.with_name('meta.json').read_text())
    with critic_file.open('rb') as stream:critic=pickle.load(stream)
    assert hashlib.sha256(Path(jax_net.__file__).read_bytes()).hexdigest()==meta['network_sha256'],'training forward hash'
    forward=jax.jit(lambda obs,gv:jax_net.net_forward(critic,'transformer',obs,gv)[2])
    assert jax.default_backend()=='cpu'
    env=dict(os.environ,REPLAY_STAGE='h8')
    proc=subprocess.Popen(['node','scripts/diagnostic_counterfactual.js',str(file)],stdout=subprocess.PIPE,text=True,env=env)
    first=json.loads(proc.stdout.readline())['initial'];before_raw=first['state'];prior_legal=first['legal']
    aux=R.ops.reward_state(1);aux['carry_items']=R.items.state(1)
    zeros=jnp.zeros((1,2),bool)
    carry=(aux,jnp.zeros((1,2),jnp.int16),zeros,zeros,zeros)
    linger=np.zeros((2,2,13,15),np.int8);causes=[[False,False],[False,False]]
    data=dict(schema=SCHEMA,scale=SCALE,ticks=[],taken_q=[],mean_q=[],advantage=[],delta=[],legal_count=[])
    reward_rows=dict((r[0],r[1]) for r in doc['rewards'])
    digest=hashlib.sha256();last=None;sample_obs=[];sample_gv=[];sample_values=[]
    # These sample positions are frozen before inference; no reward-based picks.
    chosen=set(np.random.default_rng(doc['meta']['seed']^0xA100).choice(np.arange(1,len(doc['actions'])+1),size=8,replace=False).tolist())|{1,len(doc['actions'])}
    samples=chosen|{t-1 for t in chosen if t>1}
    scene={'native':8,'open':0,'carry_home':2}[doc['meta']['scene']]
    start=time.monotonic();sample_carries=[]
    for line in proc.stdout:
        packet=json.loads(line);tick=packet['tick'];cs=packet['candidates'];n=len(cs)
        afters=[];infos=[];causal=[];lingers=[]
        before=R.convert(before_raw,linger,scene)
        if tick in samples:sample_carries.append(dict(tick=tick,carry=jax.tree.map(np.asarray,carry),before_raw=before_raw,prior_legal=prior_legal,causes=copy.deepcopy(causes),linger=linger.copy()))
        for c in cs:
            entry=c['entry'];entry['prior_legal']=prior_legal
            cause=copy.deepcopy(causes);info=R.observed(before_raw,entry,cause)
            li=advance_linger(linger,entry)
            afters.append(R.convert(entry['state'],li,scene));infos.append(info);causal.append(cause);lingers.append(li)
        # Fixed shape prevents recompilation when masks change. Padding has no weight.
        indices=list(range(n))+[n-1]*(15-n)
        after=jax.tree.map(lambda *x:np.stack(x),*[afters[i] for i in indices])
        info=jax.tree.map(lambda *x:np.stack(x),*[infos[i] for i in indices])
        new,rewards,parts=branches(carry,before,after,info)
        rewards=np.asarray(rewards)[:,0,0];parts=np.asarray(parts)[:,0,0]
        obs=np.stack([np.frombuffer(base64.b64decode(cs[i]['obs']),np.float32).reshape(24,13,15) for i in indices])
        gv=np.stack([np.frombuffer(base64.b64decode(cs[i]['global']),np.float32) for i in indices])
        values=np.asarray(forward(obs,gv))
        assert np.isfinite(values).all() and np.isfinite(rewards).all()
        selected=next(i for i,c in enumerate(cs) if c['action']==packet['taken'])
        c=cs[selected]
        expected=reward_rows.get(tick,{})
        actual={k:float(v) for k,v in zip(R.COMPONENTS,parts[selected]) if v!=0}
        assert actual==expected,f'taken counterfactual ledger components @{tick}: {actual} != {expected}'
        candidates=[dict(action=c['action'],probability=c['probability'],reward=float(rewards[i]),value=float(values[i]),terminal=c['terminal']) for i,c in enumerate(cs)]
        tq,mq,a=calculate(candidates,packet['taken'],meta['gamma'])
        delta=None if last is None else a-last;last=a
        data['ticks'].append(tick);data['legal_count'].append(n)
        for key,v in zip(('taken_q','mean_q','advantage','delta'),(tq,mq,a,delta)):
            data[key].append(None if v is None else round(v/SCALE))
        digest.update(json.dumps(dict(tick=tick,candidates=candidates),sort_keys=True,separators=(',',':')).encode())
        if tick in samples:
            sample_obs.append(obs.copy());sample_gv.append(gv.copy());sample_values.append(values.copy())
        carry=jax.tree.map(lambda x:x[selected],new)
        before_raw=c['entry']['state'];prior_legal=c['entry']['legal'];causes=causal[selected];linger=lingers[selected]
        if tick%100==0:print(file.stem,tick,'A',round(a,6),'elapsed',round(time.monotonic()-start,1),flush=True)
        if limit and tick>=limit:
            proc.terminate();proc.wait();return
    assert proc.wait()==0,'production branch stream failed'
    assert len(data['ticks'])==len(doc['actions']),'incomplete A1 stream'
    result=dict(schema=SCHEMA,status='recorded',reason='all legal joint branches measured on production Sim with cloned ledger and RNG',
        definition='Q1=r+gamma*V(next); A1=Q1(taken)-sum(pi_legal*Q1)',gamma=meta['gamma'],
        action_scope='joint_move_ability',legal_actions='production legalMask.mm x TransformerModel._abilityMask; item use included as ability=2; masked head softmax product renormalized over legal joint actions',
        probability_source='production ONNX actor logits, independent masked move and ability heads; greedy action selected, stochastic policy used as comparison distribution',
        checkpoint_sha256=doc['meta']['checkpoint_sha256'],policy_sha256=doc['meta']['model_sha256'],
        critic_sha256=meta['critic_sha256'],simulator_sha256=doc['meta']['simulator_sha256'],
        reward_hashes=doc['meta']['reward_hashes'],counterfactual_sha256=digest.hexdigest(),
        measured_count=len(data['ticks']),missing_count=0,missing_reasons={},quantization_error=.5*SCALE,
        calculation_version='production-clone-ledger-bf16-critic/v1',rng_status='taken branch exact snapshot and info parity on every tick; same seeded draw position for each legal branch',
        critic_network_sha256=meta['network_sha256'],input_encoding=meta['input_encoding'])
    if verify:
        assert data==doc['advantages'],'independent A1 recomputation differs'
        assert result['counterfactual_sha256']==doc['meta']['advantage']['counterfactual_sha256']
        file.with_suffix('.ledger-fixtures.pkl').write_bytes(pickle.dumps(sample_carries,protocol=4))
        print(file.stem,'INDEPENDENT A1 EXACT',flush=True)
    else:
        doc['advantages']=data;doc['meta']['advantage']=result
        file.write_text(json.dumps(doc,ensure_ascii=False,separators=(',',':')))
        np.savez_compressed(file.with_suffix('.critic-fixtures.npz'),obs=np.concatenate(sample_obs),global_state=np.concatenate(sample_gv),values=np.concatenate(sample_values))
        file.with_suffix('.ledger-fixtures.pkl').write_bytes(pickle.dumps(sample_carries,protocol=4))
        print(file.stem,'MEASURED',len(data['ticks']),digest.hexdigest(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+',type=Path);ap.add_argument('--critic',type=Path,required=True);ap.add_argument('--verify',action='store_true');ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    for f in args.files:run(f,args.critic,args.verify,args.limit)
