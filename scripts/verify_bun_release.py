"""CPU Python fp32/packed JSON/JS/ONNX parity on legal native Bun states.

Training JAX uses bf16; record its precision drift separately from the frozen
fp32 web authority. Never silently widen the fp32 export tolerance.
"""
import argparse
import base64
import json
import os
import pickle
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax_bomb.jax_net import transformer_forward
from scripts.export_bun_web_model import extract_transformer
from scripts.export_bun_release import sha, validate_native
from scripts.je03_native_vit_export_20261011 import NativeViT


def verify(checkpoint, model_file, fixtures_file):
    cp=Path(checkpoint);doc=json.loads(Path(model_file).read_text())
    committed=json.loads((cp/'COMMITTED.json').read_text())
    policy=cp/('ema_policy.pt' if doc['meta']['strategy']=='ema' else 'policy.pt')
    assert sha(policy)==doc['meta']['checkpoint_sha256']
    assert sha(cp/'COMMITTED.json')==doc['meta']['committed_sha256']
    assert committed['passed'] and jax.default_backend()=='cpu'
    with policy.open('rb') as stream:params=pickle.load(stream)
    validate_native(params)
    flat=np.frombuffer(base64.b64decode(doc['flat']),dtype='<f4')
    expected=extract_transformer(params)
    assert set(expected)==set(doc['tensors'])
    for name,value in expected.items():
        start,count=doc['tensors'][name]
        assert np.array_equal(flat[start:start+count],value.reshape(-1)),name
    fixtures=json.loads(Path(fixtures_file).read_text())
    assert fixtures['model_sha256']==sha(model_file)
    torch.set_num_threads(1)
    native=NativeViT(params).eval()
    actor=jax.tree.map(jnp.asarray,params);rows=[]
    for row in fixtures['samples']:
        obs=jnp.asarray(row['obs'],dtype=jnp.float32).reshape(1,24,13,15)
        state=jnp.asarray(row['state'],dtype=jnp.float32).reshape(1,24)
        with torch.no_grad():
            move,bomb,value=native(torch.from_numpy(np.asarray(obs).copy()),torch.from_numpy(np.asarray(state).copy()))
        values=dict(move=move.numpy()[0],bomb=bomb.numpy()[0],value=value.numpy()[0])
        bm,bb,bv,_=transformer_forward(actor,obs,state)
        training=dict(move=np.asarray(bm)[0],bomb=np.asarray(bb)[0],value=np.asarray(bv)[0])
        bf16={key:float(np.max(np.abs(values[key]-training[key]))) for key in values}
        errors={}
        for backend in ['js','onnx']:
            errors[backend]={key:float(np.max(np.abs(values[key]-row[backend][key]))) for key in values}
            assert max(errors[backend].values())<.002,errors
            for key in ['move','bomb']:
                mask=np.asarray(row['masks'][key],dtype=bool)
                assert np.argmax(np.where(mask,values[key],-1e9))==np.argmax(np.where(mask,row[backend][key],-1e9))
        rows.append(dict(sample=row['sample'],seat=row['seat'],max_errors=errors,masked_greedy_exact=True,
                         training_bf16_diagnostic_errors=bf16))
    return dict(passed=True,backend='cpu',python_precision='fp32 frozen web authority',training_precision='bf16 diagnostic only',
                packed_weights_exact=True,policy_sha256=sha(policy),
                model_sha256=sha(model_file),onnx_sha256=fixtures['onnx_sha256'],samples=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--fixtures',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    result=verify(a.checkpoint,a.model,a.fixtures)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
