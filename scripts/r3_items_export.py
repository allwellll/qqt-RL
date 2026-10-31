"""CPU export of R3's explicit 25-plane/37-global evaluation ABI."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json
import pickle
import sys
from pathlib import Path
import torch
from scripts.je03_native_vit_export_20261011 import NativeViT
from scripts.export_bun_web_model import extract_transformer,pack_tensors
from scripts.population_run_state import atomic_json,sha


def export(cp,dest):
    cp=Path(cp);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    parent=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'policy.pt')==parent['policy_sha256']
    receipt=dest/'EXPORT.json'
    if receipt.exists():
        d=json.loads(receipt.read_text());assert all(sha(Path(f))==h for f,h in d['files'].items());return d
    p=pickle.load((cp/'policy.pt').open('rb'))
    return export_weights(p,dest,parent)


def export_weights(p,dest,parent):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);receipt=dest/'EXPORT.json'
    assert p['tok'][0].shape[0]==225 and p['state_w'].shape[0]==37
    torch.set_num_threads(1);onnx=dest/'model.onnx';model=NativeViT(p).eval()
    with torch.no_grad():torch.onnx.export(model,(torch.zeros((1,25,13,15)),torch.zeros((1,37))),str(onnx)+'.tmp',
        input_names=['obs','state'],output_names=['move','bomb','value'],opset_version=17,dynamo=False)
    Path(str(onnx)+'.tmp').replace(onnx)
    flat,tensors=pack_tensors(extract_transformer(p));f=dest/'model.json'
    atomic_json(f,dict(meta=dict(name=dest.name,display_name=dest.name,rule='bun',qqt_map_id=806,arch='transformer',
        move_actions=5,ability_actions=3,ability_encoding=['none','bomb','use_item'],value_min=-20.,value_max=20.,
        obs_shape=[25,13,15],state_dim=37,n_players=2,embed=p['tok'][0].shape[1],patch=3,depth=len(p['blocks']),
        r3_input_encoding='r3-items-route-chain-visible-v1',checkpoint_sha256=parent['policy_sha256'],onnx_path=str(onnx),onnx_sha256=sha(onnx)),flat=flat,tensors=tensors))
    d=dict(status='PASS',checkpoint=parent,model_json=str(f),model_sha256=sha(f),files={str(z):sha(z) for z in (f,onnx)})
    atomic_json(receipt,d);return d


if __name__=='__main__':export(*sys.argv[1:])
