"""Exact packed tensor audit plus fp32 Torch/JS/ONNX on genuine states."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import pickle
import json
from pathlib import Path
import numpy as np
import torch
from scripts.je03_native_vit_export_20261011 import NativeViT
from scripts.population_run_state import atomic_json,sha

def main():
 b=Path(os.environ.get('V2_SELECTION_OUTPUT','runs/v2_fresh_selection_20261031'))/'publication';m=json.loads((b/'manifest.json').read_text());row=json.loads((b/'CATALOG_ROW.json').read_text());f=json.loads((b/'JS_ONNX.json').read_text());assert f['model_sha256']==row['sha256']
 cp=Path(m['source_ref']);assert sha(cp/'policy.pt')==row['source_sha256']
 with (cp/'policy.pt').open('rb') as stream:params=pickle.load(stream)
 torch.set_num_threads(1);model=NativeViT(params).eval();rows=[]
 for sample in f['samples']:
  with torch.no_grad():move,bomb,value=model(torch.from_numpy(np.array(sample['obs'],dtype=np.float32).reshape(1,*row['onnx']['meta']['obs_shape'])),torch.from_numpy(np.array(sample['state'],dtype=np.float32).reshape(1,-1)))
  expected=dict(move=move.numpy()[0],bomb=bomb.numpy()[0],value=value.numpy()[0]);errors={}
  for backend in ('js','onnx'):
   errors[backend]={k:float(np.max(np.abs(v-sample[backend][k]))) for k,v in expected.items()};assert max(errors[backend].values())<.002,errors
   for k in ('move','bomb'):
    mask=np.array(sample['masks'][k],dtype=bool);assert np.argmax(np.where(mask,expected[k],-1e9))==np.argmax(np.where(mask,sample[backend][k],-1e9))
  rows.append(dict(tick=sample['sample'],seat=sample['seat'],errors=errors,masked_greedy_exact=True))
 exact=json.loads((b/'EXPORT_EXACT.json').read_text());assert exact['packed_weights_exact'] and exact['model_sha256']==row['sha256']
 atomic_json(b/'PYTHON_PARITY.json',dict(status='PASS',passed=True,packed_weights_exact=True,policy_sha256=row['source_sha256'],model_sha256=row['sha256'],samples=rows,scope='fp32 deployment parity; training bf16 is not asserted bit-identical'))
 print('winner Torch fp32/JS/ONNX and masks PASS',len(rows))
if __name__=='__main__':main()
