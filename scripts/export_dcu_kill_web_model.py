"""Export a verified DCU kill Actor for the existing browser worker ABI."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import pickle

ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,required=True,help='completed evaluation directory with frozen candidate Actor')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();os.chdir(ROOT)
    if a.output.exists():parser.error('new output directory required')
    os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                      JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import onnxruntime as ort
    import torch
    from scripts.export_web_specialist import export_checkpoint
    from scripts.export_bun_web_model import extract_transformer
    from scripts.je03_native_vit_export_20261011 import NativeViT
    meta=json.loads((a.source/'SOURCE_CHECKPOINT.json').read_text())
    evaluation=json.loads((a.source/'RESULT.json').read_text())
    if evaluation['status']!='PASS_CPU_FULL_GAME_EXPORT':raise ValueError('Completed evaluation required')
    actor=a.source/'candidate_actor.pt';expected=meta['actor_sha256']
    if digest(actor)!=expected:raise ValueError('Frozen Actor SHA mismatch')
    for file,expected_file in meta['source_sha256'].items():
        if digest(ROOT/file)!=expected_file:raise ValueError('Training source mismatch: '+file)
    receipt=export_checkpoint(actor,expected,a.output,f'dcu-kill-u{meta["update"]:06d}')
    path=a.output/'model.json';doc=json.loads(path.read_text())
    doc['meta'].update(rule_version='bun-native-v4-facing',r4_mode=0,r4_full_checkpoint_sha256=expected,
        r4_full_input_abi='obs25-global38',r4_export='exact constant-mode0 derivative, existing R3 global37 ABI',
        display_name=f'对拼强化 · u{meta["update"]}',training_mode='map806-kill',evaluation_games=evaluation['games'])
    path.write_text(json.dumps(doc,ensure_ascii=False,separators=(',',':'))+'\n')
    params=pickle.loads(actor.read_bytes())
    # Every deployed tensor must equal the original fp32 Actor, except the
    # final mode row whose input is fixed at zero. No quantization/retraining.
    derived=dict(params,state_w=params['state_w'][:-1])
    packed=np.frombuffer(base64.b64decode(doc['flat']),dtype='<f4')
    for name,array in extract_transformer(derived).items():
        start,count=doc['tensors'][name]
        np.testing.assert_array_equal(packed[start:start+count],np.asarray(array,np.float32).reshape(-1))
    full=NativeViT(params).eval()
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.output/'model.onnx'),options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(20261115);fp32_maximum=0.
    for _ in range(8):
        obs=rng.normal(0,.2,(1,25,13,15)).astype(np.float32);state=rng.uniform(-1,1,(1,37)).astype(np.float32)
        actual=session.run(None,dict(obs=obs,state=state))
        with torch.no_grad():fp32=full(torch.from_numpy(obs),torch.from_numpy(np.concatenate([state,np.zeros((1,1),np.float32)],-1)))
        if not all(np.isfinite(x.numpy()).all() and np.isfinite(y).all() for x,y in zip(fp32,actual)):raise ValueError('Nonfinite fp32 inference')
        fp32_maximum=max(fp32_maximum,max(float(np.max(np.abs(x.numpy()-y))) for x,y in zip(fp32,actual)))
    if fp32_maximum>=.002:raise ValueError('Full-source fp32/ONNX parity failed')
    receipt.update(ruleset='bun-native-v4-facing',source_update=meta['update'],evaluation_games=evaluation['games'],
                   packed_weights_exact=True,full_source_fp32_onnx_max_abs=fp32_maximum,
                   scope='Exact fp32 weights and constant-zero mode projection; browser fp32 is not bit-identical to training bf16',
                   files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in (path,a.output/'model.onnx')})
    (a.output/'EXPORT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
