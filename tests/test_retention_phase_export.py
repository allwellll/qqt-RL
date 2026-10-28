"""Cross-runtime features and real native inference, CPU-only preparation."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

from scripts import retention_phase_conditioning as phase
from scripts import retention_phase_export as exporter
from scripts.export_bun_web_model import extract_transformer, pack_tensors
from scripts.je03_native_vit_export_20261011 import NativeViT
from jax_bomb import jax_net as net

ROOT = Path(__file__).resolve().parents[1]


def scenarios():
    rng = np.random.default_rng(831)
    obs = np.zeros((16, 24, 13, 15), np.float32)
    state = np.zeros((16, 24), np.float32)
    for i in range(16):
        row, col = rng.integers([0, 0], [13, 15])
        obs[i, 0, row, col] = 1
        obs[i, 14, 1:4, 1:4] = 1
        state[i, 9:11] = 1
        state[i, 20] = 1
        if i % 5 == 1:
            state[i, 10] = 0 if i % 2 else 1.5
        if i % 5 == 2:
            obs[i, 17, 8, 8] = 1
        if i % 5 >= 3:
            state[i, 13:15] = 1
        if i % 5 == 4:
            obs[i, 0] = 0; obs[i, 0, 2, 2] = 1
    return obs, state


def run_node(script, payload):
    result = subprocess.run(['node', '-e', script], input=json.dumps(payload),
                            capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(result.stdout)


def test_js_torch_jax_phase_truths_and_no_input_mutation():
    obs, state = scenarios()
    expected = np.asarray(phase.features(obs, state))
    actual = exporter.features(torch.from_numpy(obs), torch.from_numpy(state)).numpy()
    np.testing.assert_array_equal(actual, expected)
    payload = dict(obs=obs.reshape(16, -1).tolist(), state=state.tolist())
    actual = run_node("""
const p=JSON.parse(require('fs').readFileSync(0,'utf8'));
const f=require('./web/retention_phase_conditioning.js');
const before=JSON.stringify(p);
const result=p.obs.map((o,i)=>f.features(o,p.state[i]));
if(JSON.stringify(p)!==before)throw Error('input mutated');
process.stdout.write(JSON.stringify(result));
""", payload)
    np.testing.assert_array_equal(actual, expected)


def test_raw24_onnx_js_reference_and_torch_conditioning(tmp_path):
    torch.set_num_threads(1)
    base = net.init_transformer(jax.random.PRNGKey(91), 24, 13, 15, embed=8, depth=1)
    p = phase.migrate(base)
    p['state_w'] = p['state_w'].at[24:].set(
        jax.random.normal(jax.random.PRNGKey(27), (5, 8))*.5)
    p = jax.tree.map(np.asarray, p)
    obs, state = scenarios()
    model = exporter.PhaseViT(p).eval()
    # Native Torch reference with explicit appended fields must be exact.
    with torch.no_grad():
        tensors = [torch.from_numpy(obs), torch.from_numpy(state)]
        outputs = model(*tensors)
        encoded = torch.cat([tensors[1], exporter.features(*tensors)], -1)
        reference = NativeViT(p).eval()(tensors[0], encoded)
    for a, b in zip(outputs, reference):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    policy = tmp_path/'policy.pt'
    with policy.open('wb') as stream:
        pickle.dump(p, stream)
    exporter.export(policy, tmp_path/'model.onnx')
    flat, index = pack_tensors(extract_transformer(p))
    document = dict(meta=dict(arch='transformer', rule='bun', embed=8, patch=3,
                             depth=1, obs_shape=[24, 13, 15], value_min=-20., value_max=20.,
                             ability_actions=3), flat=flat, tensors=index)
    payload = dict(model=document, obs=obs.reshape(16, -1).tolist(), state=state.tolist(),
                   onnx=str(tmp_path/'model.onnx'))
    actual = run_node("""
const p=JSON.parse(require('fs').readFileSync(0,'utf8'));
const Q=require('./web/sim.js');require('./web/retention_phase_conditioning.js');
const ort=require('onnxruntime-web');
ort.env.wasm.numThreads=1;
(async()=>{const model=new Q.TransformerModel(p.model);
const session=await ort.InferenceSession.create(p.onnx,{executionProviders:['wasm']});
const result=[];
for(let i=0;i<p.obs.length;i++){
const obs=Float32Array.from(p.obs[i]),state=Float32Array.from(p.state[i]);
const out=await session.run({obs:new ort.Tensor('float32',obs,[1,24,13,15]),state:new ort.Tensor('float32',state,[1,24])});
const js=model.forward(obs,state);
result.push({js:{move:Array.from(js.move),bomb:Array.from(js.bomb),value:js.value},
onnx:{move:Array.from(out.move.data),bomb:Array.from(out.bomb.data),value:out.value.data[0]}});
}process.stdout.write(JSON.stringify(result));})().catch(e=>{console.error(e);process.exit(1)});
""", payload)
    for i, row in enumerate(actual):
        for k, index in [('move', 0), ('bomb', 1), ('value', 2)]:
            expected = outputs[index].numpy()[i]
            np.testing.assert_allclose(row['onnx'][k], expected, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(row['js'][k], expected, atol=2e-5, rtol=2e-5)


def test_install_is_process_local_and_a0_path_exact():
    # A new process exercises production net_forward import bindings; no R4
    # module or host process is patched by this test.
    code = """
import jax,jax.numpy as jnp,numpy as np
from jax_bomb.jax_net import net_forward
from scripts import retention_phase_conditioning as p
from scripts.je03_bun_stage1_20261011 import split_forward
from jax_bomb.jax_net import init_transformer
q=init_transformer(jax.random.PRNGKey(4),24,13,15,embed=8,depth=1)
o=jnp.zeros((1,24,13,15));s=jnp.zeros((1,24)).at[:,9:11].set(1)
before=net_forward(q,'transformer',o,s)
p.install()
after=net_forward(q,'transformer',o,s)
assert all(np.array_equal(x,y) for x,y in zip(before,after))
pair=p.migrate(dict(actor=q,critic=q))
out=split_forward(pair,'transformer',o,s,net_forward)
assert all(np.array_equal(x,y) for x,y in zip(before,out))
print('passed')
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, env=dict(os.environ,
                                CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAXBOMB_RULE='bun'))
    assert result.returncode == 0, result.stderr
    assert 'passed' in result.stdout
