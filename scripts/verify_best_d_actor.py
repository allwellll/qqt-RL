"""Read-only checkpoint/export and Python/JAX/JS/WASM numerical agreement."""
import argparse
import hashlib
import inspect
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import numpy as np
import jax
import jax.numpy as jnp
from export_bun_web_model import extract_transformer, pack_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path(os.environ.get('BEST_D_MODEL', '.hermes/best-d-private/model.json')),
                        help='optional private <EXTERNAL_MODEL>; never published')
    parser.add_argument('--output', type=Path, help='optional verification receipt output')
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root))
    from jax_bomb import jax_net
    evidence = root / 'web/review-evidence/best-d-20261028'
    matrix = json.loads((evidence/'matrix.json').read_text())
    source_receipt = json.loads((evidence/'source-receipt.json').read_text())
    assert source_receipt['policy_sha256'] == matrix['checkpoint_sha256']
    receipt = json.loads((args.checkpoint/'COMMITTED.json').read_text())
    sha = lambda b: hashlib.sha256(b).hexdigest()
    actor_data = (args.checkpoint/'policy.pt').read_bytes()
    assert sha(actor_data) == receipt['policy_sha256'] == matrix['checkpoint_sha256']
    state_data = (args.checkpoint/'state.pkl').read_bytes()
    assert sha(state_data) == receipt['state_sha256']
    state = pickle.loads(state_data)
    actor = pickle.loads(actor_data)
    actor = actor.get('params', actor)
    params = state['live']['params'] if 'live' in state else state['params']
    state_actor = params['actor']
    # ITER-PBT checkpoints can have one leading replica dimension.
    assert jax.tree.structure(actor) == jax.tree.structure(state_actor)
    replicated = None
    for a, b in zip(jax.tree.leaves(actor), jax.tree.leaves(state_actor)):
        b = np.asarray(b)
        if b.shape == np.shape(a):
            np.testing.assert_array_equal(a, b)
        else:
            assert b.shape[1:] == np.shape(a)
            replicated = b.shape[0]
            for replica in b:
                np.testing.assert_array_equal(a, replica)
    critic = params['critic']
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(critic))
    critic_digest = hashlib.sha256()
    for leaf in jax.tree.leaves(critic):
        value = np.asarray(leaf)
        critic_digest.update(str(value.shape).encode()+b'\0'+value.tobytes())
    model_sha = source_receipt['model_sha256']
    assert len(model_sha) == 64 and all(c in '0123456789abcdef' for c in model_sha)
    model = None
    if args.model.is_file():
        model_data = args.model.read_bytes()
        assert sha(model_data) == model_sha, 'local <EXTERNAL_MODEL> hash differs from source receipt'
        assert len(model_data) == source_receipt['model_bytes']
        model = json.loads(model_data)
    else:
        print('SKIP: local <EXTERNAL_MODEL> absent; model-file hash and exact checkpoint/export tensor comparison not performed.', flush=True)
    native_export = args.source/'runs/iter_pbt_v1/eval/slot05_c000005_u000720_holdout/model'
    native_meta = json.loads((native_export/'model.json').read_text())
    graph_sha = sha((native_export/'model.onnx').read_bytes())
    assert graph_sha == native_meta['meta']['onnx_sha256']
    assert native_meta['meta']['checkpoint_sha256'] == matrix['checkpoint_sha256']
    tensors = None
    if model is not None:
        assert native_meta['flat'] == model['flat'] and native_meta['tensors'] == model['tensors']
        flat, tensors = pack_tensors(extract_transformer(actor))
        assert model['flat'] == flat and model['tensors'] == {k:list(v) for k,v in tensors.items()}
        print('Exact checkpoint/export tensors verified against hashed local <EXTERNAL_MODEL>.', flush=True)
    network = Path(jax_net.__file__)
    assert network.read_bytes() == (args.source/'jax_bomb/jax_net.py').read_bytes()
    assert jax.default_backend() == 'cpu'
    samples = json.loads(args.samples.read_text())
    obs = jnp.asarray([s['obs'] for s in samples], jnp.float32).reshape(-1,24,13,15)
    gv = jnp.asarray([s['state'] for s in samples], jnp.float32)
    # The existing native browser export is fp32; training JAX explicitly casts
    # its backbone to bf16. Check both and report that precision gap separately.
    namespace = dict(jax_net.__dict__)
    for fn in (jax_net._tf_block, jax_net._transformer_pool):
        definition = inspect.getsource(fn).replace('bf = jnp.bfloat16','bf = jnp.float32')
        exec(compile(definition,'<native-export-fp32-reference>','exec'),namespace)
    exec(compile(inspect.getsource(jax_net.transformer_forward),'<native-export-fp32-reference>','exec'),namespace)
    move, bomb, value, _ = jax.jit(lambda o,g:namespace['transformer_forward'](actor,o,g))(obs,gv)
    train_move, train_bomb, train_value, _ = jax.jit(lambda o,g:jax_net.net_forward(actor,'transformer',o,g))(obs,gv)
    train_move, train_bomb, train_value = map(np.asarray,(train_move,train_bomb,train_value))
    move, bomb, value = map(np.asarray, (move,bomb,value))
    rows, peak, precision_peak, precision_actions = [], 0., 0., 0
    for i, s in enumerate(samples):
        errors = {}
        for backend in ('js','wasm'):
            errors[backend] = max(float(np.max(np.abs(move[i]-s[backend]['move']))),
                                  float(np.max(np.abs(bomb[i]-s[backend]['bomb']))),
                                  float(np.max(np.abs(value[i]-s[backend]['value']))))
            assert errors[backend] < .002, (s['id'],s['tick'],backend,errors[backend],move[i].tolist(),s[backend]['move'],float(np.asarray(value[i]).squeeze()),s[backend]['value'])
        action = [int(np.argmax(np.where(s['move_mask'],move[i],-np.inf))),
                  int(np.argmax(np.where(s['ability_mask'],bomb[i],-np.inf)))]
        assert action == s['action']
        training_action=[int(np.argmax(np.where(s['move_mask'],train_move[i],-np.inf))),int(np.argmax(np.where(s['ability_mask'],train_bomb[i],-np.inf)))]
        precision_error=max(float(np.max(np.abs(train_move[i]-s['js']['move']))),float(np.max(np.abs(train_bomb[i]-s['js']['bomb']))),float(abs(train_value[i]-s['js']['value'])))
        precision_peak=max(precision_peak,precision_error)
        precision_actions+=int(training_action!=action)
        peak = max(peak,*errors.values())
        rows.append(dict(id=s['id'],tick=s['tick'],errors=errors,action=action,training_bf16_max_error=precision_error,training_bf16_action=training_action))
    proof = dict(schema='qqt.best-d-actor-parity/v1',passed=model is not None,numerical_parity_passed=True,device='cpu',
                 checkpoint_sha256=sha(actor_data),state_sha256=sha(state_data),
                 model_sha256=model_sha,network_sha256=sha(network.read_bytes()),
                 wasm_graph_sha256=graph_sha,
                 exporter_sha256=sha(Path('scripts/export_bun_web_model.py').read_bytes()),
                 actor_state_exact=True,export_tensors_exact=True if model is not None else None,
                 export_tensors_status='verified' if model is not None else 'SKIP',
                 local_model_hash_status='verified' if model is not None else 'SKIP',
                 tensor_count=len(tensors) if tensors is not None else None,
                 replicas=replicated,paired_critic_tensor_sha256=critic_digest.hexdigest(),
                 critic_scope='state.pkl paired Critic tensors verified finite; no Critic/A1 inference claimed',
                 input_encoding='bun24-trap-timer-v2',sample_count=len(samples),
                 tolerance=.002,max_error=peak,greedy_action_mismatches=0,rows=rows,
                 reference_precision='JAX float32 reference of the native browser export; same network definitions, explicit backbone dtype substitution only',
                 training_bf16=dict(max_error=precision_peak,greedy_action_mismatches=precision_actions,
                   scope='Unmodified training net_forward on the identical observations; bf16 is not fp32 browser inference; mismatches retained, never claimed equal.'),
                 scope='Predeclared ticks 0/1/50/199 on every retained episode where present; exact export tensors only checked when local model is present; see export_tensors_status. Finite sample parity is not proof for all states.')
    (args.output or evidence/'actor-parity.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print('CHECKPOINT / STATE / JAX / JS / WASM numerical checks GREEN',len(samples),peak,
          '; local export tensors',proof['export_tensors_status'])


if __name__ == '__main__':
    main()
