#!/usr/bin/env python3
"""Read-only H8 g32 critic extractor.

The training checkpoint is intentionally outside this repository.  This tool
loads it on CPU, keeps replica zero, verifies all replicas are byte-identical,
and writes a compact private npz plus public-safe metadata.
"""
import argparse
import builtins
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import pickle
import sys

import numpy as np

sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_legacy_modules(source, levels):
    import jax_bomb  # noqa: F401: ensure package exists for legacy imports
    cache = pathlib.Path(source) / "jax_bomb" / "__pycache__"
    real_open = builtins.open

    def mapped_open(name, *args, **kwargs):
        name = str(name)
        if name.endswith("/jax_bomb/web/assets/maps/levels.json"):
            name = str(levels)
        return real_open(name, *args, **kwargs)

    builtins.open = mapped_open
    try:
        for name in ("je05_core_env", "je05_bun_env", "je07_bun_env"):
            files = sorted(cache.glob(name + ".cpython-312.pyc"))
            if not files:
                continue
            full = "jax_bomb." + name
            loader = importlib.machinery.SourcelessFileLoader(full, str(files[-1]))
            spec = importlib.util.spec_from_loader(full, loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            loader.exec_module(module)
    finally:
        builtins.open = real_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metadata", required=True)
    args = ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                      JAXBOMB_RULE='bun', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    source = pathlib.Path(args.source).resolve()
    sys.path.insert(0,str(source))
    cp = source / "runs/je09_bun_20261017/stage_h8_native_courier_fsm_dagger_v34/formal/bun392_bridge_pfsp8/gen000032"
    committed = json.loads((cp / "COMMITTED.json").read_text())
    if not committed.get("passed") or committed.get("generation") != 32:
        raise SystemExit("immutable H8 g32 checkpoint receipt is not valid")
    if sha((cp / "policy.pt").read_bytes()) != committed["policy_sha256"]:
        raise SystemExit("policy receipt hash mismatch")
    assert sha((cp/'ema_policy.pt').read_bytes())==committed['ema_policy_sha256'],'EMA receipt hash'
    state_hash=hashlib.sha256()
    with (cp/'state.pkl').open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):state_hash.update(chunk)
    assert state_hash.hexdigest()==committed['state_sha256'],'state receipt hash'
    plan_file=cp.parents[2]/'plan.json'
    plan=json.loads(plan_file.read_text())
    assert sha(plan_file.read_bytes())==committed['plan_sha256']
    load_legacy_modules(source, pathlib.Path(args.levels).resolve())
    with (cp / "state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    params = state["live"]["params"]["critic"]

    def replica(x, i=0):
        if isinstance(x, dict):
            return {k: replica(v, i) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(replica(v, i) for v in x)
        a = np.asarray(x)
        return np.asarray(a[i])

    def all_equal(x):
        if isinstance(x, dict):
            return all(all_equal(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return all(all_equal(v) for v in x)
        a = np.asarray(x)
        return a.shape[0] == 8 and all(np.array_equal(a[0], a[i]) for i in range(1, 8))

    if not all_equal(params):
        raise SystemExit("critic replicas are not identical")
    with (cp/'policy.pt').open('rb') as stream:actor=pickle.load(stream)
    actor=actor.get('params',actor)
    with (cp/'ema_policy.pt').open('rb') as stream:ema_actor=pickle.load(stream)
    ema_actor=ema_actor.get('params',ema_actor)
    def same(a,b):
        if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return np.array_equal(a,b)
    assert same(actor,replica(state['live']['params']['actor'])),'checkpoint actor generation mismatch'
    def shapes(x):
        if isinstance(x,dict):return {k:shapes(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [shapes(v) for v in x]
        return list(np.shape(x))
    assert shapes(actor)==shapes(ema_actor),'raw/EMA architecture mismatch'
    flat = {}

    def flatten(x, prefix="p"):
        if isinstance(x, dict):
            for k, v in x.items():
                flatten(v, prefix + "." + k)
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                flatten(v, prefix + "." + str(i))
        else:
            flat[prefix] = np.asarray(x[0] if np.asarray(x).shape[0] == 8 else x, dtype=np.float32)

    flatten(params)
    blob = b"".join(k.encode() + b"\0" + flat[k].tobytes(order="C") for k in sorted(flat))
    out = pathlib.Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **flat)
    single=replica(params)
    serialized=pickle.dumps(single,protocol=4)
    out.with_suffix('.pkl').write_bytes(serialized)
    import jax
    import jax.numpy as jnp
    from jax_bomb import jax_net
    assert jax.default_backend()=='cpu'
    # Fixed native observations are emitted by the production simulator.
    import subprocess
    fixture=json.loads(subprocess.check_output(['node','-e',
        "const Q=require('./web/sim'),P=require('./web/review_protocol'),fs=require('fs');"
        "const l=JSON.parse(fs.readFileSync('web/assets/maps/levels.json'))[0],s=P.setup({seed:2026101900,max_ticks:2400},l);"
        "const out=[];for(let t=0;t<3;t++){out.push({obs:Array.from(s.encodeObsJAX(0,24)),state:Array.from(s.encodeStateJAX(0,'bun24-trap-timer-v2'))});s.step([[4,0,0],[4,0,0]]);}console.log(JSON.stringify(out));"],text=True))
    obs=jnp.asarray([f['obs'] for f in fixture],jnp.float32).reshape(-1,24,13,15)
    gv=jnp.asarray([f['state'] for f in fixture],jnp.float32)
    forward=jax.jit(lambda p:jax_net.net_forward(p,'transformer',obs,gv)[2])
    original=np.asarray(forward(single));reloaded=np.asarray(forward(pickle.loads(serialized)))
    np.testing.assert_array_equal(original,reloaded)
    meta = {
        "schema": "qqt.h8-critic/v1", "status": "extracted", "generation": 32,
        "architecture": {"name": "transformer", "obs": [24, 13, 15], "global_dim": 24,
                          "embed": 392, "depth": 4, "heads": 4, "patch": 3,
                          "value_bins": 128},
        "gamma": plan['gamma'], "value_range": [jax_net.V_MIN,jax_net.V_MAX], "replicas": 8,
        "replicas_exact": True, "critic_sha256": sha(blob),
        "checkpoint_sha256": committed["policy_sha256"],
        "ema_policy_sha256":committed['ema_policy_sha256'],
        "selected_actor":"raw policy.pt; state.live.params.actor exact; EMA has matching architecture but is not used",
        "state_sha256": committed["state_sha256"], "parameters": len(flat),
        "device": "cpu", "input_encoding": "bun24-trap-timer-v2",
        "output": "expected HL-Gauss value over 128 bins",
        "actor_state_policy_exact":True,
        "observation_normalization":"production encodeObsJAX(24) and encodeStateJAX(bun24-trap-timer-v2); fixed channel/global scaling; no fitted running normalization",
        "network_sha256":sha((source/'jax_bomb/jax_net.py').read_bytes()),
        "plan_sha256":committed['plan_sha256'],
        "parity":{"fixture_count":len(fixture),"checkpoint_values":original.tolist(),"extracted_values":reloaded.tolist(),"max_error":float(np.max(np.abs(original-reloaded)))},
    }
    pathlib.Path(args.metadata).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
