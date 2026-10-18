"""Read-only CPU extraction from a COMMITTED checkpoint and frozen authority."""
import argparse,os,sys,json,pickle,hashlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ap=argparse.ArgumentParser();ap.add_argument('--authority',required=True,type=Path);ap.add_argument('--run',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--checkpoint-sha256',required=True);args=ap.parse_args()
a=args.authority.resolve();sys.path.insert(0,str(a));sys.path.insert(0,str(Path('scripts').resolve()))
from extract_h8_critic import load_legacy_modules
load_legacy_modules(a,Path('web/assets/maps/levels.json').resolve())
import numpy as np
b=args.run;c=b/'formal/bun392_bridge_pfsp8/gen000008';m=json.loads((c/'COMMITTED.json').read_text());sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert m['passed'] and m['policy_sha256']==args.checkpoint_sha256
for file,key in [('policy.pt','policy_sha256'),('state.pkl','state_sha256')]:assert sha(c/file)==m[key]
assert sha(b/'plan.json')==m['plan_sha256']
print('receipts verified',flush=True)
s=pickle.load((c/'state.pkl').open('rb'))
import jax
p=s['live']['params'];single=jax.tree.map(lambda x:np.asarray(x)[0],p['critic'])
assert all(np.shape(x)[0]==8 and all(np.array_equal(np.asarray(x)[0],np.asarray(x)[i]) for i in range(8)) for x in jax.tree.leaves(p['critic']))
actor=pickle.load((c/'policy.pt').open('rb'));actor=actor.get('params',actor)
assert all(np.array_equal(x,np.asarray(y)[0]) for x,y in zip(jax.tree.leaves(actor),jax.tree.leaves(p['actor'])))
o=args.output;o.mkdir(parents=True,exist_ok=True);(o/'critic.pkl').write_bytes(pickle.dumps(single,protocol=4))
from jax_bomb import jax_net
import jax.numpy as jnp
obs=jnp.linspace(-1,1,3*24*13*15).reshape(3,24,13,15);gv=jnp.linspace(-1,1,3*24).reshape(3,24)
f=jax.jit(lambda p:jax_net.net_forward(p,'transformer',obs,gv)[2]);x=np.asarray(f(single));y=np.asarray(f(pickle.loads((o/'critic.pkl').read_bytes())));np.testing.assert_array_equal(x,y)
meta=dict(schema='qqt.je10-critic/v1',generation=8,checkpoint_sha256=m['policy_sha256'],state_sha256=m['state_sha256'],critic_sha256=sha(o/'critic.pkl'),gamma=json.loads((b/'plan.json').read_text())['gamma'],network_sha256=sha(a/'jax_bomb/jax_net.py'),replicas_exact=True,actor_exact=True,parity=dict(original=x.tolist(),extracted=y.tolist(),max_error=float(abs(x-y).max())),device='cpu',input_encoding='bun24-trap-timer-v2',role='trained JE10 total reward Critic; raw-reward A1 is an explicitly approximate same-Critic diagnostic, not a separately raw-trained Critic')
(o/'meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
