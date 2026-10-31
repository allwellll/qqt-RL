"""Same initial observations/keys and native bf16 policy, isolated per backend."""
import os,json,argparse,pickle,hashlib,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/iter_pbt_v1/native_gpu')
def main():
 p=argparse.ArgumentParser();p.add_argument('backend',choices=['cpu','gpu']);a=p.parse_args()
 os.environ.update(CUDA_VISIBLE_DEVICES='0' if a.backend=='gpu' else '',JAX_PLATFORMS='cuda' if a.backend=='gpu' else 'cpu',JAX_PLATFORM_NAME='cuda' if a.backend=='gpu' else 'cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAXBOMB_RULE='bun')
 import jax,jax.numpy as j,numpy as np
 jax.config.update('jax_default_matmul_precision','highest');jax.devices()
 from scripts import expert_reward_alignment_fullmatch as F
 from jax_bomb import jax_net as net
 os.environ.update(CUDA_VISIBLE_DEVICES='0' if a.backend=='gpu' else '',JAX_PLATFORMS='cuda' if a.backend=='gpu' else 'cpu')
 F.SEED_START=20261101;F.T.configure();F.provenance.install_provenance();st,ax,keys=F.initial_batch(j.arange(8,dtype=j.int32));obs,gv,mm,bm=F.E.inputs(st)
 fingerprint=lambda x:hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
 r=json.loads(Path('runs/iter_pbt_v1/eval_requests/slot05_c000005_u000720.json').read_text());params=__import__('scripts.checkpoint_policy_artifact',fromlist=['load_actor']).load_actor(Path(r['checkpoint']))
 @jax.jit
 def forward(params,obs,gv):return F.E.train.net_forward(params,'transformer',obs,gv)
 logits=forward(params,obs,gv);jax.block_until_ready(logits)
 split=jax.vmap(lambda k:jax.random.split(k,5))(keys)
 gumbels=jax.vmap(lambda k:jax.random.gumbel(k,(5,)))(split[:,1])
 result=dict(backend=a.backend,observations_sha256=fingerprint(obs),global_state_sha256=fingerprint(gv),actor_keys=np.asarray(keys).tolist(),move_mask=np.asarray(mm).tolist(),ability_mask=np.asarray(bm).tolist(),move_logits=np.asarray(logits[0]).tolist(),ability_logits=np.asarray(logits[1]).tolist(),gumbels=np.asarray(gumbels).tolist(),parameter_dtypes=sorted({str(x.dtype) for x in jax.tree.leaves(params)}),network_compute='jax_bomb/jax_net.py _transformer_pool/_tf_block explicit bf16 casts; highest matmul does not remove bf16 arithmetic',unix=time.time())
 atomic_json(B/f'POLICY_BACKEND_{a.backend}.json',result)
if __name__=='__main__':main()
