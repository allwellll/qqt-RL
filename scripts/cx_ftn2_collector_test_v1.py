"""CPU final constructed collector ABI and full reward-path shape, using the healthy checkpoint."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1')
import json,pickle,time
from pathlib import Path
from scripts.model_cuda_overlay import install
install()
from scripts import cx_ftn2_reward_v1 as V
V.install()
import jax
import jax.numpy as j
from scripts import cc_league2_train as T,combat_regions_mixed_v2 as mixed
from scripts.population_run_state import atomic_json,sha
M=T.__dict__ # setup lives in the sealed base worker, no training main invoked.
scope={'__name__':'cpu_ftn2_schema'}
exec(compile(Path('scripts/combat_specialist_g_league_train.py').read_text(),'cpu_ftn2_schema','exec'),scope)
backend=scope['setup']()
cp=Path('runs/cc_train_20261105/training/formal/L2_S05g/u019302')
with (cp/'state.pkl').open('rb') as f:p=pickle.load(f)
take=lambda x:j.asarray(x[:2])
params=jax.tree.map(j.asarray,p['params']);st,ax,keys=jax.tree.map(take,(p['states'],p['aux'],p['keys']))
ax=mixed.migrate_aux(V.fork_aux(ax));bank=jax.tree.map(j.asarray,p['league_bank'])
proto=json.loads(Path('runs/cc_train_20261105/training/protocols/CC_LEAGUE2_L2_FTN_V2.json').read_text())
mixed.WEIGHTS=tuple(proto['arms']['L2_FTN']['opponent_weights'])
bs,ba=backend.banks(None,None,json.loads(Path(proto['r1_bank_manifest']).read_text()))
spawns=j.asarray(json.loads(Path(proto['spawn_manifest']).read_text())['spawn_cells'])
b=mixed.bun_backend();source=b._league_source
assert 'live&st.alive' in source and 'diagnostics.update' in source
shape=jax.eval_shape(mixed.collector(2,2,2026709302,spawns),params,st,ax,keys,bs,ba,bank,j.int32(0))
end,batch,exposure=shape
assert len(batch)==10 and batch[8].shape==batch[9].shape
required=['r3_f2_variant','r3_f2_frontier_event','r3_f2_selected_safe','r3_f2_post_safe','r3_f2_continuation_candidate','r3_candidate']
assert all(k in exposure['bun'] for k in required),list(exposure['bun'])
out=Path('runs/cc_train_20261105/training/ftn2_20261106/tests')
(out/'FINAL_COLLECTOR_SOURCE.py.txt').write_text(source)
atomic_json(out/'COLLECTOR_ABI.json',dict(status='PASS',backend=jax.default_backend(),unix=time.time(),batch_schema=['observations','global','actions','logprob','advantage','returns','move_mask','ability_mask','mode','live'],batch_shapes=[x.shape for x in batch],required_rollout_fields=required,source_sha256=sha(out/'FINAL_COLLECTOR_SOURCE.py.txt'),reward_sources={k:sha(k) for k in ['scripts/cx_ftn2_reward_v1.py','scripts/cx_safe_frontier_v1.py','scripts/cx_fountain_continuation_v2.py']}))
print('PASS final constructed collector ABI and full reward eval_shape',flush=True)
