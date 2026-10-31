"""Read immutable legacy source before selecting native-item state classes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1')
import json,pickle,time
from pathlib import Path
import jax,numpy as np
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2')


def main():
    source=json.loads((OUT.parent/'SOURCE_AUTHORIZATION.json').read_text())['source'];cp=Path(source['checkpoint'])
    assert sha(cp/'state.pkl')==source['state_sha256'] and sha(cp/'policy.pt')==source['policy_sha256']
    p=pickle.load((cp/'state.pkl').open('rb'))
    old=json.loads(Path('runs/bomb_threat_v1/formal/bt_slot04_c000035/SOURCE_HANDOFF.json').read_text())
    assert old['status']=='PASS' and old['source']['state_sha256']==source['state_sha256']
    names=('params','adam','combat_ring','combat_ring_cursor','combat_fixed_actor','warmup_rows','update','environment_steps','optimizer_steps','iter2_arm')
    data={k:p[k] for k in names};data=jax.tree.map(lambda x:np.asarray(x) if hasattr(x,'dtype') else x,data)
    tmp=OUT/'parent_model_adam.pkl.tmp'
    with tmp.open('wb') as f:pickle.dump(data,f,protocol=5);f.flush();os.fsync(f.fileno())
    tmp.replace(OUT/'parent_model_adam.pkl')
    atomic_json(OUT/'PARENT_EXPORT.json',dict(status='PASS',source=source,output=str(OUT/'parent_model_adam.pkl'),sha256=sha(OUT/'parent_model_adam.pkl'),
        preserved=list(names),params_digest=digest(p['params']),Adam_digest=digest(p['adam']),ring_digest=digest(p['combat_ring']),
        source_old_next_step_parity=old,old_env_aux_excluded=True,reason='anonymous graveyard cannot be truthfully mapped to native typed queues; all episodes rebuild at natural starts',unix=time.time()))

if __name__=='__main__':main()
