"""Common semantic-preserving initial placement around the frozen PPO worker."""
import os
import time
from pathlib import Path

def canonical_collector(original, receipt=None):
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from scripts.population_run_state import atomic_json
    placement=NamedSharding(Mesh(np.asarray(jax.devices()), ('_internal_pmap_axis_name',)), PartitionSpec('_internal_pmap_axis_name'))
    first=True
    def collect(params,states,aux,keys,*bank):
        nonlocal first
        if first:
            before=(states,aux,keys)
            states,aux,keys=jax.tree.map(lambda x:jax.device_put(x,placement),before)
            after=jax.device_get((states,aux,keys))
            assert jax.tree.structure(before)==jax.tree.structure(after)
            assert all(np.asarray(x).dtype==np.asarray(y).dtype and np.array_equal(x,y) for x,y in zip(jax.tree.leaves(before),jax.tree.leaves(after)))
            if receipt is not None:
                atomic_json(receipt,dict(passed=True,initial_full_state_exact=True,only_storage_placement_changed=True,unix=time.time()))
            first=False
        return original(params,states,aux,keys,*bank)
    return collect

def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import sys,json
    from scripts import ppo_scale_longrun_train as P
    from scripts import campaign_paired_train as frozen
    original=P.C.collector
    protocol=json.loads(Path(sys.argv[1]).read_text())
    stage,arm,seed=sys.argv[2:5]
    folder='formal' if stage=='formal' else 'smoke'
    receipt=Path(protocol['namespace'])/folder/f'{arm}_s{seed}'/'INITIAL_PLACEMENT.json'
    P.C.collector=lambda steps:canonical_collector(original(steps),receipt)
    frozen.main()

if __name__=='__main__':main()
