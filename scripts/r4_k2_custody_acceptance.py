"""New potential in the real composed native reward, with historical test reuse.

These controlled post-acquisition paths are diagnostic fixtures only; they do
not enter training, Bank or autonomous performance denominators.
"""
import inspect
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
from scripts import bun_native_v2_custody_acceptance as old

OUT=Path(os.environ.get('R4_K2_NATIVE_OUT','runs/r4_combat_20261031/k2_native_acceptance'))


def main():
    source=inspect.getsource(old.run)
    anchors={
        '    M.configure(True,True)':"    M.configure(True,True)\n    from scripts import r4_k2_carrier_reward as K2\n    K2.install()\n    R=K2",
        "'scripts/bun_native_v2_custody_fixture.js'":"'scripts/r4_k2_custody_fixture.js'",
        'loose=-.15':'loose=-.10',
        "'scripts/bun_native_v2_custody_reward.py','scripts/bun_native_v2_custody_collector.py','scripts/bun_native_v2_custody_train.py'":
        "'scripts/r4_k2_carrier_reward.py','scripts/r4_k2_custody_fixture.js','scripts/bun_native_v2_custody_reward.py','scripts/bun_native_v2_custody_collector.py'",
    }
    for a,b in anchors.items():
        assert source.count(a)==1,a;source=source.replace(a,b)
    namespace=dict(vars(old),OUT=OUT)
    exec(compile(source,__file__,'exec'),namespace)
    result=namespace['run']()
    if result['status']!='PASS':raise RuntimeError('real composed K2 custody acceptance BLOCKED; inspect raw records')


if __name__=='__main__':main()
