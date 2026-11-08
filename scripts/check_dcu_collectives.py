"""Small physical-DCU collective check; no model training or files are changed."""
import argparse
import json
import os


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--devices',default='0,1,2,3,4,5,6,7')
    args=parser.parse_args()
    try:ids=[int(x) for x in args.devices.split(',')]
    except ValueError:parser.error('Expected comma-separated card indices')
    if not ids or min(ids)<0 or len(set(ids))!=len(ids):parser.error('Invalid card indices')
    from qqt_rl.training.dcu import configure
    configure(ids[0])
    os.environ.update(HIP_VISIBLE_DEVICES=args.devices,CUDA_VISIBLE_DEVICES=args.devices,
        JAX_ENABLE_COMPILATION_CACHE='false')
    import jax
    import jax.numpy as j
    import numpy as np
    devices=jax.local_devices()
    if len(devices)!=len(ids) or any(d.platform!='gpu' for d in devices):
        raise RuntimeError(f'Expected {len(ids)} ROCm cards, got {devices}')
    def check(x):
        total=jax.lax.psum(x,'cards')
        gathered=jax.lax.all_gather(x,'cards')
        return total,gathered
    values=np.arange(len(ids)*1024,dtype=np.float32).reshape(len(ids),1024)/1024
    total,gathered=jax.pmap(check,axis_name='cards',devices=devices)(values)
    np.testing.assert_array_equal(np.asarray(total),np.broadcast_to(values.sum(0),values.shape))
    np.testing.assert_array_equal(np.asarray(gathered),np.broadcast_to(values,(len(ids),)+values.shape))
    print(json.dumps(dict(status='PASS_PHYSICAL_DCU_COLLECTIVES',cards=len(ids),
        device_kinds=[d.device_kind for d in devices],psum=True,all_gather=True)))


if __name__=='__main__':main()
