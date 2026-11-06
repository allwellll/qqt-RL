"""Private FTN2 worker; sealed upstream trainer and CUDA/reward import guards retained."""
import json,os,sys
from pathlib import Path

def transform_worker(source):
 pairs={
  "    args = parser.parse_args(); protocol_path": "    parser.add_argument('--until',type=int)\n    args = parser.parse_args(); protocol_path",
  "    if not args.smoke: assert (out/'SMOKE.json').exists()": "    if not args.smoke and args.until is not None: target=min(target,args.until)\n    if not args.smoke: assert (out/'SMOKE.json').exists()",
  "    if args.smoke:\n        stage('FULL_RESTORE_TOKEN',target)":
   "    if not args.smoke:\n        stage('FULL_RESTORE_TOKEN',target)\n        result=operation(params,adam,states,aux,keys,update_key,bank,generation);jax.block_until_ready(result)\n        assert finite(result)\n        atomic_json(cp/'NEXT_UPDATE.json',dict(status='PASS',digest=digest(result),diagnostic_steps_not_counted=steps))\n    if args.smoke:\n        stage('FULL_RESTORE_TOKEN',target)",
  "atomic_json(out/'COMPLETE.json', completion_receipt(receipt, protocol))":
   "atomic_json(out/('COMPLETE.json' if target==total else 'SEGMENT.json'), dict(completion_receipt(receipt,protocol),completion_status='COMPLETE' if target==total else 'SEGMENT_COMPLETE',updates_in_this_phase=len(rows)))",
  "        snapshots.publish('G', str(cp))\n        return":
   "        atomic_json(out/f'RESTORE_u{start:06d}.json',json.loads((out/'RESTORE.json').read_text()))\n        snapshots.publish('G', str(cp))\n        return",
 }
 for old,new in pairs.items():
  assert source.count(old)==1,old
  source=source.replace(old,new)
 return source

def main():
 from scripts import cx_ftn2_reward_v1 as V
 protocol=json.loads(Path(sys.argv[1]).read_text());arm=sys.argv[2]
 assert arm=='L2_FTN2' and protocol['arms'][arm]['reward_variant']==V.VERSION
 os.environ['CC_REWARD_RECEIPT']=str(Path(protocol['namespace'])/arm/f'REWARD_RECEIPT_{os.getpid()}.json')
 from scripts.cc_reward_import_hook import hook
 hook(V)
 # Compile a private copy; never edit the source sealed by other live arms.
 source=Path('scripts/cc_league2_train.py').read_text()
 anchor="    scope = {'__name__': 'cc_league2_worker', '__file__': __file__}"
 assert source.count(anchor)==1
 source=source.replace(anchor,'    source=transform_worker(source)\n'+anchor)
 scope={'__name__':'cx_ftn2_private_trainer','__file__':__file__,'transform_worker':transform_worker}
 exec(compile(source,__file__,'exec'),scope)
 scope['FORK']=scope['FORK'].replace('    from scripts import cc_league2_lib as league2',
  '    from scripts import cx_ftn2_reward_v1 as variant\n    if not cps: aux=variant.fork_aux(aux,out)\n    from scripts import cc_league2_lib as league2')
 assert 'variant.fork_aux' in scope['FORK']
 scope['main']()

if __name__=='__main__':main()
