"""Recompute CPU/GPU full-game acceptance from immutable trace artifacts."""
import json
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha
B=Path('runs/iter_pbt_v1')

def compare(cpu_path,gpu_path):
 a=json.loads(cpu_path.read_text());b=json.loads(gpu_path.read_text());assert a['policy_sha256']==b['policy_sha256']
 pairs=[];differences=[]
 for x,y in zip(a['episodes'],b['episodes'],strict=True):
  assert x['episode_index']==y['episode_index'] and x['seed']==y['seed']
  def counts(e):
   trace=np.load(e['provenance']['trace']);col=e['provenance']['telemetry_column'];names=list(trace['telemetry_names']);v=trace['telemetry'][:,col];active=trace['active'][:,col].astype(bool)
   return dict(capture=e['events']['capture'],death=e['events']['death'],bomb_placed=int(v[active,names.index('bomb_placed')].sum()),terminal=e['terminal'],termination=e['termination'])
  ca,cb=counts(x),counts(y);pairs.append(dict(episode_index=x['episode_index'],CPU=ca,GPU=cb,equal=ca==cb))
  if ca!=cb:differences.append(pairs[-1])
 result=dict(status='PASS' if not differences else 'BLOCKED',games=len(pairs),checkpoint_policy_sha256=a['policy_sha256'],CPU_reference=str(cpu_path),GPU_result=str(gpu_path),input_sha256={str(cpu_path):sha(cpu_path),str(gpu_path):sha(gpu_path)},paired_game_counts=pairs,differences=differences,scope='one checkpoint, first8 frozen dev episodes, categorical; engine/opponent/init/action RNG unchanged; not proof across all models',unexplained_differences=len(differences))
 return result

def main():
 cpu=B/'eval/slot05_c000005_u000720/native/categorical_0000.json';gpu=B/'native_gpu/parity_gpu_highest/categorical_gpu_batch_0000.json'
 result=compare(cpu,gpu);atomic_json(B/'native_gpu/PARITY_GATE.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
