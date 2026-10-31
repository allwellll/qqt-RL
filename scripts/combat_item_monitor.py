"""Read-only finite utilization evidence; cannot launch or restart training."""
import json,subprocess,time
from scripts.combat_item_run import B,read
from scripts.population_run_state import lock

def main():
 unique=lock(B/'monitor.lock')
 while not (B/'GPU_COMPLETE.json').exists() and not (B/'USER_STOP.json').exists():
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  gpu=[dict(zip(['index','memory_MiB','utilization_percent'],map(lambda v:int(v.strip()),line.split(',')))) for line in raw.splitlines()]
  status=read(B/'STATUS.json');row=dict(unix=time.time(),status={k:status.get(k) for k in ['stage','arm','seed','update','environment_steps','steps_per_second']},gpu=gpu)
  with (B/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  time.sleep(15)
if __name__=='__main__':main()
