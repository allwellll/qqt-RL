"""Persistent CPU successor, no training scheduling or worker signals."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.v2_web_release_20261031 import read,notify
B=Path('runs/v2_fresh_selection_20261031')
def active(module):
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if module.encode() in (proc/'cmdline').read_bytes().split(b'\0') and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
  except OSError:pass
 return False

def main():
 guard=lock(B/'SUCCESSOR.lock')
 while not (B/'COMPLETE.json').exists():
  if (B/'USER_STOP').exists():atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='USER_PAUSED',unix=time.time()));return
  if (B/'SCHEDULER2_BLOCKED.json').exists():atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='BLOCKED_EVALUATION',receipt='SCHEDULER2_BLOCKED.json',unix=time.time()));return
  atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='WAITING_COMPLETE_6144_MATRIX',publication_protocol_sha256=sha(B/'PUBLICATION_PROTOCOL.json'),training_untouched=True,unix=time.time()))
  time.sleep(30)
 done=read(B/'COMPLETE.json');assert done['games']==6144 and sha(B/'RESULTS.json')==done['results_sha256']
 # Resume the previously authorized, lower-priority CPU diagnostic only after v2.
 old=Path('runs/bomb_threat_v1/fresh256_slot05_tau_20261031');module='scripts.r1_fresh_tau_completion'
 if not (old/'COMPLETE.json').exists() and not active(module):
  with (old/'SUCCESSOR_RESUME.log').open('ab') as stream:subprocess.Popen([sys.executable,'-m',module],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(B/'R1_FRESH_SUCCESSOR_LAUNCHED.json',dict(status='LAUNCHED_CHECK_FORMAL_ACCEPTED_PROGRESS',module=module,existing_shards_preserved=True,unix=time.time()))
 atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='RUNNING_CONDITIONAL_PUBLICATION',selection=done['selection'],unix=time.time()))
 folder=B/'publication';folder.mkdir(exist_ok=True)
 with (folder/'OWNER.log').open('ab') as stream:
  result=subprocess.run([sys.executable,'-m','scripts.v2_web_release_20261031'],stdout=stream,stderr=subprocess.STDOUT)
 state=read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else {}
 atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='PUBLICATION_RETURNED' if result.returncode==0 else 'PUBLICATION_BLOCKED',publication_status=state.get('status'),returncode=result.returncode,unix=time.time()))
 if result.returncode:notify('v2后继链出现技术阻塞，原始证据保留publication/OWNER.log；训练不受影响。')
if __name__=='__main__':main()
