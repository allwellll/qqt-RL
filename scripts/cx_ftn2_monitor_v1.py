"""FTN2-aware read-only continuation of the existing CX status observer."""
import time,json
from pathlib import Path
from scripts import cx_takeover_status_v1 as S
source=Path('scripts/cx_takeover_status_v1.py').read_text()
old="if not config['module'].startswith('scripts.cc_league2_train'):"
assert source.count(old)==1
source=source.replace(old,"if not (config['module'].startswith('scripts.cc_league2_train') or config['module']=='scripts.cx_ftn2_train_v1'):")
scope={'__name__':'cx_ftn2_private_status'};exec(compile(source,__file__,'exec'),scope)
def main():
 while True:
  f=Path('runs/cc_train_20261105/training/ftn2_20261106/SWITCH_STATUS.json')
  row=json.loads(f.read_text()) if f.exists() else {}
  new=Path('runs/cc_train_20261105/training/formal/L2_FTN2/updates.jsonl')
  scope['ARMS'][4]='L2_FTN2' if new.exists() and row.get('stage') not in ('FAILED','FALLBACK_RELAUNCHED') else 'L2_FTN'
  try:
   v=scope['snapshot']();print(json.dumps({a:dict(update=r['update'],runtime=r['runtime']) for a,r in v['arms'].items()}),flush=True)
  except Exception as e:print(type(e).__name__,str(e),flush=True)
  time.sleep(60)
if __name__=='__main__':main()
