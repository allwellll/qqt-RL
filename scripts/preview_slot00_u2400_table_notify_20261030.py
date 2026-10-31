"""Deliver the requested 18-game table image, restricted to this preview asset."""
import hashlib,json,os,shutil,subprocess,time,urllib.request
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
B=Path('runs/bomb_threat_v1/replay_r1_slot00_u2400_preview')
def main():
 guard=lock(B/'TABLE_NOTIFY.lock')
 while not (B/'RECEIPT.json').exists():
  if (B/'FINISH_FAILURE.json').exists():return
  time.sleep(5)
 if (B/'TABLE_NOTIFICATION.json').exists() and json.loads((B/'TABLE_NOTIFICATION.json').read_text())['status']=='SENT':return
 p=B/'20261030-r1-slot00-u2400-preview-summary.png';p.write_bytes((B/'SUMMARY_TABLE.png').read_bytes());raw=p.read_bytes();helper=Path(shutil.which('jdme-notify')).parent.parent/'oss.sh'
 r=subprocess.run([str(helper),str(p)],env=dict(os.environ,PYTHON_BIN='/usr/bin/python3'),capture_output=True,text=True,timeout=180,check=True);urls={}
 for line in r.stdout.splitlines():
  if 'Location:' in line:
   label,url=line.split('Location:',1);urls['internal' if 'INNER' in label else 'external']=url.strip()
 assert set(urls)=={'external','internal'};urls['external']=urls['external'].replace('jdcloud-oss.com','jcloudcs.com').replace('http://','https://');opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));checks={}
 for label,url in urls.items():
  with opener.open(url,timeout=90) as response:b=response.read();status=response.status
  assert status==200 and b==raw;checks[label]=dict(url=url,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),status=status,byte_equal=True)
 atomic_json(B/'TABLE_UPLOAD.json',dict(status='BOTH_URLS_BYTE_VERIFIED',reads=checks,local_sha256=sha(p)))
 r=subprocess.run(['jdme-notify','send','18局同seed对照摘要表（u1280→指定配对slot00 u2400；实际旧A3SR c34，不是新R1）：'+urls['external']+'；放泡、击杀、交付、距离/分母与缺测项详见表。'],capture_output=True,timeout=60)
 atomic_json(B/'TABLE_NOTIFICATION.json',dict(status='SENT' if r.returncode==0 else 'FAILED',exit_code=r.returncode,table_url=urls['external'],unix=time.time()))
 assert r.returncode==0
 rec=json.loads((B/'RECEIPT.json').read_text());rec.update(summary_table_url=urls['external'],summary_table_sha256=sha(p),unique_games=len({(g['seed'],g['seat'],g['spawn_manifest']) for g in json.loads((B/'MANIFEST.json').read_text())['games']}));atomic_json(B/'RECEIPT.json',rec)
if __name__=='__main__':
 try:main()
 except Exception as e:atomic_json(B/'TABLE_FAILURE.json',dict(status='BLOCKED',error=str(e),unix=time.time()));raise
