"""Independent exact inventory/protocol audit for all browser P1 screen artifacts."""
import hashlib,json,time
from pathlib import Path
from scripts.cx_web_temperature_eval_v1 import OUT
from scripts.population_run_state import atomic_json,sha
def main():
 p=json.loads((OUT/'PROTOCOL.json').read_text());ph=sha(OUT/'PROTOCOL.json');count=0;nonzero_slots=0
 for f,h in p['source_sha256'].items():assert sha(f)==h,f
 for combo in p['combinations']:
  assert sha(combo['model']['model_json'])==combo['model']['model_json_sha256']
  for bot in p['bots']:
   mode='greedy' if combo['greedy'] else 'categorical';expected={g['seed']:g for g in p['games'][bot][mode]};seen=set()
   for complete in (OUT/'pairs'/combo['label']/bot.replace('@','_')).glob('shard_*/COMPLETE.json'):
    cfg=json.loads((complete.parent/'CONFIG.json').read_text());assert cfg['source_sha256']==p['source_sha256']
    for r in json.loads(complete.read_text())['results']:
     assert r['seed'] in expected and r['seed'] not in seen;seen.add(r['seed'])
     assert r['seat']==1 and r['temperature']==combo['temperature'] and r['bot']==bot
     assert r['candidate']==combo['label'] and r['target']==bot and r['mode']==mode and r['protocol_sha256']==ph
     assert r['model_json_sha256']==combo['model']['model_json_sha256'] and r['status']=='PASS' and r['review_verified'] and r['complete']
     folder=complete.parent/f"match-{r['seed']}-P1-{mode}-greedy";rf=folder/'review.json';assert sha(rf)==r['review_sha256']
     actual=json.loads((folder/'RESULT.json').read_text());assert r==actual
     review=json.loads(rf.read_text());assert review['schema']=='cx-web-native-replay-slots-v1'
     assert review['meta']['game']==expected[r['seed']] and review['meta']['learner_seat']==1
     assert review['summary']==dict(ticks=r['ticks'],done=True,winner=r['winner'])
     assert len(review['actions'])==r['ticks']<=2400
     for a in review['actions']:
      assert len(a)==6 and all(isinstance(x,int) for x in a)
      assert all(0<=a[i]<5 for i in (0,2)) and all(0<=a[i]<3 for i in (1,3)) and all(0<=a[i]<7 for i in (4,5))
      nonzero_slots+=sum(a[i]>0 and a[2*(i-4)+1]==2 for i in (4,5))
     count+=1
   assert seen==set(expected) and len(seen)==64,(combo['label'],bot,len(seen))
 assert count==960
 browser=json.loads((OUT/'BROWSER_CANDIDATE.json').read_text());assert browser['status']=='PASS_REAL_BROWSER_APP_P1_V2_TEMPERATURE'
 assert browser['human']['backend']=='worker-wasm'
 atomic_json(OUT/'ACCEPTANCE.json',dict(status='PASS_960_EXACT_INVENTORY_PROTOCOL_REPLAY_MODEL_SAMPLING_P1',games=count,protocol_sha256=ph,
  sources_verified=len(p['source_sha256']),nonzero_item_slot_use_actions=nonzero_slots,actual_browser=browser['status'],browser_backend=browser['human']['backend'],push=False,unix=time.time()))
 print('PASS',count,'exact browser P1 artifacts, nonzero-slot uses',nonzero_slots,flush=True)
if __name__=='__main__':main()
