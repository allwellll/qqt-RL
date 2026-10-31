"""Freeze the explicitly named paired export and deterministic preview selection."""
import copy,json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1'); O=B/'replay_r1_slot00_u2400_preview'; W=B/'iter2/a3sr/lineage_watch_20261030'; C=W/'candidate_slot00_u2400_1440'
def read(p):return json.loads(Path(p).read_text())
def collect(root):
 rows={};files={}
 for f in sorted(root.glob('greedy_*.json.ACCEPTED.json')):
  for name,h in read(f)['files'].items():
   assert sha(name)==h;files[name]=h
  p=Path(str(f).removesuffix('.ACCEPTED.json'))
  primary={r['seed']:r for r in read(p)['results']['natural_full2400']['episodes']}
  for r in read(str(p)+'.outcome.json')['episodes']:
   r['primary_delivery']=primary[r['seed']]['delivery'];rows[r['seed']]=r
 assert len(rows)==256
 return rows,files

def main():
 O.mkdir(parents=True,exist_ok=True);proto=read(C/'PROTOCOL.json');old=read(B/'replay_a3sr_u1280/MANIFEST.json');candidate=proto['models']['candidate'];base=proto['models']['baseline']
 for m in (candidate,base):assert sha(m['model_json'])==m['model_sha256'] and sha(Path(m['checkpoint'])/'policy.pt')==m['policy_sha256']
 fresh,files=collect(W/'fresh/slot00_u002400/hard'); paired,pf=collect(C/'paired_fresh/slot00_u002400/hard');baseline,bf=collect(C/'paired_fresh/slot05_u001280/hard');files.update(pf);files.update(bf)
 oldbase={g['seed']:g['expected'] for g in old['games']};selected=[];missing=[]
 def add(r,b,spawn,why):selected.append(dict(index=len(selected),source_index=r['seed']-read(spawn)['seed'],seed=r['seed'],seat=r['seat'],expected=copy.deepcopy(r),baseline_expected=copy.deepcopy(b),spawn_manifest=spawn,rule=why))
 for g in old['games'][:8]:add(fresh[g['seed']],oldbase[g['seed']],old['spawn_manifest'],'01–08：u1280页首8同seed同席位')
 rows=sorted((r for r in paired.values() if r['seat']==1),key=lambda r:r['seed']);spawn=proto['tiers']['hard']['spawn_manifest']
 for result,label in ((1,'胜'),(None,'平')):
  found=[r for r in rows if r['winner'] in ((None,-1) if result is None else (1,))][:3]
  assert len(found)==3
  for r in found:add(r,baseline[r['seed']],spawn,'09–14：P1配对顺序前3'+label)
 z=[r for r in rows if baseline[r['seed']]['placed'][1]==0 and baseline[r['seed']]['placed'][0]>0 and r['placed'][1]>0][:2]
 if len(z)<2:
  missing.append('baseline单方零泡而候选非零不足2；按合同采用候选零泡回退')
  z=[r for r in rows if r['placed'][1]==0][:2]
 assert len(z)==2
 for r in z:add(r,baseline[r['seed']],spawn,'15–16：'+('baseline单方零泡，候选非零' if baseline[r['seed']]['placed'][1]==0 and r['placed'][1]>0 else '候选零泡回退'))
 found=[r for r in rows if r['winner']==1 and baseline[r['seed']]['winner'] in (None,-1)][:2];assert len(found)==2
 for r in found:add(r,baseline[r['seed']],spawn,'17–18：候选胜、u1280平的顺序前2')
 assert len(selected)==18
 cfg=dict(status='FROZEN_BEFORE_RECORDING',model=candidate,games=selected,source_files=files,missing=missing,spawn_manifest=old['spawn_manifest'],teacher=False,rule_version='stock-window-v1',html='20261030-r1-slot00-u2400-preview.html',model_label='配对版本 A3SR slot00 u2400（任务别名R1，非新R1）',identity_correction='指定lineage_watch导出属于c34旧A3SR；不是c35新R1。遵从任务明确artifact，不替换权重。',unix=time.time())
 atomic_json(O/'MANIFEST.json',cfg);atomic_json(O/'IDENTITY_AMENDMENT.json',dict(status='IDENTITY_CLARIFIED',task='.hermes/TASK_preview_replay_r1_slot00_u2400_20261030.md',selected=candidate,actual_r1_checkpoint=str(B/'formal/bt_slot00_c000035/u002400'),reason=cfg['identity_correction']))
 baselinecfg=copy.deepcopy(cfg);baselinecfg['model']=base;baselinecfg['model_label']='A3SR u1280 同seed对照'
 for g in baselinecfg['games']:g['expected']=g.pop('baseline_expected')
 (O/'baseline').mkdir(exist_ok=True);atomic_json(O/'baseline/MANIFEST.json',baselinecfg)
 atomic_json(O/'TASK_CONSUMED.json',dict(status='FROZEN_RECORDING_NEXT',task='.hermes/TASK_preview_replay_r1_slot00_u2400_20261030.md',task_sha256=sha('.hermes/TASK_preview_replay_r1_slot00_u2400_20261030.md'),manifest_sha256=sha(O/'MANIFEST.json'),unix=time.time()))
 print(json.dumps(dict(entries=len(selected),candidate=candidate['checkpoint'],missing=missing)))
if __name__=='__main__':main()
