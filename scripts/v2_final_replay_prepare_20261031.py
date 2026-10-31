"""Freeze winner's fixed first eight + explicitly selected earliest outcomes."""
from pathlib import Path
from scripts import v2_fresh_selection_20261031 as core
from scripts.population_run_state import atomic_json,sha
B=core.OUT

def main():
 done=core.read(B/'COMPLETE.json');r=core.read(B/'RESULTS.json');assert done['results_sha256']==sha(B/'RESULTS.json')
 label=r['selection']['winner'];ms=r['models'];p=core.read(B/'PROTOCOL.json');rows=[];primary={};files={}
 for j in core.jobs(p):
  if j['label']!=label or j['condition']!='hunter_hard':continue
  rows+=core.validate(j,ms[label]);f=core.path(j);files.update(core.read(str(f)+'.ACCEPTED.json')['files'])
  for e in core.read(f)['results']['natural_full2400']['episodes']:primary[e['seed']]=e['delivery']
 rows.sort(key=lambda r:r['seed']);assert len(rows)==256;games=[];missing=[]
 def add(row,rule):
  e=dict(row,primary_delivery=primary[row['seed']]);games.append(dict(index=row['seed']-p['conditions']['hunter_hard']['seed'],seed=row['seed'],seat=row['seat'],rule=rule,expected=e))
 for row in rows[:8]:add(row,'预声明固定前8局，不按结果挑选')
 for seat in (0,1):
  wins=[x for x in rows if x['seat']==seat and x['winner']==seat][:2]
  losses=[x for x in rows if x['seat']==seat and x['winner'] not in (None,-1,seat)][:1]
  for row in wins:add(row,f'披露筛选：P{seat}按seed最早2胜')
  for row in losses:add(row,f'披露筛选：P{seat}按seed最早1负')
  if len(wins)<2 or not losses:missing.append(dict(seat=seat,wins=len(wins),losses=len(losses)))
 zeros=[x for x in rows if x['seat']==1 and x['placed'][1]==0 and x['placed'][0]>0][:2]
 for row in zeros:add(row,'披露筛选：P1按seed最早2个单方零泡')
 dest=B/'final_replay';dest.mkdir(exist_ok=True)
 atomic_json(dest/'MANIFEST.json',dict(status='FROZEN_BEFORE_RECORDING',model=ms[label],games=games,missing=missing,source_files=files,
  spawn_manifest=p['conditions']['hunter_hard']['spawn_manifest'],model_label=f'bun-native-v2 {label} · 固定矩阵胜出候选',
  html=f'20261031-v2-{"ownside-" if p.get("spawn_side_version") else ""}{label.replace("_","-")}-review.html',teacher=False,rule_version='bun-native-v2',
  spawn_side_version=p.get('spawn_side_version'),cages=core.read(p['conditions']['hunter_hard']['spawn_manifest']).get('cages'),selection_sha256=sha(B/'RESULTS.json'),selection_scope='fixed first8 plus disclosed earliest wins/losses/P1 unilateral zero; not representative frequency'))
 print('winner replay manifest frozen',label,len(games),len({x['index'] for x in games}))
if __name__=='__main__':main()
