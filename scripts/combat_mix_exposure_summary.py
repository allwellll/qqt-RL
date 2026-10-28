"""Passive training exposure totals with distinct nominal/executed denominators."""
import json,time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha
BASE=Path('runs/combat_mix_v1')
def main():
 results={};excluded={}
 for path in sorted(BASE.glob('formal*/*/updates.jsonl')):
  rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
  exposed=[r for r in rows if r.get('exposure')]
  if not exposed:continue
  keys=set().union(*(r['exposure'] for r in exposed));total={}
  for k in keys:
   if k in ('course','mean_reward','mean_terminal'):continue
   arrays=[np.asarray(r['exposure'][k],float) for r in exposed if k in r['exposure']]
   if arrays:total[k]=np.sum(arrays,axis=0).tolist()
  d=dict(update_count=len(rows),first_update=rows[0]['new_update'],last_update=rows[-1]['new_update'],
   actual_environment_steps=sum(r['actual_environment_steps_this_update'] for r in rows),
   actual_optimizer_steps=sum(r['actual_optimizer_steps_this_update'] for r in rows),raw_totals=total,updates_sha256=sha(path))
  if 'pool_live_ticks' in total:
   x=np.asarray(total['pool_live_ticks']);d['opponent_live_tick_fraction']=dict(zip(['rule3','hunter','hard','current','ring','fixed'],(x/x.sum()).tolist()))
  if 'scene_opponent_counts' in total:
   x=np.asarray(total['scene_opponent_counts'])[:,:,0]
   d['nominal_dispatch_live_tick_fraction']=dict(zip(['rule','g0','stage1','league'],(x.sum(0)/max(1,x.sum())).tolist()))
   d['scene_live_tick_fraction']=(x.sum(1)/max(1,x.sum())).tolist()
   d['nominal_only_warning']='Weak-role and assist overrides are not removed from nominal scene/kind counts. Do not call nominal league actual opponent exposure.'
  if 'executed_source_live_ticks' in total:
   x=np.asarray(total['executed_source_live_ticks']);d['executed_source_live_tick_fraction']=dict(zip(['rule','g0','stage1','league','weak_guard','weak_courier','assist'],(x/max(1,x.sum())).tolist()))
   x=np.asarray(total['executed_league_member_live_ticks']);actual=np.array([x[0],x[7:].sum(),x[1:7].sum()]);den=max(1,np.sum(total['executed_source_live_ticks']))
   d['actual_current_ring_fixed_per_live_tick']=dict(zip(['current','ring','fixed'],(actual/den).tolist()))
   d['executed_telemetry_updates']=sum('executed_source_live_ticks' in r['exposure'] for r in exposed)
   d['executed_denominator_note']='Only telemetry-version updates; missing earlier updates are not inferred.'
  elif 'scene_opponent_counts' in total:d['executed_source_live_tick_fraction']=None;d['missing_reason']='Initial smoke saved nominal dispatch only; passive ctx telemetry added prospectively.'
  if (path.parent/'EXCLUDED.json').exists():excluded[str(path.parent)]=d
  else:results[path.parent.name]=d
 atomic_json(BASE/'TRAINING_EXPOSURE.json',dict(runs=results,excluded_runs=excluded,unix=time.time(),no_training_control=True,ability_not_inferred_from_reward=True))
if __name__=='__main__':main()
