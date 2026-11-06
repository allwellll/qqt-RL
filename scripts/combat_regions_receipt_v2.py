"""Observe real counted/formal updates and exact full-PPO restore evidence."""
import json
import time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json, sha

B = Path('runs/combat_regions_independent_20261104')


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def main():
    current = read(B/'training/CURRENT.json'); p = read(current['protocol'])
    out = Path(p['namespace'])/current['run']; start=p['campaign_start_update']
    while True:
        rows = []
        if (out/'updates.jsonl').exists():
            for line in (out/'updates.jsonl').read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass  # Concurrent writer may have an incomplete last row.
        if rows and not (B/'FIRST_COUNTED_UPDATE.json').exists():
            atomic_json(B/'FIRST_COUNTED_UPDATE.json',dict(status='OBSERVED_SMOKE_NOT_FORMAL',row=rows[0]))
        restore=read(out/'RESTORE.json');formal=[r for r in rows if r['new_update']>start+2]
        if formal and restore:
            row=formal[0]
            assert restore['status']=='PASS' and restore['fresh_process'] and restore['full_Actor_Critic_Adam_env_aux_RNG_league_exact']
            assert row['finite'] and min(row['update_l2'].values())>0 and row['ability_update_l2']>0
            assert np.isfinite(row['loss']).all()
            e=row['exposure']['bun'];resets=np.asarray(e['multi_region_resets'])
            assert resets[1:].sum()>0 and e['multi_region_legal_fallbacks']==0
            assert all(sha(f)==h for f,h in p['source_sha256'].items())
            receipt=dict(status='PASS_ACTUAL_FIRST_FORMAL_UPDATE',update=row['new_update'],source_update=start,
                additional_counted_updates=row['new_update']-start,remaining_updates=start+1280-row['new_update'],
                finite=True,actual_environment_steps=row['actual_environment_steps_this_update'],
                actual_optimizer_steps=row['actual_optimizer_steps_this_update'],
                update_l2=row['update_l2'],ability_update_l2=row['ability_update_l2'],
                fresh_full_ppo_restore=restore,cycle_seconds=row['cycle_seconds'],env_per_second=row['env_per_second'],
                total_resets=float(resets.sum()),regional_resets=float(resets[1:].sum()),
                regional_fraction=float(resets[1:].sum()/resets.sum()),legal_fallbacks=0,
                continuous_telemetry={k:v for k,v in e.items() if k.startswith('ca_')},
                source_sha256=p['source_sha256'],row=row,capability_improvement='NOT_ESTABLISHED',unix=time.time())
            if not (B/'FIRST_FORMAL_UPDATE.json').exists():atomic_json(B/'FIRST_FORMAL_UPDATE.json',receipt)
            if len(formal)>=5:
                last=formal[-5:]
                assert all(r['finite'] for r in last)
                atomic_json(B/'STEADY_UPDATES.json',dict(status='PASS',updates=[r['new_update'] for r in last],
                    median_env_per_second=float(np.median([r['env_per_second'] for r in last])),
                    median_cycle_seconds=float(np.median([r['cycle_seconds'] for r in last])),unix=time.time()))
                return
        status=read(B/'training/OWNER_STATUS.json',{})
        if status.get('status')=='TECHNICAL_BLOCKED':
            atomic_json(B/'STARTUP_BLOCKED.json',status);return
        time.sleep(5)


if __name__=='__main__':main()
