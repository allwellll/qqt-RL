"""Freeze completed reset audits and authorized A3SR protocol before owner handoff."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/iter2/a3sr')
def main():
    jax=json.loads((B/'JAX_AUDIT.json').read_text());js=json.loads((B/'JS_AUDIT.json').read_text())
    assert jax['status']=='PASS' and jax['resets']==100000 and js['games']==1000
    green=(B/'GREEN.log').read_text();assert '[100%]' in green and 'failed' not in green
    assert jax['source_sha256']==sha(Path('scripts/bomb_threat_a3sr_spawn.py'))
    # No invented capability gate; disclose both observed early-self-defeat rates.
    safety={k:dict(games=v['games'],self_defeat_games=v['self_defeat_episodes'],self_defeat_fraction=v['self_defeat_game_fraction']) for k,v in js['groups'].items()}
    amend=dict(id='A3SR-amendment01-source-denominators',authorization='.hermes/TASK_spawn_map_random_20261030.md',
        scope='reset only; preserve original A3S real Bank, physical seat50/50 and arena mixture',
        random_probability=.5,denominator='eligible fresh empty-handed resets, including existing10% arena; certified real-acquisition Bank excluded and unchanged',
        native_fraction_caveat='50% per eligible fresh reset, not50% of all resets or occupied environment-ticks; Bank overall probability.25 before arena override',
        bank_probability=.25,conditional_bank_P0=.5,conditional_bank_P1=0,arena_fraction=.10,
        default_disabled=True,mask='only done rows inside collector; initial fresh rows at new arm initialization',
        auxiliary_state='fresh zero reward/potential/credit ledgers; no inherited map cache or causal custody, Bank aux kept unchanged',
        fallback='native positions after32 failed pairs; sampled brick removals retained, no walls/cages/base/pushables changed',
        old_spawn_extension='old fixed native-map buckets cannot address new empty cells after brick removal; separate versioned sampler over all195 cells',
        numpy_and_jax='independent RNG implementations; NumPy fixtures use productionJS audit, actual JAX sampler100k separately recorded; no bitwise cross-RNG claim',
        safety_diagnostic=safety,JS_scope='first200ticks hardHunter vs hardHunter, not full-game capability or policy proof',unix=time.time())
    atomic_json(B/'AMENDMENT01.json',amend)
    files=['scripts/bomb_threat_a3sr_spawn.py','scripts/bomb_threat_a3sr_collector.py','scripts/bomb_threat_a3sr_train.py','scripts/bomb_threat_a3sr_schedule.py',
        'scripts/bomb_threat_seat5050_collector.py','scripts/bomb_threat_seat5050_train.py','scripts/bomb_threat_seat5050_reward.py']
    proto=dict(id='BOMB-THREAT-A3SR-20261030',status='AUTHORIZED_READY',authorization='.hermes/TASK_spawn_map_random_20261030.md',
        authorization_sha256=sha(Path('.hermes/TASK_spawn_map_random_20261030.md')),amendment=str(B/'AMENDMENT01.json'),
        slot=5,seed=2026112905,formal_updates=640,formal_environment_steps=167772160,warmup_updates=20,warmup_environment_steps=5242880,
        smoke_updates=2,smoke_counts_in_budget=True,minimum_smoke_env_per_second=18000,chunk_updates=320,save_eval_every=40,
        parent='slot01 latest atomic A3 Actor at safe handoff, no score selection',reward='A3 unchanged threat/topology/chain x5 cap.6 after base-reward warmup',
        optimizer='same A3S Adam/lr7.5e-5/entropy.005/ramp40',dagger=0,combat_kl=0,
        envs_per_device=2048,rollout_ticks=128,global_minibatch=8192,epochs=1,devices=1,
        evaluation='unchanged native stock-window-v1 map/spawns; dev+holdout greedy seat split; comparison original A3S4/7 fixed1280',
        criteria=dict(P1_win_minus_loss_gain_at_least=5,P0_decline_at_most=3,single_zero_not_increased=True,self_harm_not_increased=True,P1_window_pickup_delivery_not_decreased=True),
        observation='same9 P1 brick-separated low_bomb_diag seeds, descriptive only',
        terminal='stop at640, asynchronous native dev+holdout, no implicit promotion or extension',
        gates=dict(numpy_resets=100000,jax_resets=100000,JS_games=1000,tests=str(B/'GREEN.log')),
        evidence={str(p):sha(p) for p in [B/'JAX_AUDIT.json',B/'JS_AUDIT.json',B/'RESET_AUDIT.json',B/'AMENDMENT01.json',B/'GREEN.log']},
        versioned_sources={f:sha(Path(f)) for f in files},unix=time.time())
    atomic_json(B/'PROTOCOL.json',proto)
    result=dict(status='RESET_AUDITS_COMPLETE',numpy_audit=str(B/'RESET_AUDIT.json'),jax_audit=str(B/'JAX_AUDIT.json'),
        JS_audit=str(B/'JS_AUDIT.json'),actual_jax=jax,JS_self_harm=js,not_training_exposure=True,unix=time.time())
    atomic_json(B.parent/'A3SR_SPAWN_AUDIT.json',result)
    atomic_json(B/'TASK_CONSUMED.json',dict(status='GATES_PASSED_OWNER_HANDOFF_READY',protocol=str(B/'PROTOCOL.json'),protocol_sha256=sha(B/'PROTOCOL.json'),unix=time.time()))
    print(json.dumps(dict(status='AUTHORIZED_READY',safety=safety)))
if __name__=='__main__':main()
