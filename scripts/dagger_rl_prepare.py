"""Freeze train-only mechanism audit and paired schedule before outcomes."""
import shutil,time,subprocess,sys,os
from scripts.dagger_rl_run import B,R,read
from scripts.population_run_state import atomic_json,sha

def prepare():
 assert not (B/'PROTOCOL.json').exists()
 prior=R/'runs/combat_item_20261023/v1';old=read(prior/'PROTOCOL.json');parent=old['parent'];cp=R/parent['checkpoint']
 assert sha(cp/'policy.pt')==parent['policy_sha256'] and sha(cp/'state.pkl')==parent['state_sha256']
 (B/'inputs').mkdir(exist_ok=True);shutil.copyfile(prior/'inputs/parent.pkl',B/'inputs/parent.pkl')
 (B/'cpu_eval_inputs').mkdir(exist_ok=True)
 for f in ['spawns.json','regions.json']:shutil.copyfile(prior/'cpu_eval_inputs'/f,B/'cpu_eval_inputs'/f)
 p=dict(id='DAGGER-RL-20261024-R1-v2',supersedes='r1 preformal only: known respawn omission, no formal run consumed',mode=old['mode'],parent=parent,parent_input_sha256=sha(B/'inputs/parent.pkl'),optimizer='identical saved Adam; DAgger-only freezes critic params/mu/nu; shared count advances; no reset at switch',
 arms=['sequential','joint','ppo'],paired_seeds=[2026102411,2026102412,2026102413],updates=32,switch_update=16,
 treatment='schedule of the SAME safe set-valued route+banana learner-state aggregation loss; sequential u1..16 DAgger-only then u17..32 pure on-policy PPO; joint PPO+DAgger32; ppo pure PPO32. No reward/network/opponent/curriculum change between arms.',
 common='Frozen previous control reward, persistent genuine carry, death/native-trap stock teacher/PBRS only. No combat auxiliary teacher. No glue teacher/new reward. Free-alive spontaneous theft legal without auxiliary stock reward.',
 train_mixture=old['train_mixture'],opponents=old['opponents'],bank_source=old['bank_source'],bank_sha256=old['bank_sha256'],bank_limits=old['bank_limits'],
 oracle='Current public-state native AABB complete banana slide forecast64 ticks; release away, arm by leaving original cell, return/contact toward home, slide to collision. >.5 cell benefit after .18*t walking cost; abstain fuses/blasts/pushables/near enemy/other fields/incomplete slides; future native revival/expiry uses CURRENT public timer+spawn position. Forecast not future-enemy guarantee; diagnostic causal native parity required.',
 aggregation='Persist bounded FIFO8192 accepted learner-state labels; new and old .5/.5 supervised loss; route/item separated equally when present. Query EVERY student state allarms; never override actions. Same opportunity support/caps, achieved exposure reported rather than claimed identical. Old labels used only for DAgger, PPO always fresh on-policy.',
 envs=64,steps=128,environment_steps_per_run=262144,total_environment_steps=2359296,optimizer_steps_per_run=1024,total_optimizer_steps=9216,checkpoints=[0,1,8,16,17,24,32],
 lr=7.5e-5,gamma=.995,gae_lambda=.95,clip=.1,value=.5,entropy=.005,gradient_clip=.5,epochs=1,minibatch=256,dagger_coefficient=.2,
 scale_selection='fixed .2 existing route coefficient; no holdout scan; train-only real opportunity/gradient/mechanism audit. Signal absent means technical protocol correction, not equivalent formal arms.',
 audit_budget='Reuse frozen first32 actual parent-student opportunities and native branches from r1, requery corrected visible-respawn oracle before outcomes; disclose every abstention/source. Unchanged accepted masks reuse exact physical branches, native rules unchanged. Closed-loop requery all accepted states CPU. No extra environment sampling; no diagnostic interventions in training.',
 evaluation=dict(checkpoints='all saved points immutable; natural dev64 at16/32, final parent+9models natural dev/holdout64, JS Hunter hard/normal/easy64 each, Python Tactical64, native Bank/Bank-P dev/holdout32x8; switch16/17 action-chain diagnostics',natural_dev_seed=2026102480,natural_holdout_seed=2026110100,hunter_seed=20261024,selection='paired final, last2 tail and worstseed/point, untrained parent AND matched PPO control; independent source CIs, no peak promotion; evaluation never controls launches'),
 successor=dict(question='Does sequence learn release+safe self-contact then retain it under pure RL?',if_promising='independent5 paired seeds confirm only after frozen final natural/safety evidence, separate bounded preregistration',if_zero='version coverage/teacher protocol; no repeat equivalent arms',if_regression='first-deviation/phase retention diagnosis, next single-variable preregistration; no blind multiplier increase',if_inconclusive='report uncertainty and required new budget, no unrelated GPU fill'),
 stopping='fixed9 runs; technical failure or user stop safe every-update atomic; no short ability gates. CPU independent unique owner, committed jobs+markers. No old continuous/reserve/auto publish.',unix=time.time())
 atomic_json(B/'PROTOCOL.json',p)
 atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json',dict(authorized_by='explicit_user_instruction',old_continuous_resume=False,old_global_pause_sha256=sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json'),bounded_protocol_sha256=sha(B/'PROTOCOL.json')))
 subprocess.run([sys.executable,'scripts/eval_four_bot_matrix.py','spawns','--games','64','--seed','20261024','--out',str((B/'cpu_eval_inputs/spawns.json').relative_to(R))],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu'),check=True)
 assert read(B/'cpu_eval_inputs/spawns.json')['seed']==20261024
 atomic_json(B/'CPU_EVAL_PROTOCOL.json',dict(read(prior/'CPU_EVAL_PROTOCOL.json'),hunter_seed=20261024,natural_seed_start=2026102480,holdout_seed_start=2026110100,spawns_sha256=sha(B/'cpu_eval_inputs/spawns.json')))
 atomic_json(B/'MODE_IDENTITY.json',dict(mode=p['mode'],parent_policy_sha256=parent['policy_sha256'],no_other_game_lineage=True))
 print(p['id'],sha(B/'PROTOCOL.json'))
if __name__=='__main__':prepare()
