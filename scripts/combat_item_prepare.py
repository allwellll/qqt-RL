"""Freeze the bounded paired experiment before any candidate capability evaluation."""
import json,shutil,time
from scripts.combat_item_run import B,R,read
from scripts.population_run_state import atomic_json,sha

def prepare():
 assert not (B/'PROTOCOL.json').exists(),'new namespace only'
 B.mkdir(parents=True)
 prior=R/'runs/carrier_mechanism_20261023/v1'
 parent=read(prior/'PROTOCOL.json')['models']['parent']
 cp=R/parent['checkpoint']
 assert sha(cp/'policy.pt')==parent['policy_sha256'] and sha(cp/'state.pkl')==parent['state_sha256']
 (B/'inputs').mkdir();shutil.copyfile(prior/'inputs/parent.pkl',B/'inputs/parent.pkl')
 (B/'cpu_eval_inputs').mkdir()
 for f in ['spawns.json','regions.json']:shutil.copyfile(prior/'cpu_eval_inputs'/f,B/'cpu_eval_inputs'/f)
 protocol=dict(id='COMBAT-ITEM-20261023-v1',mode='Bun806 native trap/items,24-channel actor,independent Actor/Critic; production JS final truth',
   causal_design='Three independent single-variable comparisons against a shared matched control; no combined treatment',
   parent=parent,parent_input_sha256=sha(B/'inputs/parent.pkl'),optimizer='matching full saved Adam, no reset, same copy all arms',
   arms=['control','combat','banana','glue'],paired_seeds=[2026102311,2026102312,2026102313],
   treatments=dict(control='native legacy rewards + persistent safe genuine-return/window-stock learner-state set-mass .2',
       combat='only add .05 native hard-bot safe combat set-mass on empty/free-alive learner states',
       banana='only replace legacy banana reward with complete owned slide net-home-distance after .18-cell/tick walking cost; min(.5,.2*gain), cap2, loss clawback',
       glue='only replace legacy owned glue hit reward with completed actual chase displacement suppression and home-progress; min(.5,.25*delay), cap2, loss clawback'),
   common_semantics='Stock teacher/PBRS native death/trapped only; slow is not native trapped. Carry continues home after revival, hazards retain safety filter. Free spontaneous stock acquisition legal without empty-stock auxiliary reward. Genuine loose/contact remain native PPO events.',
   train_mixture='75% natural native-spawn map806, 25% complete train-only real acquisition snapshots; natural inventory none/banana/glue uniformly granted only at spawn, never cargo. All arms same reset RNG and support.',
   bank_source=str(prior.relative_to(R)),bank_sha256=sha(prior/'BANK_FROZEN.json'),bank_limits='map806 only, learner P0, allfar; train/native32 dev/open32 holdout/checker32. Already used retention set: not novel independent confirmation.',
   opponents=dict(current=0,ring=0,fixed=1.,fixed_identity='JAX rule bot tier3; native hard teacher tier4, no action override; Python Tactical/JS Hunter separate'),
   updates=32,envs=64,steps=128,environment_steps_per_run=262144,total_environment_steps=3145728,
   optimizer_steps_per_run=1024,total_optimizer_steps=12288,checkpoints=[0,1,8,16,24,32],
   lr=7.5e-5,gamma=.995,gae_lambda=.95,clip=.1,value=.5,entropy=.005,gradient_clip=.5,epochs=1,minibatch=256,
   scale_selection='No scan: bounded coefficients frozen from existing .5 item-event cap and .05 existing combat supervision; train-only first batch magnitude audit, no holdout tuning',
   eval=dict(dev='Bank dev at u1/8/16/24/32, natural JS64 at u16 (same declared protocol)',
       final='All12 final + shared no-update control: Python Tactical64; production JS Hunter hard/normal/easy64; natural JS64; Bank and Bank-P dev/holdout32x8; holdout natural64 disjoint frozen seeds',
       seeds=dict(hunter=20261023,natural_dev=2026102380,natural_holdout=2026102700),
       selection='paired seed final and tail curves, worst seed/checkpoint, no peak promotion; safety/control/opportunity denominators; no public release'),
   stopping='Fixed bounded12 runs, technical failure repair/resume only; no short capability gate. GPU Bank evaluation then exit, no automatic old/reserve successor.',
   async_owner='scripts.combat_item_cpu; immutable committed hash jobs, per-checkpoint COMPLETE marker, no GPU launch capability',
   unix=time.time())
 atomic_json(B/'PROTOCOL.json',protocol)
 atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json',dict(authorized_by='explicit_user_instruction',old_continuous_resume=False,old_global_pause_sha256=sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json'),bounded_protocol_sha256=sha(B/'PROTOCOL.json')))
 atomic_json(B/'CPU_EVAL_PROTOCOL.json',dict(read(prior/'CPU_EVAL_PROTOCOL.json'),holdout_seed_start=2026102700))
 shutil.copyfile('/tmp/combat_item_RED.log',B/'TDD_RED.log')
 atomic_json(B/'SOURCE_SEAL.json',dict(sources={str(p.relative_to(R)):sha(p) for p in list((R/'scripts').glob('combat_item*.*'))+list((R/'tests').glob('test_combat_item*'))},parent_input_sha256=sha(B/'inputs/parent.pkl')))
 print(json.dumps(protocol,ensure_ascii=False,indent=2))
if __name__=='__main__':prepare()
