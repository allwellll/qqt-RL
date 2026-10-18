'use strict';
// Historical compatibility fixtures remain verified independently of public admission.
// The current catalog is exhaustively gated by test_review_admission.js.
const assert=require('assert'),fs=require('fs'),zlib=require('zlib'),os=require('os'),path=require('path'),{spawnSync}=require('child_process');
const R=require('./review_route'),P=require('./review_protocol'),A=require('./review_advantage'),E=require('../scripts/export_diagnostic_route');
const fixture=require('../tests/fixtures/route_replay');
(async()=>{
  assert.equal(R.signed(.00025),'+0.00025');assert.equal(R.signed(-.000001),'-0.000001');
  const {replay,source}=fixture(),oldIdentity=await P.hash(P.identity(replay));
  assert.equal(R.atTick(replay,1).route_potential_delta,null);assert.equal(R.atTick(replay,1).shaped_advantage_a1,null);
  const recomputed=structuredClone(source);
  const {document:d,report}=await E.convert(replay,source,recomputed);
  P.validate(d);assert.notEqual(await P.hash(P.identity(d)),oldIdentity);
  assert.equal(await P.hash(P.identity(replay)),oldIdentity,'do not mutate old replay');
  assert.equal(R.atTick(d,1).route_potential_delta,1.055);assert.equal(R.atTick(d,2).route_potential_delta,-.94);
  assert.equal(R.atTick(d,3).route_potential_delta,null,'no forward filling');
  assert.equal(R.atTick(d,4).distance_before,'INF');assert.equal(R.atTick(d,4).phase_transition,true);
  assert.equal(R.atTick(d,d.actions.length).phi_after,0);assert.equal(R.atTick(d,1).shaped_advantage_a1,null);
  assert.match(R.atTick(d,1).target_id,/^target-[a-f0-9]{64}$/);
  assert(!JSON.stringify(d.route).includes('fixture-object'));assert.equal(report.raw_bytes,Buffer.byteLength(JSON.stringify(d)+'\n'));
  assert.equal(report.gzip_bytes,zlib.gzipSync(JSON.stringify(d)+'\n',{level:9}).length);
  for(const mutate of [x=>x.route.route_potential_delta[0]+=10,x=>x.route.route_shaped_reward[0]+=50,
    x=>x.route.ticks[1]=1,x=>x.route.phi_before.pop(),x=>x.route.phi_before[1]+=10,x=>x.route.distance_before[0]=Infinity,
    x=>x.route.distance_after[0]=-1,x=>x.route.targets[0].id='private-id',x=>x.route.full_states=[],x=>x.meta.route.local_path='forbidden',x=>x.route.topology_evidence[2]=null,
    x=>x.route.shaped_advantage_a1[0]=1,x=>x.route.phi_after[x.route.ticks.length-1]=1,
    x=>x.meta.route.policy_sha256='0'.repeat(64),x=>x.route.phase_transition[0]='true',
    x=>x.route.route_phase[0]='invented',x=>x.route.route_phase[0]='A_ENEMY_DROP',x=>x.route.invalidated[2]=true,x=>x.route.safety_reject_reason[0]='unexplained prose',
    x=>{x.route.topology_gain[0]=1;x.route.safety_reject_reason[0]='unsafe';},x=>x.route.terminal[0]=true]){
    const bad=structuredClone(d);mutate(bad);assert.throws(()=>P.validate(bad),/route/);
  }
  const missing=structuredClone(source);missing.events[0].phi_before=null;
  missing.events[0].route_potential_delta=null;missing.events[0].route_shaped_reward=null;
  const m=(await E.convert(replay,missing,structuredClone(missing))).document;assert.equal(R.atTick(m,1).phi_before,null);
  const onlyNull=structuredClone(replay);onlyNull.meta.route=null;onlyNull.route=null;P.validate(onlyNull);assert.equal(R.atTick(onlyNull,1).route_phase,null);
  const bad=structuredClone(recomputed);bad.events[0].phi_after++;
  await assert.rejects(E.convert(replay,source,bad),/recomput/);
  const shaped=structuredClone(source);shaped.meta.shaped_critic={schema:'qqt.route-shaped-a1/v1',trained_on_route_reward:true,
    checkpoint_sha256:replay.meta.checkpoint_sha256,critic_sha256:'b'.repeat(64),reward_ledger_sha256:source.meta.reward_ledger_sha256,
    counterfactual_sha256:'c'.repeat(64),action_scope:'joint_move_ability',legal_actions:'production mask; joint renormalization'};
  const candidate=t=>({tick:t,taken:replay.actions[t-1].slice(0,2),candidates:[
    {action:replay.actions[t-1].slice(0,2),legal:true,probability:.2,reward:1,nextValue:2,terminal:false},
    {action:[99,99],legal:false,probability:100,reward:999,nextValue:999,terminal:false},
    {action:[(replay.actions[t-1][0]+1)%5,replay.actions[t-1][1]],legal:true,probability:.3,reward:0,nextValue:1,terminal:true}]});
  shaped.shaped_results=[candidate(1),candidate(2),candidate(4)];
  const s=(await E.convert(replay,shaped,structuredClone(shaped))).document;
  assert(Math.abs(R.atTick(s,1).shaped_critic_q1_taken-2.99)<1e-6);
  assert(Math.abs(R.atTick(s,1).shaped_critic_q1_policy_mean-1.196)<1e-6);
  assert(Math.abs(R.atTick(s,1).shaped_advantage_a1-1.794)<1e-6);
  assert.equal(R.atTick(s,1).shaped_advantage_delta,null);assert.equal(R.atTick(s,2).shaped_advantage_delta,0);
  assert.equal(R.atTick(s,4).shaped_advantage_delta,null);
  for(const change of [x=>x.meta.route.shaped_critic.trained_on_route_reward=false,x=>x.route.shaped_advantage_delta[2]=0,x=>x.route.shaped_advantage_a1[0]+=10,x=>x.meta.route.shaped_critic.critic_sha256=null]){
    const bad=structuredClone(s);change(bad);assert.throws(()=>P.validate(bad),/route/);
  }
  const {document:again}=await E.convert(replay,source,recomputed);assert.deepEqual(again,d,'deterministic regeneration');
  const mutated=structuredClone(d);mutated.route.phi_after[0]++;assert.notEqual(await P.hash(P.identity(mutated)),d.meta.replay_identity);
  const rewardTick=replay.rewards.find(r=>r[0]>4&&r[0]<replay.actions.length)?.[0];
  assert(R.summary(d,rewardTick).includes('路线奖励'));assert.deepEqual(d.rewards.find(r=>r[0]===rewardTick),replay.rewards.find(r=>r[0]===rewardTick),'shared tick uses original reward evidence once');
  const taskReplay=structuredClone(replay),taskSource=structuredClone(source);
  taskReplay.meta.advantage=A.missing({...taskReplay.meta,gamma:.995});taskReplay.advantages=A.empty();
  taskReplay.meta.replay_identity=await P.hash(P.identity(taskReplay));taskSource.replay_identity=taskReplay.meta.replay_identity;
  taskSource.task_results=[candidate(1),candidate(2),candidate(4)];
  taskSource.task_advantage={...taskReplay.meta.advantage,status:'recorded',reason:'synthetic original task critic fixture only',critic_sha256:'d'.repeat(64),counterfactual_sha256:'e'.repeat(64)};
  const task=(await E.convert(taskReplay,taskSource,structuredClone(taskSource))).document;
  assert(Math.abs(A.atTick(task,1).advantage-1.794)<1e-6);assert.equal(R.atTick(task,1).shaped_advantage_a1,null);
  assert.equal(A.atTick(task,4).delta,null);
  const verifyExisting=structuredClone(taskSource);verifyExisting.replay_identity=task.meta.replay_identity;
  const preserved=(await E.convert(task,verifyExisting,structuredClone(verifyExisting))).document;assert.deepEqual(preserved.advantages,task.advantages);
  verifyExisting.task_results[0].candidates[0].reward++;
  await assert.rejects(E.convert(task,verifyExisting,structuredClone(verifyExisting)),/original task mismatch/);
  assert.deepEqual(d.advantages,replay.advantages,'original task critic untouched');assert.deepEqual(d.rewards,replay.rewards);
  for(const row of require('../reports/review_catalog_before_gate_20261018.json').replays.filter(r=>!r.diagnostic)){const old=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));P.validate(old);assert.equal(R.atTick(old,1).phi_before,null);}
  const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'qqt-route-export-'));
  try{
    for(const [name,value]of [['replay',replay],['source',source],['recomputed',recomputed]])fs.writeFileSync(path.join(tmp,name+'.json'),JSON.stringify(value));
    const args=['scripts/export_diagnostic_route.js','--replay',path.join(tmp,'replay.json'),'--input',path.join(tmp,'source.json'),'--recomputed',path.join(tmp,'recomputed.json'),'--output',path.join(tmp,'output.json'),'--report',path.join(tmp,'report.json')];
    assert.equal(spawnSync(process.execPath,args,{encoding:'utf8'}).status,0);const bytes=fs.readFileSync(path.join(tmp,'output.json'));
    const report=JSON.parse(fs.readFileSync(path.join(tmp,'report.json')));assert.equal(report.raw_bytes,bytes.length);assert.equal(report.sha256,await P.sha256(bytes));
    assert.equal(spawnSync(process.execPath,args,{encoding:'utf8'}).status,1,'never overwrite');assert(bytes.equals(fs.readFileSync(path.join(tmp,'output.json'))));
    args[args.indexOf('--output')+1]='web/review-replays/forbidden-test-fixture.json';
    assert.equal(spawnSync(process.execPath,args,{encoding:'utf8'}).status,1);assert(!fs.existsSync('web/review-replays/forbidden-test-fixture.json'),'never publish a fixture');
  }finally{fs.rmSync(tmp,{recursive:true,force:true});}
  console.log('Route GREEN: old identities, null/INF/negative/phase/terminal, PBRS recompute, target redaction, isolated shaped critic weighted legal A1, first/gap delta, compact sizes, tamper RED');
})().catch(e=>{console.error(e);process.exitCode=1;});
