'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),zlib=require('zlib'),P=require('./review_protocol'),R=require('./review_route'),V=require('./review_presentation'),A=require('./review_advantage');
(async()=>{
  const c=require('./review-catalog.json'),row=c.replays[0];assert(row.diagnostic);assert.equal(row.seed,2026101732);
  const bytes=fs.readFileSync('web/review-replays/'+row.file),d=JSON.parse(bytes),proof=require('./review-evidence/route-v35-g32.json'),critic=require('./review-evidence/route-v35-critic.json');
  assert.equal(await P.sha256(bytes),row.sha256);assert.equal(bytes.length,row.bytes);assert.equal(zlib.gzipSync(bytes,{level:9}).length,row.gzip_bytes);
  assert.equal(await P.sha256(fs.readFileSync('web/review-sim/route-v35.js')),P.ROUTE_SIM_HASH);
  assert.equal(proof.ticks,1200);assert.equal(proof.actions_reinferred,1200);assert.equal(proof.state_transitions_verified,1200);assert.equal(proof.route_recalculated,1200);assert.equal(proof.max_residual,0);
  assert.equal(critic.tensor_count,75);assert(critic.all_tensors_exact);assert.equal(critic.critic_sha256,d.meta.critic_sha256);assert.equal(critic.critic_sha256,proof.files['critic_params.npz'].sha256);
  assert.equal(proof.files['trajectory.json'].sha256,d.meta.route.source_sha256);assert.equal(proof.recomputed_sha256,d.meta.route.recomputed_sha256);
  P.validate(d);assert.equal(d.route.ticks.length,1200);assert.equal(d.meta.route.semantics,'diagnostic-post-target-uncorrected/v1');assert.equal(d.summary.reward[0],null);assert.equal(d.rewards.length,0);assert.deepEqual(A.atTick(d,100),{taken_q:null,mean_q:null,advantage:null,delta:null});
  const p=new P.Player(d,require('../scripts/eval_web_hunter').loadLevel());await p.verify();const clone=new P.Player(d,p.level);let reward=0,continuous=[0,0],phaseChanges=0;
  for(let tick=1;tick<=1200;tick++){
    const r=R.atTick(d,tick);assert.notEqual(r.route_potential_delta,null);assert.equal(r.shaped_advantage_a1,null);assert.equal(r.topology_gain,null);assert.equal(r.invalidated,null);assert.equal(r.safety_reject_reason,null);
    assert(Math.abs(r.route_potential_delta-(.995*r.phi_after-r.phi_before))<=2e-6);assert(Math.abs(r.route_shaped_reward-.05*r.route_potential_delta)<=1.05e-6);reward+=r.route_shaped_reward;
    if(r.phase_transition)phaseChanges++;
    const before=p.sim;clone.step();for(let pid=0;pid<2;pid++)if(V.canInterpolate(before,clone.sim,pid,d.events,d.actions[tick-1][pid*2])){
      const values=[.1,.25,.4,.6,.75,.9].map(alpha=>V.motion(before,clone.sim,alpha,true,d.events,[d.actions[tick-1][0],d.actions[tick-1][2]]).visualPos[pid*2+(before.pos[pid*2]===clone.sim.pos[pid*2]?1:0)]);
      const sign=Math.sign(values.at(-1)-values[0]);assert(sign&&values.every((v,i)=>!i||(v-values[i-1])*sign>0));continuous[pid]++;
    }p.step();
  }
  assert(continuous.every(n=>n>10));assert(phaseChanges>0);assert(Math.abs(reward-proof.route_reward)<1200e-6);assert.notEqual(R.atTick(d,1200).phi_after,0);assert.equal(R.atTick(d,1200).terminal_correction,false);
  for(const mutate of [x=>delete x.meta.route.semantics,x=>x.meta.diagnostic=false,x=>x.meta.route.semantics='invented',x=>x.route.phi_after[0]++,x=>x.route.route_shaped_reward[0]=100,x=>x.summary.reward=[0,0],x=>x.rewards.push([1,{route_shaped_reward:1},{}]),x=>x.route.shaped_advantage_a1[0]=1]){
    const bad=structuredClone(d);mutate(bad);if(bad.route.phi_after[0]===1)bad.route.phi_after[0]=100;assert.throws(()=>P.validate(bad));
  }
  assert(!bytes.includes(Buffer.from('state_before'))&&!bytes.includes(Buffer.from('state_after')));assert(bytes.length<200000);assert(row.gzip_bytes<12000);
  console.log('Real g32 GREEN: source hashes, 75 exact critic tensors receipt, 1200 production state hashes/route equations, honest null reward/A, diagnostic terminal semantics, both players monotonic float motion '+continuous.join('/'));
})().catch(e=>{console.error(e);process.exitCode=1;});
