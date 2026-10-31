'use strict';
// Read-only replay and inference instrumentation. Never changes recorded actions.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const assert=require('node:assert/strict');
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
function selectedTicks(length,first){
  const selected=new Set();
  for(let i=0;i<length;i+=16)selected.add(i);
  if(length)selected.add(length-1);
  if(first!==null)for(let i=Math.max(0,first-8);i<Math.min(length,first+9);i++)selected.add(i);
  return new Set([...selected].sort((a,b)=>a-b));
}
function pack(z,mm,bm){
  assert.equal(z.obs.length,4680,'obs shape');assert.equal(z.state.length,24,'state shape');
  assert.equal(mm.length,5,'move mask shape');assert.equal(bm.length,3,'ability mask shape');
  assert.equal(z.move.length,5,'move output shape');assert.equal(z.bomb.length,3,'ability output shape');
  return Float32Array.from([...z.obs,...z.state,...mm,...bm,...z.move,...z.bomb]);
}
async function run(checkpoint,output){
  const P=require('../web/review_protocol'),authority=require('./eval_web_hunter');
  const inference=require('./je05_web_hunter');
  const folder=path.join(checkpoint,'retention');
  const replayFile=path.join(folder,'REVIEW_FIXED_NATIVE.json');
  const traceFiles=fs.readdirSync(folder).filter(x=>/^fixed-natural_full2400-.*\.trace.json$/.test(x));
  assert.equal(traceFiles.length,1);
  const traceFile=path.join(folder,traceFiles[0]);
  const replay=JSON.parse(fs.readFileSync(replayFile)),trace=JSON.parse(fs.readFileSync(traceFile));
  assert.equal(replay.meta.seed,2026102480);assert.equal(replay.meta.policy,'greedy');
  assert.deepEqual(replay.actions,trace.actions);assert.equal(trace.task,'natural_full2400');
  assert.equal(replay.actions.length,trace.diagnostic_ticks.length);
  const player=new P.Player(replay,authority.loadLevel());
  assert.equal(hash(JSON.stringify(P.state(player.sim))),hash(JSON.stringify(trace.initial)));
  const modelFile=path.join(folder,'model/model.json');
  const loaded=await inference.loadModel(modelFile),forward=loaded.model.forward.bind(loaded.model);
  let last;
  loaded.model.forward=async(obs,state)=>{const z=await forward(obs,state);last={obs:Float32Array.from(obs),state:Float32Array.from(state),move:Array.from(z.move),bomb:Array.from(z.bomb)};return z;};
  const chosen=selectedTicks(replay.actions.length,trace.first_deviation?.tick??null);
  const bytes=[],rows=[];let steps=0;
  const recordedChecks=new Map(replay.checks);
  const noRandom=()=>{throw Error('deterministic inference unexpectedly draws RNG');};
  for(let t=0;t<replay.actions.length;t++){
    const s=player.sim;
    assert.equal(s.t,t);assert.equal(hash(JSON.stringify(P.state(s))),trace.diagnostic_ticks[t].state_before_sha256,'pre-action state hash');
    if(recordedChecks.has(t))assert.equal(hash(JSON.stringify(P.state(s))),recordedChecks.get(t));
    const result=await loaded.model.act(s,0,noRandom);
    const actual=[result[0],result[2]?2:result[1]?1:0];
    assert.deepEqual(actual,replay.actions[t].slice(0,2),'production greedy action differs from immutable trace');
    if(chosen.has(t)){
      const mask=s.legalMask();const row=pack(last,mask.mm[0],loaded.model._abilityMask(s,mask,0));
      bytes.push(Buffer.from(row.buffer));rows.push({tick:t,phase:trace.diagnostic_ticks[t].phase,state_sha256:trace.diagnostic_ticks[t].state_before_sha256,action:actual});
    }
    assert.ok(player.step());steps++;
  }
  assert.equal(player.sim.done,replay.summary.done);assert.equal(player.sim.winner,replay.summary.winner);
  if(recordedChecks.has(player.sim.t))assert.equal(hash(JSON.stringify(P.state(player.sim))),recordedChecks.get(player.sim.t));
  fs.mkdirSync(output,{recursive:true});const binary=Buffer.concat(bytes);fs.writeFileSync(path.join(output,'frames.f32'),binary);
  const receipt={passed:true,rows,columns:4720,frames_sha256:hash(binary),replay_sha256:hash(fs.readFileSync(replayFile)),trace_sha256:hash(fs.readFileSync(traceFile)),policy_sha256:replay.meta.checkpoint_sha256,model_json_sha256:hash(fs.readFileSync(modelFile)),onnx_sha256:loaded.model.backendSha,simulator_sha256:hash(fs.readFileSync('web/sim.js')),reexecuted_physics_steps:steps,new_independent_environment_samples:0,optimizer_steps:0,all_recorded_state_hashes_and_greedy_actions_exact:true};
  fs.writeFileSync(path.join(output,'FRAMES.json'),JSON.stringify(receipt));
  console.log(path.basename(checkpoint),rows.length,'frames',steps,'replayed steps exact');
}
if(require.main===module)run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={selectedTicks,pack,run};
