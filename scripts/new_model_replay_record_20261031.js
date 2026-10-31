'use strict';
// Passive recorder around the existing production-JS evaluator. CPU inference.
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const B=process.env.NEW_REPLAY_BASE||'runs/new_model_replays_20261031';
const [label,startText,endText]=process.argv.slice(2),start=Number(startText),end=Number(endText);
const cfg=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json')),models=JSON.parse(fs.readFileSync(B+'/MODELS.json'));
const m=models[label],condition=cfg.conditions.hunter_hard;
assert(m&&start>=0&&end<=256&&start<end);
assert.equal(sha(fs.readFileSync(m.model_json)),m.model_sha256);
assert.equal(sha(fs.readFileSync(condition.spawn_manifest)),condition.spawn_sha256);
process.env.QQT_STOCK_RULE='stock-window-v1';process.env.BUN_V2_RULE='bun-native-v2';
process.env.BUN_V2_TIER='hard';process.env.BUN_V2_OPPONENT='bun.hunter';
const Q=require('../web/sim-bun-native-v2');
const P=require('../../cx_publish_r1_1805_20261030/web/review_protocol');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const spawn=JSON.parse(fs.readFileSync(condition.spawn_manifest));
const original=Q.Sim.prototype.step,recordings=new Map();
Q.Sim.prototype.step=function(actions){
 if(!this.__btIdentity)return original.call(this,actions);
 const seed=this.__btIdentity.seed;
 let record=recordings.get(seed);
 if(!record){record={sim:this,actions:[],checks:[[0,sha(JSON.stringify(P.state(this)))]],events:[]};recordings.set(seed,record);}
 record.actions.push([actions[0][0],actions[0][2]?2:Number(actions[0][1]),actions[1][0],actions[1][2]?2:Number(actions[1][1])]);
 const info=original.call(this,actions);
 for(let p=0;p<2;p++){
  if(info.placed[p])record.events.push({tick:this.t,kind:'bomb',player:p});
  if(info.died[p])record.events.push({tick:this.t,kind:'death',player:p});
 }
 if(this.t%200===0||this.done)record.checks.push([this.t,sha(JSON.stringify(P.state(this)))]);
 return info;
};
const dir=`${B}/scan/${label}/shard_${String(start).padStart(3,'0')}`;
fs.mkdirSync(dir,{recursive:true});
// Scheduling only: bounded prefetch workers and the sequential prefix consumer
// share this atomic batch lock, so a frozen seed is never inferred twice.
const proofPath=dir+'/ACCEPTED.json',lockPath=dir+'/recording.lock';
const waitCell=new Int32Array(new SharedArrayBuffer(4)),deadline=Date.now()+1800000;
while(true){
 if(fs.existsSync(proofPath)){
  const p=JSON.parse(fs.readFileSync(proofPath));assert.equal(p.status,'PASS');
  for(const [f,h]of Object.entries(p.files))assert.equal(sha(fs.readFileSync(f)),h);
  console.log(JSON.stringify({status:'REUSED_LOCKED_BATCH',label,start,end}));process.exit(0);
 }
 try{fs.mkdirSync(lockPath);break;}catch(e){if(e.code!=='EEXIST')throw e;}
 assert(Date.now()<deadline,'batch lock did not release; preserve evidence instead of duplicate evaluation');
 Atomics.wait(waitCell,0,0,500);
}
process.on('exit',()=>{try{fs.rmdirSync(lockPath);}catch(e){if(e.code!=='ENOENT')throw e;}});
process.env.JE10_ROUTE_TASK='natural_full2400';process.env.JE10_ROUTE_GAMES='256';
process.env.JE10_ROUTE_START=String(start);process.env.JE10_ROUTE_END=String(end);
process.env.JE10_ROUTE_SEED_START=String(condition.seed);
process.argv=[process.argv[0],process.argv[1],'greedy',m.model_json,dir+'/evaluation.json',
 'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',condition.spawn_manifest];
require('./r3_items_candidate_eval');
let completed=false;
process.on('beforeExit',async()=>{
 if(completed)return;completed=true;
 const output=JSON.parse(fs.readFileSync(dir+'/evaluation.json'));
 const side=JSON.parse(fs.readFileSync(dir+'/evaluation.json.outcome.json'));
 assert.equal(side.engine,'production_JS');assert.equal(side.mode,'greedy');assert(!side.teacher);
 assert.equal(side.rule_version,'bun-native-v2');assert.equal(side.episodes.length,end-start);
 const games=[];
 for(const expected of side.episodes){
  const {sim:s,actions,checks,events}=recordings.get(expected.seed);
  assert(s.done);assert.equal(s.t,expected.ticks);assert.equal(actions.length,s.t);
  const sourceIndex=expected.seed-condition.seed,index=m.index_base+sourceIndex,seat=expected.seat;
  assert.equal(seat,sourceIndex%2);
  const pair=spawn.spawn_cells[sourceIndex%spawn.spawn_cells.length];
  require('./bun_spawn_side').validate(pair);assert.deepEqual(expected.spawn_cells,pair);assert.deepEqual(expected.respawn_cells,pair);
  const ep=output.results.natural_full2400.episodes.find(e=>e.seed===expected.seed);
  const meta={id:`new-${label}-${expected.seed}`,seed:expected.seed,tick_ms:100,max_ticks:2400,
   scene:'native',native_rules:true,rule_version:'bun-native-v2',model:m.display_name,
   checkpoint:m.checkpoint,checkpoint_sha256:m.policy_sha256,model_sha256:m.model_sha256,
   training_seed:m.training_seed,simulator_sha256:sha(fs.readFileSync('web/sim-bun-native-v2.js')),
   opponent_sha256:sha(fs.readFileSync('web/bun_hunter_bot.js')),map_sha256:await P.hash(level),map:806,
   policy:'greedy',teacher:false,opponent:'production JS Hunter/hard',learner_seat:seat,
   reward_status:'unrecorded',training_reward_status:'unrecorded',spawn_side_version:'own-side-native-v1',
   spawn_cells:pair,cages:level.bun_bases,frozen_birth:{schema:'qqt.frozen-birth/v1',cells:pair},
   recorded_at:new Date().toISOString()};
  const doc={schema:P.SCHEMA,meta,actions,checks,events,rewards:[],
   summary:{ticks:s.t,done:s.done,winner:s.winner,reward:[null,null],failures:[]}};
  meta.replay_identity=await P.hash(P.identity(doc));P.validate(doc);await new P.Player(doc,level).verify();
  const dest=`${B}/game_${String(index).padStart(3,'0')}`;fs.mkdirSync(dest,{recursive:true});
  fs.writeFileSync(dest+'/review.json',JSON.stringify(doc));
  fs.writeFileSync(dest+'/MATCH.json',JSON.stringify({status:'MATCH',index,review_verified:true,
   review_sha256:sha(fs.readFileSync(dest+'/review.json')),full_actions_state_hash_terminal:true,
   evaluation_sha256:sha(fs.readFileSync(dir+'/evaluation.json')),outcome_sha256:sha(fs.readFileSync(dir+'/evaluation.json.outcome.json'))},null,2));
  games.push({index,source_index:sourceIndex,seed:expected.seed,seat,
   expected:{...expected,primary_delivery:ep.delivery},model:m});
 }
 const proof={status:'PASS',games,files:Object.fromEntries(['evaluation.json','evaluation.json.outcome.json'].map(n=>[dir+'/'+n,sha(fs.readFileSync(dir+'/'+n))])),unix:Date.now()/1000};
 fs.writeFileSync(dir+'/ACCEPTED.json.tmp',JSON.stringify(proof,null,2));fs.renameSync(dir+'/ACCEPTED.json.tmp',dir+'/ACCEPTED.json');
 console.log(JSON.stringify({status:'PASS',label,start,end,games:games.length,full_review_verified:true}));
});
