'use strict';
// Convert already recorded production-JS actions; no model inference or fake frames.
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const B=process.env.V2_SELECTION_OUTPUT||'runs/v2_fresh_selection_20261031',cfg=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json'));
const Q=require('../web/sim-bun-native-v2');
const P=require('../../cx_publish_r1_1805_20261030/web/review_protocol');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const models=JSON.parse(fs.readFileSync(B+'/MODELS.json')),sha=b=>crypto.createHash('sha256').update(b).digest('hex');
(async()=>{
 const games=[];let index=0;
 for(const label of ['baseline',...cfg.candidate_order]){
  const folder=`${B}/shards/hunter_hard/${label}`,file=folder+'/shard_000.json',accept=JSON.parse(fs.readFileSync(file+'.ACCEPTED.json'));
  for(const [f,h]of Object.entries(accept.files))assert.equal(sha(fs.readFileSync(f)),h);
  const expected=JSON.parse(fs.readFileSync(file+'.outcome.json')).episodes[0],m=models[label];
  const traceFile=`${folder}/fixed-natural_full2400-${expected.seed}.trace.json`,trace=JSON.parse(fs.readFileSync(traceFile));
  const spawn=JSON.parse(fs.readFileSync(cfg.conditions.hunter_hard.spawn_manifest)),pair=spawn.spawn_cells[0];
  if(cfg.spawn_side_version){require('./bun_spawn_side').validate(pair);assert.deepEqual(expected.spawn_cells,pair);assert.deepEqual(expected.respawn_cells,pair);}
  const s=new Q.Sim(expected.seed);s.reset(level,{nativeItems:true,nativeTrap:true});s.maxSteps=2400;
  for(let p=0;p<2;p++){s.pos[2*p]=pair[p][0]+.5;s.pos[2*p+1]=pair[p][1]+.5;}s.bunSpawnPos=pair.map(a=>a.map(x=>x+.5));s._gen++;
  assert.equal(await P.hash(P.state(s)),await P.hash(trace.initial));
  const replay={schema:P.SCHEMA,meta:{id:`v2-${label}-${expected.seed}`,seed:expected.seed,tick_ms:100,max_ticks:2400,
   scene:'native',native_rules:true,rule_version:'bun-native-v2',model:label,checkpoint:m.checkpoint,checkpoint_sha256:m.policy_sha256,
   model_sha256:m.model_sha256,simulator_sha256:sha(fs.readFileSync('web/sim-bun-native-v2.js')),opponent_sha256:sha(fs.readFileSync('web/bun_hunter_bot.js')),
   map_sha256:await P.hash(level),map:806,policy:'greedy',teacher:false,opponent:'production JS Hunter/hard',learner_seat:expected.seat,
   reward_status:'unrecorded',training_reward_status:'unrecorded',spawn_side_version:cfg.spawn_side_version,spawn_cells:pair,cages:level.bun_bases,frozen_birth:{schema:'qqt.frozen-birth/v1',cells:pair},
   source_trace:traceFile,source_trace_sha256:sha(fs.readFileSync(traceFile)),recorded_at:new Date().toISOString()},
   actions:trace.actions,checks:[[0,await P.hash(P.state(s))]],events:[],rewards:[]};
  const placed=[0,0],deaths=[0,0],self=[0,0];
  for(let t=0;t<trace.actions.length;t++){
   const a=trace.actions[t],info=s.step([[a[0],Number(a[1]===1),Number(a[1]===2)],[a[2],Number(a[3]===1),Number(a[3]===2)]]);
   assert.equal(await P.hash(P.state(s)),trace.diagnostic_ticks[t].state_hash,`recorded tick${t+1}`);
   for(let p=0;p<2;p++){placed[p]+=Number(info.placed[p]);deaths[p]+=Number(info.died[p]);self[p]+=Number(info.ownBombDefeat?.[p]||0);
    if(info.placed[p])replay.events.push({tick:s.t,kind:'bomb',player:p});if(info.died[p])replay.events.push({tick:s.t,kind:'death',player:p});}
   if(s.t%200===0||s.done)replay.checks.push([s.t,await P.hash(P.state(s))]);
  }
  assert.equal(s.done,true);assert.equal(s.winner,expected.winner);assert.equal(s.t,expected.ticks);
  assert.deepEqual(placed,expected.placed);assert.deepEqual(deaths,expected.deaths);assert.deepEqual(self,expected.self_deaths);assert.deepEqual(s.bunStored,expected.stored);
  replay.summary={ticks:s.t,done:s.done,winner:s.winner,reward:[null,null],failures:[]};replay.meta.replay_identity=await P.hash(P.identity(replay));
  P.validate(replay);await new P.Player(replay,level).verify();
  const dest=`${B}/replay/game_${String(index).padStart(3,'0')}`;fs.mkdirSync(dest,{recursive:true});fs.writeFileSync(dest+'/review.json',JSON.stringify(replay));
  fs.writeFileSync(dest+'/MATCH.json',JSON.stringify({status:'MATCH',index,review_verified:true,review_sha256:sha(fs.readFileSync(dest+'/review.json')),
    source_trace_sha256:replay.meta.source_trace_sha256,all_recorded_tick_hashes:true,terminal_and_placements_deaths:true}));
  games.push({index,seed:expected.seed,seat:expected.seat,rule:label+' · 固定hard首局，非按结果挑选',expected,model:m});index++;
 }
 const manifest={status:'VERIFIED_FIXED_FIRST_GAMES',model:models.baseline,games,spawn_manifest:cfg.conditions.hunter_hard.spawn_manifest,
  model_label:'bun-native-v2 · 固定候选与网页基线',html:cfg.spawn_side_version?'20261031-v2-ownside-candidates-review.html':'20261031-v2-candidates-review.html',teacher:false,rule_version:'bun-native-v2',spawn_side_version:cfg.spawn_side_version,
  limitations:['fixed first hard game per model, all P0; not outcome-selected and not P1 capability proof','reward not recorded','Sim.step only; not human frameStep']};
 fs.writeFileSync(B+'/replay/MANIFEST.json',JSON.stringify(manifest,null,2));console.log('six fixed first-game full Review PASS');
})().catch(e=>{console.error(e);process.exitCode=1;});
