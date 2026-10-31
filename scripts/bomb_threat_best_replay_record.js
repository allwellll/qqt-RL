'use strict';
const fs=require('fs'),crypto=require('crypto'),assert=require('assert');
process.env.QQT_STOCK_RULE='stock-window-v1';process.env.BT_FRESH_TIER='hard';
const Q=require('../web/sim-stock-window-v1'),P=require('../web/review_protocol'),A=require('./eval_web_hunter');
const base=process.env.BT_REPLAY_BASE||'runs/bomb_threat_v1/replay_best_u800',manifest=JSON.parse(fs.readFileSync(base+'/MANIFEST.json'));
const index=Number(process.argv[2]),entry=manifest.games.find(e=>e.index===index),destination=`${base}/game_${String(index).padStart(3,'0')}`;
assert(entry);const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const model=manifest.model.model_json,modelBytes=fs.readFileSync(model),spawnFile='runs/bomb_threat_v1/fresh256_slot01_u800/hard_spawns.json',spawn=JSON.parse(fs.readFileSync(spawnFile));
assert.equal(sha(modelBytes),manifest.model.model_sha256);
let replay,simRef;const original=Q.Sim.prototype.step;
Q.Sim.prototype.step=function(actions){
 if(this.__btIdentity?.seed!==entry.seed)return original.call(this,actions);
 if(!replay){simRef=this;replay={actions:[],checks:[[0,sha(JSON.stringify(P.state(this)))]],events:[]};}
 replay.actions.push([actions[0][0],actions[0][2]?2:Number(actions[0][1]),actions[1][0],actions[1][2]?2:Number(actions[1][1])]);
 const info=original.call(this,actions);
 for(let p=0;p<2;p++){if(info.placed[p])replay.events.push({tick:this.t,kind:'bomb',player:p});if(info.died[p])replay.events.push({tick:this.t,kind:'death',player:p});}
 if(this.t%200===0||this.done)replay.checks.push([this.t,sha(JSON.stringify(P.state(this)))]);
 return info;
};
fs.mkdirSync(destination,{recursive:true});
process.env.JE10_ROUTE_TASK='natural_full2400';process.env.JE10_ROUTE_GAMES='256';process.env.JE10_ROUTE_START=String(index);process.env.JE10_ROUTE_END=String(index+1);process.env.JE10_ROUTE_SEED_START=String(spawn.seed);
process.argv=[process.argv[0],process.argv[1],'greedy',model,destination+'/evaluation.json','runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',spawnFile];
require('./bomb_threat_fresh256_eval');
process.on('beforeExit',async()=>{
 if(!replay||replay.meta)return;
 const level=A.loadLevel(),s=simRef,actual=s.__bt,expected=entry.expected;
 const fields=['winner','placed','kills','causal_kills','deaths','ticks','done','stored'];
 const comparison=Object.fromEntries(fields.map(k=>[k,JSON.stringify(actual[k])===JSON.stringify(expected[k])]));
 const matched=Object.values(comparison).every(Boolean);
 replay.schema=P.SCHEMA;replay.meta={id:`best-u800-hard-${entry.seed}`,seed:entry.seed,tick_ms:100,max_ticks:2400,
  scene:'native',native_rules:true,model:'A3 u800 候选（B fresh256 FAIL，未晋升）',checkpoint:manifest.model.checkpoint,
  checkpoint_sha256:manifest.model.policy_sha256,model_sha256:sha(modelBytes),simulator_sha256:sha(fs.readFileSync('web/sim-stock-window-v1.js')),
  opponent_sha256:sha(fs.readFileSync('web/bun_hunter_bot.js')),map_sha256:await P.hash(level),map:806,policy:'greedy',teacher:false,
  opponent:'production JS Hunter/hard',learner_seat:entry.seat,reward_status:'unrecorded',training_reward_status:'unrecorded',
  frozen_birth:{schema:'qqt.frozen-birth/v1',cells:spawn.spawn_cells[index%spawn.spawn_cells.length]},scene_name:'顺序固定自然完整局',scene_rules:'原生带砖stock-window-v1，老师关闭，无人工持包',recorded_at:new Date().toISOString()};
 replay.summary={ticks:s.t,done:s.done,winner:s.winner,reward:[null,null],failures:[]};replay.rewards=[];
 replay.meta.replay_identity=await P.hash(P.identity(replay));P.validate(replay);await new P.Player(replay,level).verify();
 fs.writeFileSync(destination+'/review.json',JSON.stringify(replay));fs.writeFileSync(destination+'/MATCH.json',JSON.stringify({status:matched?'MATCH':'REPLAY_MISMATCH',comparison,index,seed:entry.seed,seat:entry.seat,expected,actual,review_verified:true,review_sha256:sha(fs.readFileSync(destination+'/review.json'))},null,2));
 console.log(JSON.stringify({index,status:matched?'MATCH':'REPLAY_MISMATCH',ticks:s.t,comparison}));
});
