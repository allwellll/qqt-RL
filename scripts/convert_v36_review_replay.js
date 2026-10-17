'use strict';
const fs=require('node:fs');
const crypto=require('node:crypto');
const Q=require('../web/sim');
const P=require('../web/review_protocol');
const Arena=require('../web/tactical_arena');
const authority=require('./eval_web_hunter');

const sourcePath='web/replays/bun392-je09-route-pbrs-continuous-v36-g32-seed-2026101736.json';
const modelPath='runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02/authority_je09_v1/bun392_bridge_pfsp8_g000032/model/model.json';
const outputPath='web/review-replays/h8-route-v36-g32-2026101736.json';
const id='h8-route-v36-g32-2026101736';
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');

async function main(){
  const source=JSON.parse(fs.readFileSync(sourcePath));
  const modelBytes=fs.readFileSync(modelPath),model=JSON.parse(modelBytes);
  if(source.meta.checkpoint_sha256!==model.meta.checkpoint_sha256)throw Error('source/checkpoint hash mismatch');
  if(sha(modelBytes)!==source.meta.model_json_sha256)throw Error('source/model JSON hash mismatch');
  if(source.meta.seed!==2026101736||source.actions.length!==300)throw Error('unexpected fixed-seed source trajectory');
  if(source.actions.some(row=>row[2]||row[5]))throw Error('review schema cannot represent item actions');

  const level=Arena.buildTacticalArena(authority.loadLevel());
  const sim=new Q.Sim(source.meta.seed);
  sim.reset(level);
  const actions=source.actions.map(row=>[row[0],row[1],row[3],row[4]]);
  const checks=[[0,await P.hash(P.state(sim))]];
  for(const row of actions){
    sim.step([[row[0],Number(row[1]===1),Number(row[1]===2)],[row[2],Number(row[3]===1),Number(row[3]===2)]]);
    if(sim.t%50===0||sim.t===actions.length)checks.push([sim.t,await P.hash(P.state(sim))]);
  }
  const levelHash=await P.hash(level);
  const meta={id,episode_id:source.meta.id,seed:source.meta.seed,scene:'native',scene_name:'自然规则对局片段（战术清砖图，300 tick 截断）',
    scene_rules:'固定 seed production Sim 对局片段；Tactical Arena 清除可破坏砖，无道具；原生模拟器规则配置；300 tick 截断，非完整终局',
    arena:'tactical_v2',native_rules:false,truncated:true,variant:'native',tick_ms:100,max_ticks:actions.length,
    model:'H8 v36 连续路线 PBRS g32',checkpoint:'H8/v36/g32/raw Actor',checkpoint_sha256:model.meta.checkpoint_sha256,
    model_sha256:source.meta.model_json_sha256,simulator_sha256:sha(fs.readFileSync('web/sim.js')),
    opponent:'bun.tactical_v2',opponent_sha256:sha(fs.readFileSync('web/bun_rule_bot.js')),
    map:'806',map_sha256:await P.hash(authority.loadLevel()),arena_map_sha256:levelHash,policy:'greedy',backend:'production JavaScript TransformerModel / Tactical v2',
    reward_status:'unrecorded',outcome_status:'未胜、未交付；片段结束时仍未终局',advantage_status:'未记录',advantages_status:'unrecorded',
    source_replay_sha256:sha(fs.readFileSync(sourcePath)),checkpoint_model_json_sha256:sha(modelBytes),
    source_hashes:{'web/sim.js':sha(fs.readFileSync('web/sim.js')),'web/tactical_arena.js':sha(fs.readFileSync('web/tactical_arena.js')),
      'web/bun_rule_bot.js':sha(fs.readFileSync('web/bun_rule_bot.js')),'web/assets/maps/levels.json':sha(fs.readFileSync('web/assets/maps/levels.json')),
      [sourcePath]:sha(fs.readFileSync(sourcePath))}};
  const doc={schema:P.SCHEMA,meta,actions,summary:{ticks:actions.length,done:false,winner:null,score:sim.bunScore.slice(),
    stored:sim.bunStored.map(row=>row.slice()),reward:[null,null]},rewards:[],events:[],checks};
  meta.replay_identity=await P.hash(P.identity(doc));
  P.validate(doc);
  await new P.Player(doc,level).verify();
  fs.writeFileSync(outputPath,JSON.stringify(doc));
  const bytes=fs.readFileSync(outputPath);
  const catalogPath='web/review-catalog.json',catalog=JSON.parse(fs.readFileSync(catalogPath));
  catalog.replays=catalog.replays.filter(row=>row.id!==id);
  catalog.replays.push({id,name:'H8 v36 连续路线 PBRS g32',file:outputPath.replace('web/review-replays/',''),seed:meta.seed,variant:'native',scene:'native',
    scene_name:meta.scene_name,scene_rules:meta.scene_rules,model:meta.model,opponent:meta.opponent,checkpoint_sha256:meta.checkpoint_sha256,
    map:meta.map,map_sha256:meta.map_sha256,diagnostic:true,summary:doc.summary,sha256:sha(bytes),bytes:bytes.length,
    gzip_bytes:require('node:zlib').gzipSync(bytes,{level:9}).length});
  catalog.raw_bytes=catalog.replays.reduce((n,row)=>n+row.bytes,0);
  catalog.gzip_bytes=catalog.replays.reduce((n,row)=>n+row.gzip_bytes,0);
  catalog.screened_episodes=catalog.replays.length;
  catalog.model='H8 v36 连续路线 PBRS g32；含既有 H8 v35/v34 对照';
  fs.writeFileSync(catalogPath,JSON.stringify(catalog,null,2)+'\n');
  console.log(JSON.stringify({id,file:outputPath,sha256:sha(bytes),bytes:bytes.length,ticks:actions.length,done:doc.summary.done,
    winner:doc.summary.winner,score:doc.summary.score,stored:doc.summary.stored,checkpoint_sha256:meta.checkpoint_sha256,
    model_json_sha256:meta.model_sha256,source_replay_sha256:meta.source_replay_sha256,map_sha256:levelHash,
    checks:checks.length,deterministic:true,advantages:'unrecorded',route:'not present'}));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
