'use strict';
// Passive observer around the exact frozen evaluation entry point. Does not
// change actions, random streams, masks, simulator rules, or model inference.
const fs=require('fs'),crypto=require('crypto'),assert=require('assert');
const Q=require('../web/sim-stock-window-v1'),P=require('../web/review_protocol');
const A=require('./eval_web_hunter');
const [request,offsetText,destination]=process.argv.slice(2),offset=Number(offsetText);
const req=JSON.parse(fs.readFileSync(request)),split=req.split||'dev';
const spawnFile=`runs/bomb_threat_v1/${split}_spawns.json`,spawn=JSON.parse(fs.readFileSync(spawnFile));
const model=`runs/bomb_threat_v1/eval/${req.name}/model/model.json`,modelBytes=fs.readFileSync(model),modelDoc=JSON.parse(modelBytes);
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const targetSeed=spawn.seed+offset,original=Q.Sim.prototype.step;
let replay,initial,simRef;const trace=[],counts={requests:[0,0],placed:[0,0],deaths:[0,0],rejects:[{},{}],cells:[new Set(),new Set()]};
function component(s,p,ignoreBricks=false){
 const [r,c]=s.centerCell(p),start=r*15+c,seen=new Set([start]),queue=[start];
 for(let q=0;q<queue.length;q++){const i=queue[q],rr=Math.floor(i/15),cc=i%15;for(const[dr,dc]of Q.DIRS.slice(0,4)){
  const y=rr+dr,x=cc+dc,j=y*15+x;if(y>=0&&y<13&&x>=0&&x<15&&!s.wall[j]&&(ignoreBricks||!s.brick[j])&&!seen.has(j)){seen.add(j);queue.push(j);}
 }}return [...seen].sort((a,b)=>a-b);
}
Q.Sim.prototype.step=function(actions){
 if(this.__btIdentity?.seed!==targetSeed)return original.call(this,actions);
 if(!initial){initial=P.state(this);simRef=this;replay={actions:[],checks:[[0,sha(JSON.stringify(initial))]],events:[]};}
 const tick=this.t,before={pos:Array.from(this.pos),alive:Array.from(this.alive),trapped:Array.from(this.trapped),carried:Array.from(this.bunCarried)};
 const reasons=actions.map((a,p)=>{
  const [r,c]=this.centerCell(p),i=this._actionCell(a[4],r*15+c),why=[];
  if(!this.alive[p])why.push('dead');if(this.fuse[i]>1)why.push('existing_bomb_after_fuse_decrement');
  if(this.trapped[p]>0)why.push('native_sugar');if(this.brick[i])why.push('brick');
  if(this.isBun&&this.bunCarried[p]>=0)why.push('carrying');if(this.liveBombs(p)>=this.bombsCap[p])why.push('capacity_at_predecrement');
  return {requested:a[1]===1,cell:i,why,fuse:this.fuse[i],cap:this.bombsCap[p],live:this.liveBombs(p),mask:this.legalMask().bm[p]};
 });
 const row=[actions[0][0],actions[0][2]?2:Number(actions[0][1]),actions[1][0],actions[1][2]?2:Number(actions[1][1])];
 replay.actions.push(row);const info=original.call(this,actions);
 for(let p=0;p<2;p++){
  counts.requests[p]+=Number(reasons[p].requested);counts.placed[p]+=Number(info.placed[p]);counts.deaths[p]+=Number(info.died[p]);counts.cells[p].add(this.centerCell(p).join(','));
  if(reasons[p].requested&&!info.placed[p]){assert(reasons[p].why.length,'unexplained rejection');for(const why of reasons[p].why)counts.rejects[p][why]=(counts.rejects[p][why]||0)+1;}
  if(info.placed[p]||info.died[p])replay.events.push({tick:this.t,kind:info.placed[p]?'bomb':'death',player:p});
 }
 trace.push({tick:tick+1,before,actions:row,placement:reasons,actual:Array.from(info.placed),pos:Array.from(this.pos),alive:Array.from(this.alive),trapped:Array.from(this.trapped)});
 if(this.t%200===0||this.done)replay.checks.push([this.t,sha(JSON.stringify(P.state(this)))]);
 return info;
};
fs.mkdirSync(destination,{recursive:true});
process.env.JE10_ROUTE_TASK='natural_full2400';process.env.JE10_ROUTE_GAMES='64';process.env.JE10_ROUTE_START=String(offset);process.env.JE10_ROUTE_END=String(offset+1);process.env.JE10_ROUTE_SEED_START=String(spawn.seed);
process.argv=[process.argv[0],process.argv[1],'greedy',model,`${destination}/evaluation.json`,'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',spawnFile];
require('./bomb_threat_eval');
process.on('beforeExit',async()=>{
 if(!replay||replay.meta)return;
 const level=A.loadLevel(),s=simRef;
 replay.schema=P.SCHEMA;replay.meta={id:`zero-bomb-${req.name}-${targetSeed}`,seed:targetSeed,tick_ms:100,max_ticks:2400,
  scene:'native',native_rules:true,model:req.name,checkpoint:req.checkpoint,checkpoint_sha256:modelDoc.meta.checkpoint_sha256,
  model_sha256:sha(modelBytes),simulator_sha256:sha(fs.readFileSync('web/sim-stock-window-v1.js')),
  opponent_sha256:sha(fs.readFileSync('web/bun_hunter_bot.js')),map_sha256:await P.hash(level),map:806,
  policy:'greedy',teacher:false,opponent:'bun.hunter/normal',reward_status:'unrecorded',training_reward_status:'unrecorded',
  frozen_birth:{schema:'qqt.frozen-birth/v1',cells:spawn.spawn_cells[offset]},
  scene_name:'零放泡结构审计：固定出生完整自然局',scene_rules:'原生带砖；冻结出生；老师关闭；非人工持包'};
 replay.summary={ticks:s.t,done:s.done,winner:s.winner,reward:[null,null],failures:[]};replay.rewards=[];
 replay.meta.replay_identity=await P.hash(P.identity(replay));P.validate(replay);await new P.Player(replay,level).verify();
 // Restore only the captured frame for geometry, never use it for inference.
 const geometry=new Q.Sim(targetSeed);geometry.reset(level,{nativeTrap:true,nativeItems:true});geometry.restoreReplay(initial);
 const components=[component(geometry,0),component(geometry,1)],wallOnly=[component(geometry,0,true),component(geometry,1,true)];
 const summary={seed:targetSeed,offset,split,request,model,policy_sha256:replay.meta.checkpoint_sha256,model_sha256:sha(modelBytes),
  ticks:s.t,done:s.done,winner:s.winner,...counts,cells:counts.cells.map(x=>[...x]),initial_positions:initial.pos,
  open_components:components,wall_only_components:wallOnly,open_connected:components[0].some(i=>components[1].includes(i)),
  wall_only_connected:wallOnly[0].some(i=>wallOnly[1].includes(i)),review_verified:true};
 fs.writeFileSync(`${destination}/review.json`,JSON.stringify(replay));fs.writeFileSync(`${destination}/trace.json`,JSON.stringify(trace));fs.writeFileSync(`${destination}/AUDIT.json`,JSON.stringify(summary,null,2));
 console.log(JSON.stringify({destination,...summary,cells:summary.cells.map(x=>x.length),open_components:components.map(x=>x.length),wall_only_components:wallOnly.map(x=>x.length)}));
});
