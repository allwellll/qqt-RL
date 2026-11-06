'use strict';
// Independent frozen-policy matches. No production source patches, rule edits,
// privileged observations, search, or reward. Both policies see the same tick.
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const Q=require('./native_eval_simulator_v1'),S=require('../web/five-item-slots-v1');
const {loadModel}=require('./interceptor_primary_inference_v1');
const Source=require('./combat_specialist_g7_source_metrics'),Continuous=require('./combat_regions_metrics_v2');
const R3=require('./r3_items_metrics'),Custody=require('./interceptor_primary_chain_metrics_v1');
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const fileHash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
const cell=(s,p)=>Math.floor(s.pos[p*2])*15+Math.floor(s.pos[p*2+1]);
const inCorner=i=>Math.floor(i/15)>=9&&i%15>=11;
function neighbors(i){const r=Math.floor(i/15),c=i%15;return [r>0?i-15:-1,r<12?i+15:-1,c>0?i-1:-1,c<14?i+1:-1].filter(x=>x>=0);}
function exits(s,start,extra=-1,bubbles=true){
 const seen=new Set([start]),queue=[start],gates=new Set();let distance=null;
 const dist=new Map([[start,0]]);
 for(let k=0;k<queue.length;k++){
  const i=queue[k];if(!inCorner(i)){gates.add(i);distance??=dist.get(i);continue;}
  for(const n of neighbors(i))if(!seen.has(n)&&!s.wall[n]&&!s.brick[n]&&!s.pushable[n]&&n!==extra&&(!bubbles||s.fuse[n]<=1)){
   seen.add(n);queue.push(n);dist.set(n,dist.get(i)+1);
  }
 }
 return {gates:gates.size,distance,reachable:queue.length};
}
function setup(game,level){
 const s=new Q.Sim(game.seed);s.reset(level,{nativeTrap:true,nativeItems:true});s.maxSteps=2400;
 for(let p=0;p<2;p++){s.pos[2*p]=game.spawn[p][0]+.5;s.pos[2*p+1]=game.spawn[p][1]+.5;}
 s.bunSpawnPos=game.native_respawn.map(a=>a.map(x=>x+.5));s._gen++;
 return s;
}
function threat(s,p){
 const q=1-p,manhattan=Math.abs(s.pos[2*p]-s.pos[2*q])+Math.abs(s.pos[2*p+1]-s.pos[2*q+1]);
 const homes=[];const [r,c]=s.bunBases[s.team[p]];
 for(let y=r;y<r+3;y++)for(let x=c;x<c+3;x++)homes.push(y*15+x);
 const home=R3.field(s,homes).distance,own=home[cell(s,p)],enemy=home[cell(s,q)];
 return {enemy_alive:Boolean(s.alive[q]),enemy_controlled:s.trapped[q]>0,distance:manhattan,
  route_intercept_proxy:s.alive[q]&&s.trapped[q]===0&&enemy<1e6&&own<1e6&&enemy<=own+2,
  band:!s.alive[q]?'enemy_dead':s.trapped[q]>0?'enemy_controlled':manhattan>6?'alive_far':enemy<=own+2?'alive_near_intercept':'alive_near_other'};
}
async function main(){
 const cfg=JSON.parse(fs.readFileSync(process.argv[2]));
 const left=await loadModel(cfg.model_json),right=await loadModel(cfg.opponent_model_json);
 assert([3,7].includes(left.meta.ability_actions));assert([3,7].includes(right.meta.ability_actions));
 Source.install(Q);R3.install(Q);Continuous.install(Q);
 const levels=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')),results=[];
 fs.mkdirSync(cfg.output,{recursive:true});
 for(const game of cfg.games){
  const id=`match-${game.seed}-P${game.seat}-${game.mode}-${game.opponent_mode}`,dir=cfg.output+'/'+id;
  if(fs.existsSync(dir+'/RESULT.json')){
   const v=JSON.parse(fs.readFileSync(dir+'/RESULT.json'));assert.equal(v.model_json_sha256,left.sha);assert.equal(v.opponent_model_json_sha256,right.sha);
   assert.equal(fileHash(dir+'/review.json'),v.review_sha256);results.push(v);continue;
  }
  const level=game.level||levels.find(x=>x.qqt_id===(game.map_id||806)),s=setup(game,level),p=game.seat,q=1-p;
  assert(s.alive.every(Boolean));assert(game.spawn.every(([r,c])=>!s.wall[r*15+c]&&!s.brick[r*15+c]&&!s.pushable[r*15+c]));
  s.__btIdentity=true;R3.exportMetrics(s);Source.attach(s,{learner:p});Continuous.attach(s,p);
  left.model.greedy=game.mode==='greedy';right.model.greedy=game.opponent_mode==='greedy';
  const lr=rng(game.action_seed),rr=rng(game.action_seed^0x5CA17),actions=[],checks=[[0,hash(s.snapshotReplay())]],frames=[],pressure=[],controlPositions=[],pickups=[];
  const occupancy=[0,0],aliveTicks=[0,0],blocked=[null,null],escapes=[],lastPlaced=[-10000,-10000];
  const frame=()=>frames.push({t:s.t,pos:Array.from(s.pos),alive:Array.from(s.alive),trapped:Array.from(s.trapped),carried:Array.from(s.bunCarried),wall:Array.from(s.wall),brick:Array.from(s.brick),fuse:Array.from(s.fuse),fire:Array.from(s.lastReplayCovered||Array(195).fill(0)),done:s.done,winner:s.winner});frame();
  while(!s.done){
   const tick=s.t,prePos=Array.from(s.pos),preTrap=Array.from(s.trapped),preCarry=Array.from(s.bunCarried),beforeCells=[cell(s,0),cell(s,1)];
   const contexts=[threat(s,0),threat(s,1)];
   const own=await left.model.act(s,p,lr),enemy=await right.model.act(s,q,rr),dec=[];dec[p]=left.model.lastDecision;dec[q]=right.model.lastDecision;
   const joint=[];joint[p]=own;joint[q]=enemy;
   const exitBefore=inCorner(beforeCells[q])?exits(s,beforeCells[q]):null;
   const marginal=exitBefore&&Continuous.legalCapacity(s,p)&&dec[p].ability===1?exits(s,beforeCells[q],beforeCells[p]):null;
   const offset=Array.from(s.fuse).some((f,i)=>f>1&&s.owner[i]===p&&f<Q.CFG.fuse-2);
   for(let k=0;k<2;k++){
    const corner=inCorner(beforeCells[k]);aliveTicks[k]+=Number(s.alive[k]);occupancy[k]+=Number(s.alive[k]&&corner);
    if(corner&&s.alive[k]&&!s.trapped[k]&&!blocked[k]&&exits(s,beforeCells[k],-1,false).gates>0&&exits(s,beforeCells[k]).gates===0)blocked[k]={player:k,start_tick:tick,start_cell:beforeCells[k]};
    if(blocked[k]&&(!corner||!s.alive[k])){escapes.push({...blocked[k],end_tick:tick,outcome:!s.alive[k]?'death':'left_corner_alive',carrying:preCarry[k]>=0});blocked[k]=null;}
   }
   actions.push([dec[0].move,dec[0].ability,dec[1].move,dec[1].ability]);
   const info=s.step(joint);
   if(info.placed[p]&&exitBefore)pressure.push({tick:s.t,target_cell:beforeCells[q],own_cell:beforeCells[p],exit_gates_before:exitBefore.gates,
    exit_gates_after_cell:marginal?.gates??null,exit_distance_before:exitBefore.distance,exit_distance_after_cell:marginal?.distance??null,
    exit_cut_proxy:!!marginal&&marginal.gates<exitBefore.gates,reachable_cut_proxy:!!marginal&&marginal.reachable<exitBefore.reachable,
    staggered_live_fuses:offset,actual_gap:s.t-lastPlaced[p]});
   for(let k=0;k<2;k++){
    if(info.placed[k])lastPlaced[k]=s.t;
    if(preTrap[k]===0&&s.trapped[k]>0)controlPositions.push({tick:s.t,victim:k,cell:beforeCells[k],corner:inCorner(beforeCells[k]),pos:prePos.slice(2*k,2*k+2)});
    if(preCarry[k]<0&&s.bunCarried[k]>=0)pickups.push({tick:s.t,player:k,origin:s.bunCarried[k],...contexts[k]});
   }
   if(s.t%200===0||s.done)checks.push([s.t,hash(s.snapshotReplay())]);if(s.t%10===0||s.done)frame();
  }
  const source=Source.exportMetrics(s),continuous=Continuous.result(s),r3=R3.exportMetrics(s);
  const cornerControls=source.controls.filter(e=>e.victim===q&&e.causal_kind==='unique_other'&&controlPositions.some(x=>x.tick===e.tick&&x.victim===q&&x.corner));
  const cornerIds=new Set(cornerControls.map(e=>e.id)),cornerKills=source.deaths.filter(e=>e.victim===q&&e.uninterrupted_control&&cornerIds.has(e.initial_control?.id));
  const pressureControls=cornerControls.map(c=>({control_id:c.id,tick:c.tick,witnesses:pressure.filter(e=>e.tick<=c.tick&&c.tick-e.tick<=60&&(e.exit_cut_proxy||e.reachable_cut_proxy||e.staggered_live_fuses))}));
  const custodies=[Custody.summarize(source,r3,0),Custody.summarize(source,r3,1)];
  for(const e of pickups)e.custody=custodies[e.player].custodies.find(c=>c.start_tick===e.tick)||null;
  const replay={schema:'resume-gpu67-native-replay-v1',meta:{seed:game.seed,game,model:cfg.candidate,opponent:cfg.target,learner_seat:p,
   model_json_sha256:left.sha,opponent_model_json_sha256:right.sha,level_sha256:hash(level),simulator_sha256:Q.SIMULATOR_SHA256,rule:'bun-native-v2',attribute_overrides:false},level,actions,checks,summary:{ticks:s.t,done:s.done,winner:s.winner}};
  // Replay in an independent fresh simulator; no inference or observer state.
  const fresh=setup(game,level),expected=new Map(checks);assert.equal(hash(fresh.snapshotReplay()),expected.get(0));
  for(const a of actions){fresh.step([S.engineAction(a[0],a[1]),S.engineAction(a[2],a[3])]);if(expected.has(fresh.t))assert.equal(hash(fresh.snapshotReplay()),expected.get(fresh.t),`fresh replay @${fresh.t}`);}
  assert(fresh.done);assert.equal(fresh.winner,s.winner);
  fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(dir+'/review.json',JSON.stringify(replay));fs.writeFileSync(dir+'/FRAMES.json',JSON.stringify({meta:replay.meta,frames}));
  const result={status:'PASS',candidate:cfg.candidate,target:cfg.target,seed:game.seed,seat:p,mode:game.mode,opponent_mode:game.opponent_mode,start_region:game.region,
   map_id:game.map_id||806,brick_stage:game.brick_stage||'native_complete',initial_bricks:level.brick.filter(Boolean).length,ticks:s.t,winner:s.winner,complete:true,review_verified:true,review_sha256:fileHash(dir+'/review.json'),
   model_json_sha256:left.sha,opponent_model_json_sha256:right.sha,continuous_attack:continuous,source_metrics:source,r3,primary_custody:custodies[p],both_custodies:custodies,
   corner:{alive_ticks:aliveTicks,occupancy_ticks:occupancy,controls:cornerControls.length,kills:cornerKills.length,pressure,pressure_controls:pressureControls,blocked_episodes:[...escapes,...blocked.filter(Boolean).map(e=>({...e,outcome:'censored_terminal',end_tick:s.t}))]},
   pickups,onnx_json_max_error:[left.model.forwardMaxError,right.model.forwardMaxError],
   limitations:['exit and staggered placements are geometry/timing proxies; no intention claim','pressure to subsequent strict control/kill is temporal association, not per-bubble treatment effect','small frozen screen is not a capability or transport-safety proof']};
  fs.writeFileSync(dir+'/RESULT.json',JSON.stringify(result));results.push(result);console.log(JSON.stringify({id,candidate:cfg.candidate,target:cfg.target,winner:s.winner,corner_controls:cornerControls.length,corner_kills:cornerKills.length,ticks:s.t}));
 }
 fs.writeFileSync(cfg.output+'/COMPLETE.json',JSON.stringify({status:'PASS',results},null,2));
}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
module.exports={setup,threat,exits};
