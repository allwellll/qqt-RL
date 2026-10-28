'use strict';
// Native real death/drop with randomized legal 2-8 step spawn and orientation.
// No teacher/actions/observations are injected after initialization.
const fs=require('fs'),assert=require('node:assert/strict'),crypto=require('crypto');
const QQT=require('../web/sim.js'),authority=require('./eval_web_hunter.js');
const inference=require('./je05_web_hunter.js'),previous=require('./je08_web_diagnostics.js');
const bridge=require('./je09_web_bridge.js'),residual=require('./eval_web_hunter_residual.js');
const manifest={schema:'je09.real-drop-stage-b/v1',seed:2026101613,holdout_seed:2026101614,games:16,ticks:400,assist:false,distance:[2,8],opponent:'real Hunter normal',fixture:'native trap1 death/drop attributed to learner; not a policy kill',spawn:'seeded legal pair, alternating seats, requested distance 2+game%7, independent orientations',pickup_return_window:120, target:'outside own topology half; legal return distance <=24', promotion:'pickup>=13,25%>=12,home-cross>=8,capture>=4 of16; independent holdout plus retention'};
if(process.env.JE09_NAMESPACE) {
 const declaration=JSON.parse(fs.readFileSync(process.env.JE09_NAMESPACE+'/ROUND_DECLARATION.json','utf8'));
 manifest.seed=declaration.seeds.stage_b_main;manifest.holdout_seed=declaration.seeds.stage_b_holdout;
 manifest.experiment_id=declaration.experiment_id;
}
function setup(seed,game){
 const team=game%2,enemy=1-team,rng=QQT.mulberry32(seed),s=previous.setup('drop_return',seed,team,null,manifest.ticks);
 const legal=Array.from(s.wall,(x,i)=>!x&&s._bunBaseTeam(Math.floor(i/QQT.W),i%QQT.W)<0);
 const homeMaps=[0,1].map(t=>{const goals=[],base=s.bunBases[t];for(let r=base[0];r<base[0]+3;r++)for(let c=base[1];c<base[1]+3;c++)goals.push(r*QQT.W+c);return previous.distance(s.wall,s.brick,goals);});
 const targets=legal.map((x,i)=>x&&homeMaps[team][i]<=24&&homeMaps[team][i]>homeMaps[enemy][i]?i:-1).filter(x=>x>=0),originalTarget=targets[Math.floor(rng()*targets.length)];
 let target=originalTarget;
 const wanted=2+game%7;
 function starts(t){const d=previous.distance(s.wall,s.brick,[t]);return {d,candidates:legal.map((ok,i)=>ok&&d[i]===wanted&&Math.hypot(Math.floor(i/QQT.W)-Math.floor(t/QQT.W),i%QQT.W-t%QQT.W)>=1.9?i:-1).filter(i=>i>=0)};}
 let {d,candidates}=starts(target);
 if(!candidates.length){const offset=targets.indexOf(target);for(let j=1;j<targets.length;j++){target=targets[(offset+j)%targets.length];({d,candidates}=starts(target));if(candidates.length)break;}}
 assert.ok(candidates.length);const start=candidates[Math.floor(rng()*candidates.length)];
 for(const [t,index]of [[team,start],[enemy,target]]){s.pos[t*2]=Math.floor(index/QQT.W)+.5;s.pos[t*2+1]=index%QQT.W+.5;s.lastMoveDir[t]=Math.floor(rng()*4);}
 s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];s._gen++;
 return {s,team,enemy,...(target!==originalTarget?{fixture_repair:{original_target_no_legal_start:true,original_target:originalTarget,replacement_target:target,seed_unchanged:true}}:{}),initial:{start,target,distance:d[start],orientation:s.lastMoveDir.slice(),return_distance:homeMaps[team][target],enemy_distance:homeMaps[enemy][target]}};
}
class Tracker extends bridge.Tracker {
 constructor(team){super(team);this.firstHome=null;this.returnPath=[];}
 observe(s,b,info){
  const old=this.first;super.observe(s,b,info);
  if(this.first!==null&&!this.broken&&s.t-this.first<=120&&s.bunCarried[this.team]===1-this.team){
   const cell=s.centerCell(this.team),index=cell[0]*QQT.W+cell[1];
   this.returnPath.push({tick:s.t,cell:index,distance:this.maps[this.team][index]});
  }
  if(old===null&&this.first!==null){const cell=s.centerCell(this.team),i=cell[0]*QQT.W+cell[1];this.firstHome=this.maps[this.team][i]<=this.maps[1-this.team][i];}
 }
 result(){
  const r=super.result(),within=e=>this.first!==null&&e.tick>this.first&&e.tick-this.first<=120;
  const has=event=>this.trace.some(e=>e.event===event&&within(e));
  const progress25=has('progress25'),progress50=has('progress50');
  let reverse=0,downhill=0;const seen=new Set();
  for(let i=0;i<this.returnPath.length;i++){const p=this.returnPath[i];seen.add(p.cell);if(i&&p.cell!==this.returnPath[i-1].cell){const delta=p.distance-this.returnPath[i-1].distance;if(delta>0)reverse++;if(delta<0)downhill++;}}
  return {...r,return_path:this.returnPath,reverse_cell_transitions:reverse,downhill_cell_transitions:downhill,unique_return_cells:seen.size,legacy_marks:{progress25:r.progress25,progress50:r.progress50,home_half:r.home_half},
   pickup_return:has('pickup_return'),progress25,progress50,
   home_half:!this.firstHome&&has('home_half')&&has('pickup_return'),first_home:this.firstHome,
   earliest:!r.eligible_drop?'eligible_drop':!r.toward_drop?'toward_drop':!r.arrival?'arrival':!r.first_pickup?'first_pickup':!has('pickup_return')?'pickup_return':!progress25?'progress25':(!this.firstHome&&has('home_half')&&has('pickup_return'))?(!r.capture?'capture':'passed'):'home_half'};
 }
}
async function run(model,out,mode='evaluation'){
 assert.equal(process.env.CUDA_VISIBLE_DEVICES||'','');assert.ok(['evaluation','holdout','confirmation'].includes(mode));
 const loaded=await inference.loadModel(model),base=mode==='holdout'?manifest.holdout_seed:mode==='confirmation'?2026101615:manifest.seed,episodes=[],begin=Date.now();
 for(let game=0;game<16;game++){
  const seed=(base+game*104729)>>>0,{s,team,enemy,initial,fixture_repair}=setup(seed,game),tracker=new Tracker(team),rng=QQT.mulberry32(seed^0x515154),hunter=authority.createHunter('normal');
  tracker.trapCause[enemy][team]=true;
  hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:'je09-stageA-'+mode+'-'+game,seed,ruleset:'bun',max_ticks:400,metadata:{}});
  async function step(settling){
   const before=previous.snapshot(s),raw=await loaded.model.act(s,team,rng),bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng),actions=[];
   actions[team]=[raw[0],settling?0:Number(raw[1]),settling?0:Number(raw[2]||0)];actions[enemy]=[bot.move,settling?0:Number(bot.ability===1),settling?0:Number(bot.ability===2)];
   if(settling)s.done=false;tracker.observe(s,before,s.step(actions));
  }
  while(!s.done&&s.t<400)await step(false);
  const primary=s.t;s.maxSteps=primary+125;while(residual.hasHazards(s)&&s.t<primary+124)await step(true);assert.ok(!residual.hasHazards(s));
  episodes.push({game,seed,team,initial,...(fixture_repair?{fixture_repair}:{}),...tracker.result(),primary_ticks:primary});
 }
 const data={passed:true,assist:false,mode,manifest,model_json_sha256:loaded.sha,results:bridge.summarize(episodes),wall_seconds:(Date.now()-begin)/1000,fixture_totalization:'original valid targets exact; undefined target has deterministic cyclic next valid target, same seed and constraints; fixed32 unchanged',scope:'same episode native controlled drop -> pursuit -> first pickup -> finite-window return; not policy kill or full natural chain'};
 fs.writeFileSync(out+'.tmp',JSON.stringify(data));fs.renameSync(out+'.tmp',out);return data;
}
if(require.main===module)run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={manifest,setup,Tracker,run};
