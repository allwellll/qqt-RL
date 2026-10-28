'use strict';
// Native real death/drop with randomized legal 2-8 step spawn and orientation.
// No teacher/actions/observations are injected after initialization.
const fs=require('fs'),assert=require('node:assert/strict'),crypto=require('crypto');
const QQT=require('../web/sim.js'),authority=require('./eval_web_hunter.js');
const inference=require('./je05_web_hunter.js'),previous=require('./je08_web_diagnostics.js');
const bridge=require('./je09_web_bridge.js'),residual=require('./eval_web_hunter_residual.js');
const manifest={schema:'je09.randomized-stage-a/v1',seed:process.env.JE09_NAMESPACE?.includes("stage_b_")?2026101616:2026101543,holdout_seed:process.env.JE09_NAMESPACE?.includes("stage_b_")?2026101617:2026101544,games:16,ticks:400,assist:false,distance:[2,8],opponent:'real Hunter normal',fixture:'native trap1 death/drop attributed to learner; not a policy kill',spawn:'seeded legal pair, alternating seats, requested distance 2+game%7, independent orientations',promotion:'pickup >=13/16 plus independent holdout and Hunter/Tactical/carry retention'};
if(process.env.JE09_NAMESPACE?.includes('stage_c_')) {
 const declaration=JSON.parse(fs.readFileSync(process.env.JE09_NAMESPACE+'/ROUND_DECLARATION.json','utf8'));
 manifest.seed=declaration.seeds.stage_a_main;manifest.holdout_seed=declaration.seeds.stage_a_holdout;
 manifest.experiment_id=declaration.experiment_id;
}
function setup(seed,game){
 const team=game%2,enemy=1-team,rng=QQT.mulberry32(seed),s=previous.setup('drop_return',seed,team,null,manifest.ticks);
 const legal=Array.from(s.wall,(x,i)=>!x&&s._bunBaseTeam(Math.floor(i/QQT.W),i%QQT.W)<0);
 const targets=legal.map((x,i)=>x?i:-1).filter(x=>x>=0),target=targets[Math.floor(rng()*targets.length)];
 const d=previous.distance(s.wall,s.brick,[target]),wanted=2+game%7,candidates=[];
 for(let i=0;i<d.length;i++)if(legal[i]&&d[i]===wanted&&Math.hypot(Math.floor(i/QQT.W)-Math.floor(target/QQT.W),i%QQT.W-target%QQT.W)>=1.9)candidates.push(i);
 assert.ok(candidates.length);const start=candidates[Math.floor(rng()*candidates.length)];
 for(const [t,index]of [[team,start],[enemy,target]]){s.pos[t*2]=Math.floor(index/QQT.W)+.5;s.pos[t*2+1]=index%QQT.W+.5;s.lastMoveDir[t]=Math.floor(rng()*4);}
 s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];s._gen++;
 return {s,team,enemy,initial:{start,target,distance:d[start],orientation:s.lastMoveDir.slice()}};
}
async function run(model,out,mode='evaluation'){
 assert.equal(process.env.CUDA_VISIBLE_DEVICES||'','');assert.ok(['evaluation','holdout'].includes(mode));
 const loaded=await inference.loadModel(model),base=mode==='holdout'?manifest.holdout_seed:manifest.seed,episodes=[],begin=Date.now();
 for(let game=0;game<16;game++){
  const seed=(base+game*104729)>>>0,{s,team,enemy,initial}=setup(seed,game),tracker=new bridge.Tracker(team),rng=QQT.mulberry32(seed^0x515154),hunter=authority.createHunter('normal');
  tracker.trapCause[enemy][team]=true;
  hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:'je09-stageA-'+mode+'-'+game,seed,ruleset:'bun',max_ticks:400,metadata:{}});
  async function step(settling){
   const before=previous.snapshot(s),raw=await loaded.model.act(s,team,rng),bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng),actions=[];
   actions[team]=[raw[0],settling?0:Number(raw[1]),settling?0:Number(raw[2]||0)];actions[enemy]=[bot.move,settling?0:Number(bot.ability===1),settling?0:Number(bot.ability===2)];
   if(settling)s.done=false;tracker.observe(s,before,s.step(actions));
  }
  while(!s.done&&s.t<400)await step(false);
  const primary=s.t;s.maxSteps=primary+125;while(residual.hasHazards(s)&&s.t<primary+124)await step(true);assert.ok(!residual.hasHazards(s));
  episodes.push({game,seed,team,initial,...tracker.result(),primary_ticks:primary});
 }
 const data={passed:true,assist:false,mode,manifest,model_json_sha256:loaded.sha,results:bridge.summarize(episodes),wall_seconds:(Date.now()-begin)/1000,scope:'controlled-drop first-pickup capability only; not full natural chain'};
 fs.writeFileSync(out+'.tmp',JSON.stringify(data));fs.renameSync(out+'.tmp',out);return data;
}
if(require.main===module)run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={manifest,setup,run};
