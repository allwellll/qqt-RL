'use strict';
const fs=require('fs'),Q=require('../web/sim.js'),inf=require('./je05_web_hunter.js'),p=require('./je08_web_diagnostics.js'),authority=require('./eval_web_hunter.js');
async function run(model,out,regions){
 const loaded=await inf.loadModel(model),table=JSON.parse(fs.readFileSync(regions))[1],results={};
 for(const task of ['hunter_easy','hunter_normal','hunter_hard','carry_near','carry_mid','carry_far']){
  const episodes=[];
  for(let game=0;game<4;game++){
   const seed=(2411000000+game*104729)>>>0,team=game%2,enemy=1-team,rng=Q.mulberry32(seed);
   let s;
   if(task.startsWith('carry'))s=p.setup(task,seed,team,table,300);
   else{s=new Q.Sim(seed);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});s.maxSteps=300;}
   const hunter=authority.createHunter(task.startsWith('hunter')?task.split('_')[1]:'normal');hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:task+'-'+game,seed,ruleset:'bun',max_ticks:300,metadata:{}});
   let kills=0,deaths=0,capture=0,bombs=0;
   while(!s.done&&s.t<300){const raw=await loaded.model.act(s,team,rng),b=p.snapshot(s),bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng),a=[];a[team]=raw;a[enemy]=[bot.move,Number(bot.ability===1),Number(bot.ability===2)];const i=s.step(a);kills+=Number(i.causalKill?.[team]||i.creditedKill?.[team]);deaths+=Number(i.died[team]);bombs+=Number(i.placed[team]);capture+=Number(s.bunStored[team][enemy]>b.stock[team][enemy]);}
   episodes.push({game,seed,kills,deaths,capture,bombs});
  }
  results[task]={games:4,kills:episodes.reduce((a,x)=>a+x.kills,0),deaths:episodes.reduce((a,x)=>a+x.deaths,0),capture:episodes.filter(x=>x.capture).length,bombs:episodes.reduce((a,x)=>a+x.bombs,0),episodes};
 }
 fs.writeFileSync(out,JSON.stringify({completed:true,assist:false,directional_only:true,settlement:'primary300 only; not authoritative safe-win metric',model_json_sha256:loaded.sha,results}));
}
if(require.main===module)run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
