'use strict';
// Original fixed32 timing, plus independently seeded reachable distance gates.
const fs=require('fs'),assert=require('node:assert/strict');
const Q=require('../web/sim.js'),p=require('./je08_web_diagnostics.js');
const b=require('./je09_web_bridge.js'),stageB=require('./je09_web_stage_b.js');
const oracle=require('./je09_protocol_diagnosis.js'),inf=require('./je05_web_hunter.js');
const authority=require('./eval_web_hunter.js'),residual=require('./eval_web_hunter_residual.js');
const namespace=process.env.JE09_NAMESPACE||'runs/je09_bun_20261015/stage_d_distance_set_pfsp8_v9';
function declaration(){return JSON.parse(fs.readFileSync(namespace+'/ROUND_DECLARATION.json','utf8'));}
function setup(seed,game,bucket=null){
 const team=game%2,enemy=1-team,s=p.setup('drop_return',seed,team,null,1200),rng=Q.mulberry32(seed);
 if(bucket){
  const pairs=[],wanted=bucket[0]+game%(bucket[1]-bucket[0]+1);
  for(let target=0;target<Q.H*Q.W;target++)if(!s.wall[target]&&s._bunBaseTeam(Math.floor(target/Q.W),target%Q.W)<0){
   const d=p.distance(s.wall,s.brick,[target]);
   for(let start=0;start<d.length;start++)if(!s.wall[start]&&d[start]===wanted)pairs.push([start,target]);
  }
  assert.ok(pairs.length,'unreachable requested bucket');const [start,target]=pairs[Math.floor(rng()*pairs.length)];
  for(const [t,i]of [[team,start],[enemy,target]]){s.pos[t*2]=Math.floor(i/Q.W)+.5;s.pos[t*2+1]=i%Q.W+.5;s.lastMoveDir[t]=Math.floor(rng()*4);}
  s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];s._gen++;
 }
 const cell=s.centerCell(team),e=s.centerCell(enemy),target=e[0]*Q.W+e[1],map=p.distance(s.wall,s.brick,[target]);
 return {s,team,enemy,target,map,initial:{pos:Array.from(s.pos),heading:s.lastMoveDir.slice(),distance:map[cell[0]*Q.W+cell[1]],target,max_steps:s.maxSteps,respawn_ticks:s.bunRespawnTicks}};
}
async function evaluate(model,out,mode='main',bucketIndex='0'){
 assert.equal(process.env.CUDA_VISIBLE_DEVICES||'','');const dec=declaration(),isOracle=mode==='oracle',gate=mode==='gate',bi=Number(bucketIndex);
 const loaded=isOracle?null:await inf.loadModel(model),groups=isOracle?['bucket0','bucket1','bucket2','bucket3','main','holdout']:[gate?'bucket'+bi:mode],results={};
 for(const group of groups){
  const bucket=group.startsWith('bucket')?dec.distance_buckets[Number(group.slice(6))]:null;
  const base=bucket?(isOracle?dec.seeds.oracle:dec.seeds.curriculum_gate)+Number(group.slice(6))*2000000:dec.seeds[group==='holdout'?'fixed_holdout':'fixed_main'];
  const episodes=[];
  for(let game=0;game<16;game++){
   const seed=(base+game*104729)>>>0,{s,team,enemy,target,map,initial}=setup(seed,game,bucket),t=new stageB.Tracker(team),rng=Q.mulberry32(seed^0x515154),hunter=authority.createHunter('normal'),path=[];
   t.trapCause[enemy][team]=true;hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:'stage-d-'+group+'-'+game,seed,ruleset:'bun',max_ticks:1200,metadata:{}});
   let dropEnd=null,pickupAt=null,reverse=0,previous=null;
   async function step(settling){
    const before=p.snapshot(s),cell=s.centerCell(team),d=map[cell[0]*Q.W+cell[1]],loose=s.bunLoose[target*2+enemy]>0;
    let raw;
    if(isOracle){const own=s.bunBases[team],goals=[];for(let r=own[0];r<own[0]+3;r++)for(let c=own[1];c<own[1]+3;c++)goals.push(r*Q.W+c);const home=p.distance(s.wall,s.brick,goals);const ix=home.indexOf(0);raw=[oracle.expert(s,team,s.bunCarried[team]===enemy?ix:target),0,0];}
    else raw=await loaded.model.act(s,team,rng);
    if(loose&&previous!==null&&d>previous)reverse++;if(loose)previous=d;
    if(s.t<125)path.push({tick:s.t,pos:Array.from(s.pos),cell:cell[0]*Q.W+cell[1],distance:d,loose,respawn:s.bunRespawn.slice(),heading:s.lastMoveDir.slice(),mask:s.legalMask().mm[team],action:raw});
    const bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng),actions=[];
    actions[team]=[raw[0],settling?0:Number(raw[1]),settling?0:Number(raw[2]||0)];actions[enemy]=[bot.move,settling?0:Number(bot.ability===1),settling?0:Number(bot.ability===2)];
    if(settling)s.done=false;t.observe(s,before,s.step(actions));
    if(loose&&s.bunLoose[target*2+enemy]===0&&dropEnd===null)dropEnd=s.t;
    if(t.first!==null&&pickupAt===null)pickupAt=t.first;
   }
   const ticks=isOracle||gate?125:1200;
   while(!s.done&&s.t<ticks)await step(false);
   const primary=s.t;s.maxSteps=primary+125;while(residual.hasHazards(s)&&s.t<primary+124)await step(true);assert.ok(!residual.hasHazards(s));
   const result=t.result();episodes.push({game,seed,team,initial,...result,pickup_tick:pickupAt,drop_end_tick:dropEnd,bag_returned:dropEnd!==null&&pickupAt===null,approach_reverse_cells:reverse,min_distance:Math.min(...path.filter(x=>x.loose).map(x=>x.distance)),path,primary_ticks:primary});
  }
  results[group]=b.summarize(episodes);
 }
 const data={passed:true,assist:false,oracle:isOracle,mode,results,model_json_sha256:loaded?.sha??null,native_fixed_ticks:1200,native_respawn_ticks:100,scope:'controlled native attributed drop; independent of full natural chain'};
 fs.writeFileSync(out+'.tmp',JSON.stringify(data));fs.renameSync(out+'.tmp',out);return data;
}
if(require.main===module)evaluate(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={setup,evaluate};
