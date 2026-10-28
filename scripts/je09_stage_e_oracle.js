'use strict';
// Diagnostic forks start from the exact real first-pickup state. No training action takeover.
const fs=require('fs'),crypto=require('crypto'),assert=require('node:assert/strict');
const Q=require('../web/sim.js'),fixed=require('./je09_web_stage_d.js'),p=require('./je08_web_diagnostics.js');
const stB=require('./je09_web_stage_b.js'),inf=require('./je05_web_hunter.js'),a=require('./eval_web_hunter.js'),o=require('./je09_protocol_diagnosis.js');
function clone(v){if(v===null||typeof v!=='object')return v;if(ArrayBuffer.isView(v))return new v.constructor(v);if(Array.isArray(v))return v.map(clone);const x=Object.create(Object.getPrototypeOf(v));for(const k of Object.keys(v))x[k]=clone(v[k]);return x;}
function robust(s,team,map,memory){
 const danger=s.dangerMap(),blocked=Uint8Array.from(s.wall,(v,i)=>Number(v||s.brick[i]||s.fuse[i]>0));
 const goals=map.map((v,i)=>v===0?i:-1).filter(i=>i>=0),route=Array(Q.N).fill(1e9),seen=new Set();goals.forEach(i=>route[i]=0);
 for(let k=0;k<Q.N;k++){let at=-1;for(let i=0;i<Q.N;i++)if(!seen.has(i)&&(at<0||route[i]<route[at]))at=i;if(at<0||route[at]>1e8)break;seen.add(at);const r=Math.floor(at/Q.W),c=at%Q.W;for(const [dr,dc]of Q.DIRS){const rr=r+dr,cc=c+dc,i=rr*Q.W+cc;if(rr<0||rr>=Q.H||cc<0||cc>=Q.W||blocked[i])continue;route[i]=Math.min(route[i],route[at]+1+200*danger[at]+100*Number(s.blastLinger[at]>0));}}
 const y=s.pos[team*2],x=s.pos[team*2+1],r=Math.floor(y),c=Math.floor(x),cell=r*Q.W+c;
 if(memory.target===undefined||blocked[memory.target]||danger[memory.target]>0||Math.abs(y-Math.floor(memory.target/Q.W)-.5)+Math.abs(x-memory.target%Q.W-.5)<.16){
  let best=cell,score=route[cell];for(const [dr,dc]of Q.DIRS){const rr=r+dr,cc=c+dc,i=rr*Q.W+cc;if(rr<0||rr>=Q.H||cc<0||cc>=Q.W||blocked[i])continue;const candidate=route[i]+1+200*danger[i];if(candidate<score+1e-6){best=i;score=candidate;}}memory.target=best;
 }
 const ty=Math.floor(memory.target/Q.W)+.5,tx=memory.target%Q.W+.5,mask=s.legalMask().mm[team],step=Q.CFG.stepLen*s.spdG[team]*s.playerMoveScale(team);
 let best=4,bestScore=Math.abs(y-ty)+Math.abs(x-tx)+100*danger[cell];
 for(let k=0;k<4;k++){if(!mask[k])continue;const [yy,xx]=s._steer(y,x,k,blocked,step,team),i=Math.floor(yy)*Q.W+Math.floor(xx),score=Math.abs(yy-ty)+Math.abs(xx-tx)+100*danger[i]+100*Number(s.blastLinger[i]>0);if(score<bestScore-1e-7){best=k;bestScore=score;}}return best;
}
function home(s,team){const b=s.bunBases[team],goals=[];for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)goals.push(r*Q.W+c);return p.distance(s.wall,s.brick,goals);}
async function run(out){
 const ns=process.env.JE09_NAMESPACE,dec=JSON.parse(fs.readFileSync(ns+'/ROUND_DECLARATION.json'));
 const paths={C:'runs/je09_bun_20261015/stage_c_return_set_pfsp8_v8/authority_je09_v1/bun392_bridge_pfsp8_g000008/model/model.json',D:'runs/je09_bun_20261015/stage_d_distance_set_pfsp8_v9/authority_je09_v1/bun392_bridge_pfsp8_g000016/model/model.json'};
 const models={};for(const k of Object.keys(paths))models[k]=await inf.loadModel(paths[k]);
 const episodes=[];
 for(let game=0;game<16;game++){
  const seed=(dec.seeds.oracle+game*104729)>>>0,{s,team,enemy,target,initial}=fixed.setup(seed,game),tracker=new stB.Tracker(team),rng=Q.mulberry32(seed^0x515154),hunter=a.createHunter('normal');
  tracker.trapCause[enemy][team]=true;hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:'stage-e-oracle-'+game,seed,ruleset:'bun',max_ticks:1200,metadata:{}});
  while(!s.done&&s.t<101&&tracker.first===null){const b=p.snapshot(s),raw=await models.D.model.act(s,team,rng),bot=hunter.adapter.act({tick:s.t,metadata:{sim:s}},enemy,rng),acts=[];acts[team]=raw;acts[enemy]=[bot.move,Number(bot.ability===1),Number(bot.ability===2)];tracker.observe(s,b,s.step(acts));}
  assert.ok(tracker.first!==null,'Stage D real first pickup required');
  const snapshot=s.snapshotReplay(),stateSha=crypto.createHash('sha256').update(JSON.stringify(snapshot)).digest('hex'),map=home(s,team),cell=s.centerCell(team),distance=map[cell[0]*Q.W+cell[1]],forks={};
  for(const who of (process.argv[3]==='oracle-only'?['oracle']:['C','D','oracle'])){
   const sim=clone(s),t=clone(tracker),bot=a.createHunter('normal'),rng=Q.mulberry32(seed^0xE100),path=[],memory={};bot.adapter.reset({schema:'qqt.bot.context/v1',episode_id:'fork-'+game,seed,ruleset:'bun',max_ticks:1200,metadata:{}});
   while(!sim.done&&sim.t<snapshot.t+400&&!t.captured){const before=p.snapshot(sim),here=sim.centerCell(team),d=map[here[0]*Q.W+here[1]],raw=who==='oracle'?[robust(sim,team,map,memory),0,0]:await models[who].model.act(sim,team,rng),other=bot.adapter.act({tick:sim.t,metadata:{sim}},enemy,rng),acts=[];path.push({tick:sim.t,pos:Array.from(sim.pos),distance:d,mask:sim.legalMask().mm[team],action:raw});acts[team]=raw;acts[enemy]=process.argv[4]==='geometry-only'?[4,0,0]:[other.move,Number(other.ability===1),Number(other.ability===2)];t.observe(sim,before,sim.step(acts));}
   forks[who]={...t.result(),path,start_state_sha256:stateSha};
  }
  episodes.push({game,seed,team,initial,pickup_tick:tracker.first,return_distance:distance,start_state_sha256:stateSha,snapshot,forks});
 }
 const result={passed:episodes.every(e=>e.return_distance>=24&&e.return_distance<=36&&e.forks.oracle.capture),oracle_only_diagnostic:true,training_takeover:false,geometry_only_opponent_noop:process.argv[4]==='geometry-only',models:Object.fromEntries(Object.entries(models).map(([k,v])=>[k,v.sha])),episodes,summary:Object.fromEntries((process.argv[3]==='oracle-only'?['oracle']:['C','D','oracle']).map(k=>[k,{games:16,capture:episodes.filter(e=>e.forks[k].capture).length,progress25:episodes.filter(e=>e.forks[k].progress25).length,reverse:episodes.reduce((n,e)=>n+e.forks[k].reverse_cell_transitions,0)}]))};
 fs.writeFileSync(out,JSON.stringify(result));console.log(result.summary);assert.ok(result.passed);
}
if(require.main===module)run(process.argv[2]).catch(e=>{console.error(e);process.exitCode=1;});
