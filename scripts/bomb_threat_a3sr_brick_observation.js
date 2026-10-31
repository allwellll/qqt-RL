'use strict';
// Nine preregistered same-seed P1 brick-separated games; descriptive only.
const fs=require('fs'),assert=require('assert');
process.env.QQT_STOCK_RULE='stock-window-v1';process.env.BT_FRESH_TIER='hard';
const Q=require('../web/sim-stock-window-v1'),{bfs:legacyBfs}=require('./bomb_threat_low_bomb_metrics');
// Production crates are walk-over pickups; keep the original selected9 seeds
// but do not reproduce the historical observer's false crate barrier.
const bfs=(s,start,ignoreBricks=false)=>legacyBfs({...s,crate:null},start,ignoreBricks);
const [model,out,idx]=process.argv.slice(2),index=Number(idx);
const manifest=JSON.parse(fs.readFileSync('runs/bomb_threat_v1/iter2/a3sr/BRICK_OBSERVATION_MANIFEST.json'));
const entry=manifest.games.find(e=>e.index===index);assert(entry&&entry.seat===1);
const m={index,seed:entry.seed,seat:1,ticks:0,active_both_ticks:0,brick_only_ticks:0,connected_ticks:0,near5_ticks:0,
 placed:0,all_sources_bricks_removed:0,first_connected_tick:null,first_near5_tick:null,
 scope:'passive full native2400tick greedy; all-source removed bricks not causal learner credit; no gate'};
const original=Q.Sim.prototype.step;
Q.Sim.prototype.step=function(actions){
 if(this.__btIdentity?.seed!==entry.seed)return original.call(this,actions);
 const p=1,start=this.centerCell(p),target=this.centerCell(0),d=bfs(this,start[0]*15+start[1])[target[0]*15+target[1]],
  bd=bfs(this,start[0]*15+start[1],true)[target[0]*15+target[1]],
  active=this.alive[0]&&this.alive[1]&&!this.trapped[0]&&!this.trapped[1],brick=Array.from(this.brick);
 m.ticks++;m.active_both_ticks+=Number(active);m.brick_only_ticks+=Number(active&&d<0&&bd>=0);
 m.connected_ticks+=Number(active&&d>=0);m.near5_ticks+=Number(active&&d>=0&&d<=5);
 if(active&&d>=0&&m.first_connected_tick===null)m.first_connected_tick=this.t;
 if(active&&d>=0&&d<=5&&m.first_near5_tick===null)m.first_near5_tick=this.t;
 const info=original.call(this,actions);m.placed+=Number(info.placed[p]);
 m.all_sources_bricks_removed+=brick.reduce((n,v,i)=>n+Number(v&&!this.brick[i]),0);
 m.done=this.done;m.winner=this.winner;return info;
};
process.env.JE10_ROUTE_TASK='natural_full2400';process.env.JE10_ROUTE_GAMES='256';
process.env.JE10_ROUTE_START=String(index);process.env.JE10_ROUTE_END=String(index+1);
process.env.JE10_ROUTE_SEED_START='2026142900';
process.argv=[process.argv[0],process.argv[1],'greedy',model,out,'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',manifest.spawn_manifest];
require('./bomb_threat_fresh256_eval');
process.on('beforeExit',()=>{
 if(m.status)return;assert(m.done&&m.ticks>0&&m.ticks<=2400,'complete natural game required');
 m.status='COMPLETE';fs.writeFileSync(out+'.brick_observation.json.tmp',JSON.stringify(m,null,2));fs.renameSync(out+'.brick_observation.json.tmp',out+'.brick_observation.json');
});
