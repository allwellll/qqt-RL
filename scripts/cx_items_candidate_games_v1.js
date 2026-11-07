'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const c=JSON.parse(fs.readFileSync(process.argv[2])),Q=require(path.resolve(c.engine));
const A=require(path.resolve(c.production+'/web/tactical_arena.js'));
const base=JSON.parse(fs.readFileSync(c.production+'/web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const catalog=JSON.parse(fs.readFileSync(c.catalog)).rows;
const result=[];
for(let i=0;i<c.count;i++){
 const seed=c.seeds[c.split]+i,seat=1;let level=JSON.parse(JSON.stringify(base)),specialist=null;
 if(c.kind==='duel'){
  const split=c.split==='holdout'?'holdout':'dev',rows=catalog.filter(r=>r.split===split);
  // Distinct variants for screen/dev; dev/holdout have separate scenario seed streams.
  const r=rows[(i%8)*4+(c.split==='screen'?0:1+Math.floor(i/8))];assert(r);
  specialist=r;level={...A.buildTacticalArena(base),wall:r.wall.flat().map(Number),brick:r.brick.flat().map(Number),
   spawns:[...Array(4).fill(r.spawn[0]),...Array(4).fill(r.spawn[1])],push_boxes:[],initial_crates:[],layers:null};
 } else if(c.kind==='bun.tactical_v2')level=A.buildTacticalArena(base);
 else if(c.kind==='strong_full'&&i%3!==0){
  // Bounded complete-game probe: intact, partial, and cleared brick layouts.
  level.brick=level.brick.map((x,k)=>i%3===1?(k%3?x:0):0);level.layers=null;
 }
 const sim=new Q.Sim(seed);sim.reset(level,{nativeItems:true,nativeTrap:true});
 const g={seed,seat,mode:'greedy',opponent_mode:'tau03',action_seed:(seed^0x515154)>>>0,level,map_id:806,
  spawn:[0,1].map(p=>Array.from(sim.pos.slice(p*2,p*2+2),v=>Math.floor(v))),
  native_respawn:sim.bunSpawnPos.map(x=>x.map(v=>Math.floor(v))),region:c.kind==='duel'?specialist.family:'web_native_default',
  brick_stage:c.kind==='duel'?'artificial_'+specialist.family:c.kind==='bun.tactical_v2'?'web_tactical_arena':c.kind==='strong_full'?['complete','partial','cleared'][i%3]:'complete',
  evaluation_split:'items_candidates_'+c.split,horizon:c.kind==='duel'?300:2400};
 if(specialist){g.specialist=specialist;g.spawn=specialist.spawn;g.native_respawn=specialist.spawn;}
 result.push(g);
}
fs.writeFileSync(c.output,JSON.stringify(result));
