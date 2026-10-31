'use strict';
const fs=require('node:fs'),zlib=require('node:zlib'),Q=require('../web/sim-kill-arena');
const [mapIndexText,seedText,target]=process.argv.slice(2),mapIndex=Number(mapIndexText),seed=Number(seedText);
const record=JSON.parse(fs.readFileSync('jax_bomb/maps/kill_maps.json')).maps[mapIndex],s=new Q.Sim(seed);
const pair=[record.map.spawns[seed%record.map.spawns.length],record.map.spawns[(seed+1)%record.map.spawns.length]];
s.resetArena(record.map,{itemsDisabled:true,spawnPair:pair});
function snapshot(s){return Object.fromEntries(['pos','alive','hp','invuln','bombsCap','blastCap','spdG','trapped',
 'wall','brick','bush','pushable','crate','fuse','owner','bombBlast','blastLinger','fieldItem','fieldOwner','fieldArmed',
 'movementStatus','movementStatusTicks','slideDir','lastMoveDir','horzLinger','vertLinger','brickLinger','pushT',
 'arenaKills','arenaDeaths','arenaSelf','arenaLifeAge'].map(k=>[k,Array.from(s[k])]))}
const initial=snapshot(s),rng=Q.mulberry32(seed^0x1234abcd),rows=[];
// A persistent random walk produces contact, bomb chains and sugar deaths;
// actions are declared once and replayed identically by JAX, never a teacher.
let directions=[4,4];
for(let t=0;t<1200;t++){
 if(t%4===0)directions=[Math.floor(rng()*5),Math.floor(rng()*5)];
 const actions=[0,1].map(p=>[directions[p],Number(rng()<.16),0]);
 const r=s.step(actions);
 rows.push({actions,after:snapshot(s),events:r.arena_events,done:s.done,winner:s.winner});
}
const doc={ruleset:Q.RULE_VERSION,map_index:mapIndex,map_sha256:record.map_sha256,seed,items:'disabled',
 engine:'production-derived independent JS Sim.step',initial,rows};
fs.mkdirSync(require('node:path').dirname(target),{recursive:true});
fs.writeFileSync(target+'.tmp',zlib.gzipSync(JSON.stringify(doc)));fs.renameSync(target+'.tmp',target);
console.log(JSON.stringify({games:1,ticks:rows.length,deaths:s.arenaDeaths,kills:s.arenaKills,file:target}));
