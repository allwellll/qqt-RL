'use strict';
const fs=require('node:fs'),Q=require('../web/sim-bun-native-v2'),Arena=require('../web/tactical_arena');
function games(bot,mode,count=64){
 const base=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
 const level=bot==='bun.tactical_v2'?Arena.buildTacticalArena(base):base;
 return Array.from({length:count},(_,i)=>{
  const seed=2026619000+i,s=new Q.Sim(seed);s.reset(level,{nativeItems:true,nativeTrap:true});
  return {seed,seat:1,mode,opponent_mode:'greedy',action_seed:(seed^0x515154)>>>0,level,map_id:806,
   spawn:[0,1].map(p=>Array.from(s.pos.slice(p*2,p*2+2),v=>Math.floor(v))),native_respawn:s.bunSpawnPos.map(a=>a.map(x=>Math.floor(x))),
   region:'web_native_default',brick_stage:bot==='bun.tactical_v2'?'web_tactical_arena_clear':'web_native_full',evaluation_split:'cx_web_temperature_v1'};
 });
}
if(require.main===module){const c=JSON.parse(fs.readFileSync(process.argv[2]));fs.writeFileSync(c.output,JSON.stringify(games(c.bot,c.mode,c.count)));}
module.exports={games};
