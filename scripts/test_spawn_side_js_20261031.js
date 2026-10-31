'use strict';
const fs=require('fs'),assert=require('assert'),Q=require('../web/sim-bun-native-v2'),S=require('./bun_spawn_side');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
let cases=0;
for(let seed=0;seed<256;seed++){
 const sim=new Q.Sim(seed);sim.reset(level,{nativeItems:true,nativeTrap:true});
 const native=[0,1].map(p=>Array.from(sim.pos.slice(2*p,2*p+2),Math.floor));S.validate(native);
 const pair=[level.bun_spawns[0][seed%4],level.bun_spawns[1][Math.floor(seed/4)%4]];S.apply(sim,pair);
 sim._killPlayer(0);sim._killPlayer(1);
 for(let t=0;t<sim.bunRespawnTicks;t++)sim.step([[4,0,0],[4,0,0]]);
 assert(sim.alive.every(Boolean));assert.deepEqual(Array.from(sim.pos),pair.flatMap(c=>c.map(x=>x+.5)));
 cases++;
}
for(const pair of [[[4,12],[4,2]],[[4,7],[4,12]],[[4,2],[4,7]]])assert.throws(()=>S.validate(pair));
console.log('JS own-side native + frozen birth + actual death/respawn PASS',cases);
