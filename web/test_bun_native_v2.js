'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const native=require('../web/sim-bun-native-v2'),window=require('../web/sim-stock-window-v1');
const level=require('../web/assets/maps/levels.json').find(x=>x.qqt_id===806);
assert.equal(native.RULE_VERSION,'bun-native-v2');
const canonical=fs.readFileSync('web/sim.js','utf8').replace('  const QQT = {',"  const QQT = {\n    RULE_VERSION: 'bun-native-v2',");
assert.equal(fs.readFileSync('web/sim-bun-native-v2.js','utf8'),canonical);
for(const seat of [0,1])for(const mode of ['free','slow','dead','sugar','loose','delivery'])for(const Q of [native,window]){
 const s=new Q.Sim(20261030);s.reset(level,{nativeItems:true,nativeTrap:true});
 const enemy=1-seat,b=s.bunBases[mode==='delivery'?seat:enemy];
 s.pos[seat*2]=b[0]+1.5;s.pos[seat*2+1]=b[1]+1.5;
 s.alive.fill(true);s.trapped.fill(0);s.bunCarried.fill(-1);s.bunLoose.fill(0);
 if(mode==='slow'){s.movementStatus[enemy]=1;s.movementStatusTicks[enemy]=10;}
 if(mode==='dead')s.alive[enemy]=false;
 if(mode==='sugar')s.trapped[enemy]=10;
 if(mode==='loose'){const [r,c]=s.centerCell(seat);s.bunLoose[(r*15+c)*2+enemy]=1;}
 if(mode==='delivery'){s.bunCarried[seat]=enemy;s.bunStored[enemy][enemy]=0;}
 s._bunUpdate();
 assert.equal(s.bunCarried[seat],mode==='delivery'?-1:(Q===native||['dead','sugar','loose'].includes(mode)?enemy:-1));
 if(mode==='delivery'){assert.equal(s.winner,seat);assert.equal(s.done,true);}
}
console.log('native-v2 snapshot, both-seat free/slow/death/sugar/loose/delivery rule regression PASS');
