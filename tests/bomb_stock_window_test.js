'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const Q=require('../web/sim-stock-window-v1');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const results=[];
for(const seat of [0,1])for(const mode of ['free','slow','dead','sugar','expired','loose','delivery']){
 const s=new Q.Sim(20261029);s.reset(level,{nativeTrap:true,nativeItems:true});
 s.brick.fill(0);s.trapped.fill(0);s.alive.fill(true);s.bunCarried.fill(-1);s.bunLoose.fill(0);
 const enemy=1-seat,base=s.bunBases[mode==='delivery'?seat:enemy];
 // bunBases are 3x3 top-left cells; choose their center.
 s.pos[2*seat]=base[0]+1.5;s.pos[2*seat+1]=base[1]+1.5;
 if(mode==='dead')s.alive[enemy]=false;
 if(mode==='sugar')s.trapped[enemy]=10;
 if(mode==='slow'){s.movementStatus[enemy]=1;s.movementStatusTicks[enemy]=20;}
 if(mode==='expired')s.bunRespawn[enemy]=50;
 if(mode==='loose'){const [r,c]=s.centerCell(seat);s.bunLoose[(r*15+c)*2+enemy]=1;}
 if(mode==='delivery'){s.bunCarried[seat]=enemy;s.bunStored[enemy][enemy]=0;}
 s._bunUpdate();
 const expected=['dead','sugar','loose'].includes(mode)?enemy:-1;
 assert.equal(s.bunCarried[seat],expected,`${seat}/${mode}`);
 if(mode==='delivery')assert.equal(s.bunStored[seat][enemy],1);
 results.push({seat,mode,carried:s.bunCarried[seat],stored:s.bunStored});
}
console.log(JSON.stringify(results));
