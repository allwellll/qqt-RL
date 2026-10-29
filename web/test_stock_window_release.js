'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),crypto=require('crypto');
const Legacy=require('./sim'),Stock=require('./sim-stock-window-v1');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
function fresh(Q,seed){const s=new Q.Sim(seed);s.reset(level,{nativeItems:true,nativeTrap:true,teams:[0,1]});return s;}
function atEnemy(Q,state){const s=fresh(Q,20261030);s.brick.fill(0);s.bunCarried.fill(-1);s.bunLoose.fill(0);s.alive.fill(true);s.trapped.fill(0);const b=s.bunBases[1];s.pos[0]=b[0]+1.5;s.pos[1]=b[1]+1.5;if(state==='dead')s.alive[1]=false;if(state==='trapped')s.trapped[1]=30;s._bunUpdate();return s.bunCarried[0];}
assert.equal(atEnemy(Legacy,'alive'),1);assert.equal(atEnemy(Stock,'alive'),-1);
for(const state of ['dead','trapped']){assert.equal(atEnemy(Legacy,state),1);assert.equal(atEnemy(Stock,state),1);}
let frames=0;
for(let seed=20261030;seed<20261046;seed++){
 const a=fresh(Legacy,seed),b=fresh(Stock,seed),rng=Legacy.mulberry32(seed^0xF3A);
 // Disable stock inventory only: physical items/traps/respawn remain native.
 for(const s of [a,b])s.bunStored.forEach(x=>x.fill(0));
 for(let tick=0;tick<240;tick++){
  const acts=[0,1].map(()=>[Math.floor(rng()*5),Number(rng()<.2),Number(rng()<.08)]);
  const x=a.step(acts),y=b.step(acts);
  assert.deepEqual(b.snapshotReplay(y),a.snapshotReplay(x),`non-stock frame seed=${seed} tick=${tick}`);frames++;
 }
}
assert.equal(crypto.createHash('sha256').update(fs.readFileSync('web/sim-stock-window-v1.js')).digest('hex'),'2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45');
assert(fs.readFileSync('web/index.html','utf8').includes('src="sim-stock-window-v1.js"'));
for(const f of ['web/review.html','web/training.html'])assert(!fs.readFileSync(f,'utf8').includes('sim-stock-window-v1.js'));
console.log(`stock-window release: PASS, ${frames} native non-stock frames, alive/death/native-trap inventory gates`);
