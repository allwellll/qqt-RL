'use strict';
// Full diagnostic games. No model training data. Each carrier acquires its
// package by walking to real stock; optional control/contact fixtures are
// injected only AFTER this causal acquisition and mirrored on both engines.
const Q=require('../web/sim-bun-native-v2');
const level=require('../web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const [seat,seed,mode]=process.argv.slice(2).map((x,i)=>i<2?Number(x):x);
const s=new Q.Sim(seed);s.reset(level,{nativeTrap:true,nativeItems:true});s.maxSteps=2400;
for(const k of ['brick','brickLinger','bush','cover','crate','fuse','pushable','fieldItem','invuln','spawnProtection'])s[k].fill(0);
s.itemSlots=[[],[]];s.owner.fill(-1);
s.pos.set(seat===0?[2.5,5.5,10.5,10.5]:[10.5,3.5,2.5,9.5]);
s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];
function projection(){return {t:s.t,pos:Array.from(s.pos),alive:Array.from(s.alive),hp:Array.from(s.hp),
 carried:Array.from(s.bunCarried),stored:s.bunStored.map(x=>x.slice()),loose:Array.from(s.bunLoose),respawn:Array.from(s.bunRespawn),
 trapped:Array.from(s.trapped),invuln:Array.from(s.invuln),done:s.done,winner:s.winner};}
const initial={...projection(),wall:Array.from(s.wall),spawn:s.bunSpawnPos,stats:[Array.from(s.bombsCap),Array.from(s.blastCap),Array.from(s.spdG)]};
function move(p,team){
 const b=s.bunBases[team],goals=new Set();for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)goals.add(r*15+c);
 const [r,c]=s.centerCell(p),cell=r*15+c,dist=new Array(195).fill(1000),q=[...goals];for(const z of q)dist[z]=0;
 const ds=[[-1,0],[1,0],[0,-1],[0,1]];
 for(let i=0;i<q.length;i++){const z=q[i],rr=Math.floor(z/15),cc=z%15;for(const [dr,dc] of ds){const a=rr+dr,bb=cc+dc,v=a*15+bb;if(a>=0&&a<13&&bb>=0&&bb<15&&!s.wall[v]&&dist[v]>dist[z]+1){dist[v]=dist[z]+1;q.push(v);}}}
 if(goals.has(cell))return 4;
 let best=4,score=dist[cell];for(let k=0;k<4;k++){const a=r+ds[k][0],bb=c+ds[k][1],v=a*15+bb;if(a>=0&&a<13&&bb>=0&&bb<15&&dist[v]<score){best=k;score=dist[v];}}
 return best;
}
const rows=[];let injected=false,transfers=[];
const resolve=s._updateNativeTraps;
// Observe transfer at the exact pre-deposit contact edge, including recovery
// deposited in the same tick. Final bunCarried alone would miss that event.
const kill=s._killPlayer;
s._killPlayer=function(p){if(this.bunCarried[p]<0&&this.bunCarried[1-p]>=0&&this.trapped[p]>0)
 transfers.push({victim:p,receiver:1-p,origin:this.bunCarried[1-p]});return kill.call(this,p);};
while(!s.done&&s.t<2400){
 let injection=null;
 if(mode!=='delivery'&&!injected&&s.bunCarried[seat]>=0){
  injected=true;s.trapped[seat]=mode==='drop'?1:10;
  if(mode==='transfer'){s.pos[(1-seat)*2]=s.pos[seat*2];s.pos[(1-seat)*2+1]=s.pos[seat*2+1];}
  injection={trapped:Array.from(s.trapped),pos:Array.from(s.pos)};
 }
 const actions=[[4,0,0],[4,0,0]];
 for(let p=0;p<2;p++)if(p===seat||mode==='transfer'&&injected){
  actions[p][0]=move(p,s.bunCarried[p]>=0?s.team[p]:1-s.team[p]);
 }
 if(mode==='transfer'&&injection)actions[1-seat][0]=4;
 transfers=[];const info=s.step(actions);
 rows.push({injection,actions:actions.map(x=>[x[0],x[1]||x[2]*2]),after:projection(),death:Array.from(info.died),transfers});
}
console.log(JSON.stringify({rule:Q.RULE_VERSION,seat,seed,mode,initial,rows}));
