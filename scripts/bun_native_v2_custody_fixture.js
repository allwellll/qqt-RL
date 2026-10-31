'use strict';
// Counterfactual TEST fixtures only. Every package starts as native inventory.
// Control/position interventions happen after real acquisition and are mirrored
// in JAX. These are not autonomous games, training Bank states or win evidence.
const Q=require('../web/sim-bun-native-v2');
const level=require('../web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const seat=Number(process.argv[2]||0),mode=process.argv[3]||'cycles',seed=2026208100+seat;
const s=new Q.Sim(seed);s.reset(level,{nativeTrap:true,nativeItems:true});s.maxSteps=5000;
for(const k of ['brick','brickLinger','bush','cover','crate','fuse','pushable','fieldItem','invuln','spawnProtection'])s[k].fill(0);
s.itemSlots=[[],[]];s.owner.fill(-1);
s.pos.set([2.5,5.5,10.5,10.5]);s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];
if(mode==='ordinary'){
 s.pos.set(seat===0?[10.5,8.5,10.5,10.5]:[10.5,10.5,10.5,8.5]);
 const z=10*15+10;s.fuse[z]=1;s.owner[z]=seat;s.bombBlast[z]=1;
}
function projection(){return {t:s.t,pos:Array.from(s.pos),alive:Array.from(s.alive),hp:Array.from(s.hp),
 carried:Array.from(s.bunCarried),stored:s.bunStored.map(x=>x.slice()),loose:Array.from(s.bunLoose),respawn:Array.from(s.bunRespawn),
 trapped:Array.from(s.trapped),invuln:Array.from(s.invuln),done:s.done,winner:s.winner};}
const initial={...projection(),wall:Array.from(s.wall),spawn:s.bunSpawnPos,stats:[Array.from(s.bombsCap),Array.from(s.blastCap),Array.from(s.spdG)],fuse:Array.from(s.fuse),owner:Array.from(s.owner),bombBlast:Array.from(s.bombBlast)};
function field(goals){const d=Array(195).fill(32767),q=goals.filter(z=>!s.wall[z]);for(const z of q)d[z]=0;
 for(let i=0;i<q.length;i++){const z=q[i],r=Math.floor(z/15),c=z%15;for(const [dr,dc] of [[-1,0],[1,0],[0,-1],[0,1]]){
 const a=r+dr,b=c+dc,v=a*15+b;if(a>=0&&a<13&&b>=0&&b<15&&!s.wall[v]&&d[v]>d[z]+1){d[v]=d[z]+1;q.push(v);}}}return d;}
function cage(t){const b=s.bunBases[t],out=[];for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)out.push(r*15+c);return out;}
function toward(p,goals){const d=field(goals),[r,c]=s.centerCell(p);let best=4,score=d[r*15+c];
 [[-1,0],[1,0],[0,-1],[0,1]].forEach(([dr,dc],k)=>{const a=r+dr,b=c+dc;if(a>=0&&a<13&&b>=0&&b<15&&d[a*15+b]<score){best=k;score=d[a*15+b];}});return best;}
function canonicalCell(p){return [Math.floor(Math.round(s.pos[p*2]*1000)/1000),Math.floor(Math.round(s.pos[p*2+1]*1000)/1000)];}
function phi(team){const enemy=1-team;if(s.bunStored[team][team]>0)return 0;
 const f=field(cage(enemy)),span=Math.max(1,Math.min(...cage(team).filter(z=>!s.wall[z]).map(z=>f[z]))),[r,c]=canonicalCell(enemy);
 if(s.bunCarried[enemy]===team)return -.30-.50*Math.max(0,Math.min(1,1-f[r*15+c]/span));
 let target=-1;for(let z=0;z<195;z++)if(s.bunLoose[z*2+team]>0){target=z;break;}
 if(s.bunCarried[team]===team){const rc=canonicalCell(team);target=rc[0]*15+rc[1];}
 return target<0?-.8:-.10-.10*Math.max(0,Math.min(1,1-field([target])[r*15+c]/span));}
const cycleMode=['cycles','dropcycles'].includes(mode);
let cycles=0,active=false,intercepted=false,tail=0;const rows=[];
while(!s.done&&s.t<4500){let injection=null;
 const enemy=1-seat;
 if(mode==='ordinary'&&s.t===1){
  s.pos[seat*2]=s.pos[enemy*2];s.pos[seat*2+1]=s.pos[enemy*2+1];s.blastLinger.fill(0);
  injection={pos:Array.from(s.pos),blastLinger:Array.from(s.blastLinger)};
 }
 if(cycleMode&&active&&!intercepted&&s.bunCarried[enemy]===seat){
  const f=field(cage(enemy)),span=Math.min(...cage(seat).map(z=>f[z])),rc=s.centerCell(enemy);
  if(1-f[rc[0]*15+rc[1]]/span>=.45){
   if(mode==='dropcycles'){
    const z=rc[0]*15+rc[1];s.fuse[z]=1;s.owner[z]=seat;s.bombBlast[z]=1;
    injection={fuse:Array.from(s.fuse),owner:Array.from(s.owner),bombBlast:Array.from(s.bombBlast)};
   }else{
    s.trapped[enemy]=10;s.pos[seat*2]=s.pos[enemy*2];s.pos[seat*2+1]=s.pos[enemy*2+1];
    injection={trapped:Array.from(s.trapped),pos:Array.from(s.pos)};
   }
   intercepted=true;
  }
 }
 const before=projection(),phiBefore=[phi(0),phi(1)],actions=[[4,0,0],[4,0,0]];
 if(cycleMode){
  if(cycles<5){
   if(!intercepted)actions[enemy][0]=toward(enemy,cage(s.bunCarried[enemy]>=0?enemy:seat));
   if(s.bunCarried[seat]===seat)actions[seat][0]=toward(seat,cage(seat));
   else if(mode==='dropcycles'&&intercepted&&s.bunCarried[enemy]<0){
    const goals=[];for(let z=0;z<195;z++)if(s.bunLoose[z*2+seat]>0)goals.push(z);
    if(goals.length)actions[seat][0]=toward(seat,goals);
   }
  }
 }else if(mode!=='ordinary'){const p=mode==='loss'?enemy:seat;actions[p][0]=toward(p,cage(s.bunCarried[p]>=0?p:1-p));}
 const result=s.step(actions),after=projection(),phiAfter=[phi(0),phi(1)];
 rows.push({injection,before,actions:actions.map(x=>[x[0],0]),after,phiBefore,phiAfter,death:Array.from(result.died)});
 if(cycleMode){
  if(s.bunCarried[enemy]===seat)active=true;
  if(active&&intercepted&&s.bunStored[seat][seat]>0){cycles++;active=false;intercepted=false;}
  if(cycles===5&&++tail>12)break;
 }
 if(mode==='ordinary'&&s.t>=25)break;
}
if(cycleMode&&cycles!==5)throw Error('incomplete cycles '+cycles);
if(!['cycles','dropcycles','ordinary'].includes(mode)&&!s.done)throw Error('missing terminal');
console.log(JSON.stringify({seat,seed,mode,initial,rows,cycles,limitations:['post-acquisition controlled contact fixture; not autonomous evidence','Sim.step fields only; clear-brick map806']}));
