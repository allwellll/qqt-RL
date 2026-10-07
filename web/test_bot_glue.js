'use strict';
const assert=require('assert'),fs=require('fs'),path=require('path'),Q=require('./sim'),Coop=require('./bun_coop_hunter_bot');
const seed=20261007,cell=5*Q.W+9;
function scene({corridor=true,owner=1,threat=false,bombs=true}={}) {
 const s=new Q.Sim(seed);s.reset('open',{nativeItems:true,nativeTrap:true});
 for(const k of ['wall','brick','crate','fuse','blastLinger'])s[k].fill(0);
 s.invuln.fill(0);s.spawnProtection.fill(0);s.isBun=true;s.bunInitial=[Infinity,Infinity];
 s.pos.set([10.5,1.5,5.5,8.9]);
 s.pos[owner*2]=5.5;s.pos[owner*2+1]=9.5;s._addItemToSlots(owner,Q.ITEM_SLOW_GLUE);assert(s._placeHeldItem(owner));
 s.pos.set([10.5,1.5,5.5,8.9]);s._armFieldItems();
 s.crate[5*Q.W+11]=1;s.crateType[5*Q.W+11]=0;
 if(corridor)for(let c=5;c<14;c++){s.wall[4*Q.W+c]=1;s.wall[6*Q.W+c]=1;}
 if(threat){s.wall[4*Q.W+9]=0;s.fuse[4*Q.W+9]=5;s.owner[4*Q.W+9]=0;s.bombBlast[4*Q.W+9]=3;}
 if(!bombs)s.bombsCap[1]=0;
 return s;
}
function run(s,difficulty,ticks=45){
 const b=new Coop.BunCoopHunterBot({difficulty,seed}),initial=Array.from(s.pos.slice(2,4)),trace=[];
 for(let i=0;i<ticks;i++){
  const a=b.act(s,1);trace.push({tick:s.t,pos:Array.from(s.pos.slice(2,4)),action:a.slice(0,3),reason:b.lastDecision.reason,mode:b.lastDecision.mode});
  s.step([[4,0,0,1],a]);assert(s.alive[1]&&!s.trapped[1],'glue decision cannot self-trap or die');
 }
 assert(trace.slice(0,8).some(row=>Math.abs(row.pos[0]-initial[0])+Math.abs(row.pos[1]-initial[1])>.1),'position progresses within 8 ticks');
 return trace;
}
const evidence=[];
for(const difficulty of ['easy','normal','hard'])for(const owner of [0,1]){
 const s=scene({owner}),trace=run(s,difficulty,16);
 if(difficulty!=='easy') {
   assert.equal(s.fieldItem[cell],0,'safe corridor consumes own or opponent glue');
   assert(trace.some(r=>r.pos[1]>10),'route progresses beyond glue rather than stopping at entry');
 } else assert(trace.some(r=>r.pos[1]<7),'easy changes route rather than holding');
 evidence.push({case:'eat',difficulty,owner,trace});
}
{
 const s=scene({corridor:false}),trace=run(s,'hard',16);
 assert.equal(s.fieldItem[cell],2,'available bypass avoids taking unnecessary slow');
 assert(trace.slice(0,8).some(r=>Math.abs(r.pos[0]-5.5)>.2),'bypass makes vertical progress');
 evidence.push({case:'bypass',trace});
}
{
 const s=scene({threat:true});let ownBlastCoversGlue=false;
 const resolve=s._resolveExplosions;
 s._resolveExplosions=function(...args){
  const result=resolve.apply(this,args);
  ownBlastCoversGlue ||= result.sources.some(source=>source.owner===1 && source.covered[cell]);
  return result;
 };
 const trace=run(s,'hard',40);
 assert.equal(trace[0].reason,'bomb_clear_glue');assert.equal(trace[0].action[1],1,'legal bomb placed');
 assert.equal(s.fieldItem[cell],0,'actual explosions clear glue');
 assert(ownBlastCoversGlue,'the Bot bubble really explodes with a ray through the glue cell');
 assert(trace.some(r=>r.tick>=30),'complete fuse horizon verified without self-trapping');
 evidence.push({case:'clear',ownBlastCoversGlue,trace});
}
{
 const s=scene({threat:true,bombs:false}),trace=run(s,'hard',40);
 assert(trace.slice(0,8).some(r=>r.action[0]===2),'no unsafe bomb: retreat immediately');
 assert(trace.slice(0,4).every(r=>r.action[1]===0));evidence.push({case:'cannot-clear-retreat',trace});
}
{
 const s=scene(),snapshot=s.snapshotReplay(),copy=scene();copy.restoreReplay(snapshot);
 assert.deepEqual(run(s,'hard',16),run(copy,'hard',16),'fixed seed snapshot restores identical decisions and progress');
}
if(process.env.EVIDENCE_DIR){fs.mkdirSync(process.env.EVIDENCE_DIR,{recursive:true});fs.writeFileSync(path.join(process.env.EVIDENCE_DIR,'bot-glue.json'),JSON.stringify({seed,evidence},null,2)+'\n');}
console.log('Bot glue seed 20261007: own/enemy eat, bypass, bomb/retreat, multi-tick safety, difficulties and snapshot passed');
module.exports={scene,run};
