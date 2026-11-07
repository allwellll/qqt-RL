'use strict';
// Exercise authoritative engine methods, never a reimplementation of them.
const fs=require('fs'),path=require('path');
const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const Q=require(path.resolve(input.engine));
const results=input.cases.map(c=>{
  const s=new Q.Sim(c.seed||7);s.reset('open',{nativeItems:true,nativeTrap:true});
  s.isBun=true;
  for(const k of ['wall','brick','pushable','cover','crate','fieldItem','fuse','blastLinger','brickLinger','bunLoose'])s[k].fill(0);
  s.pendingCrateType.fill(-1);s.airdropFalls=[];s._pendingDeathDrops=[];
  for(const [k,v] of Object.entries(c.fields||{}))s[k].set(v.flat(Infinity));
  s.pos.set(c.pos.flat());s.alive=c.alive.slice();s.trapped=(c.trapped||[0,0]).slice();
  s.crateType.fill(-1);s.superCrate.fill(0);s.crateCount.fill(0);
  for(let i=0;i<Q.N;i++){
    const code=c.crate.flat()[i],t=c.tactical.flat()[i];
    if(code||t){s.crate[i]=1;s.crateType[i]=t?t+2:(code-1)%3;s.superCrate[i]=code>=4&&code<=6?1:0;s.crateCount[i]=c.counts.flat()[i];}
  }
  s.airdropFalls=(c.falls||[]).map(cell=>({cell}));
  const legal=Array.from({length:Q.N},(_,i)=>Number(s._dropCellAvailable(i)));
  if(c.kind==='death'){
    let n=0;s.rng=()=>c.draws[n++];
    s._pendingDeathDrops=c.drops.map(d=>({type:d[0],isSuper:!!d[1],count:d[2],row:d[3],col:d[4]}));
    s._releaseDeathDrops();
  }else{
    s.crateRate=c.rate;s.itemsEnabled=c.enabled;s.itemDuel=!!c.duel;
    s.superFraction=c.super_fraction;s.tacticalItemFraction=c.tactical_fraction;
    const tape=[];
    for(const d of c.draws){tape.push(d[0]);if(s.tacticalItemFraction>0)tape.push(d[1]);tape.push(d[2]);if(!(d[1]<s.tacticalItemFraction))tape.push(d[3]);}
    let n=0;s.rng=()=>tape[n++];s._applyItemSupply();
  }
  return {legal,crate:Array.from(s.crate,(_,i)=>s.crate[i]&&s.crateType[i]<3?1+s.crateType[i]+3*s.superCrate[i]:0),
    tactical:Array.from(s.crate,(_,i)=>s.crate[i]&&s.crateType[i]>=3?s.crateType[i]-2:0),
    counts:Array.from(s.crateCount),pending:s._pendingDeathDrops.map(d=>[d.type,Number(!!d.isSuper),d.count]),supply:s.itemSupply};
});
fs.writeFileSync(process.argv[3],JSON.stringify(results));
