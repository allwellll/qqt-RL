'use strict';
// CPU-only frozen-state encoder/logit witness. No training or production edits.
const fs=require('fs'),Q=require('../web/sim-stock-window-v1'),A=require('./eval_web_hunter');
const input=JSON.parse(fs.readFileSync(process.argv[2])),rows=[];
const modelDoc=JSON.parse(fs.readFileSync(input.model)),model=new Q.TransformerModel(modelDoc);
const sw=(a)=>a.map((_,i)=>a[1-i]);
function exchange(s){
 const t=new Q.Sim(1);t.reset(A.loadLevel(),{nativeTrap:true,nativeItems:true});t.restoreReplay(s.snapshotReplay());
 const fields=['alive','hp','invuln','bombsCap','blastCap','spdG','trapped','bunRespawn','heldItem','movementStatus','bunSpawnPos'];
 for(const k of fields)t[k]=sw(Array.from(s[k]));
 t.pos.set([s.pos[2],s.pos[3],s.pos[0],s.pos[1]]);t.bunBases=sw(s.bunBases);
 t.bunCarried=s.bunCarried.slice().reverse().map(x=>x<0?x:1-x);
 t.bunStored=s.bunStored.slice().reverse().map(x=>x.slice().reverse());
 for(let i=0;i<195;i++){
  if(t.owner[i]>=0)t.owner[i]=1-t.owner[i];if(t.fieldOwner[i]>=0)t.fieldOwner[i]=1-t.fieldOwner[i];
  t.bunLoose[2*i]=s.bunLoose[2*i+1];t.bunLoose[2*i+1]=s.bunLoose[2*i];
 }t._gen++;return t;
}
for(const r of input.rows){
 const s=new Q.Sim(r.seed);s.reset(A.loadLevel(),{nativeTrap:true,nativeItems:true});s.maxSteps=2400;
 for(const [k,v]of Object.entries(r.js)){
  if(ArrayBuffer.isView(s[k]))s[k].set(v.flat(Infinity));else s[k]=v;
 }
 for(let i=0;i<195;i++){const c=r.crate[i];s.crateType[i]=c>0&&c<7?(c-1)%3:-1;s.superCrate[i]=Number(c>=4&&c<=6);if(r.tactical_crate[i]>0){s.crate[i]=1;s.crateType[i]=r.tactical_crate[i]===1?3:4;}}
 s._gen++;const original=[],relabeled=[];const exchanged=exchange(s);
 for(let pid=0;pid<2;pid++){
  const x=model._getLaggedInputs(s,pid),z=model.forward(x.obs,x.state);
  const y=model._getLaggedInputs(exchanged,1-pid),zz=model.forward(y.obs,y.state);
  original.push({obs:Array.from(x.obs),state:Array.from(x.state),move:Array.from(z.move),ability:Array.from(z.bomb)});
  relabeled.push({obs_max:Math.max(...x.obs.map((v,i)=>Math.abs(v-y.obs[i]))),state_max:Math.max(...x.state.map((v,i)=>Math.abs(v-y.state[i]))),
   move_max:Math.max(...z.move.map((v,i)=>Math.abs(v-zz.move[i]))),ability_max:Math.max(...z.bomb.map((v,i)=>Math.abs(v-zz.bomb[i])))});
 }
 rows.push({id:r.id,original,relabeled});
}
const directions=[];
for(let pid=0;pid<2;pid++)for(let move=0;move<5;move++){
 const s=new Q.Sim(17);s.reset(A.loadLevel(),{nativeTrap:true,nativeItems:true});
 for(const key of ['wall','brick','pushable','fuse','blastLinger','crate','bush','fieldItem','fieldArmed'])s[key].fill(0);
 s.pos.set([5.5,6.5,7.5,8.5]);s.spdG.fill(1);s._gen++;
 const start=[s.pos[2*pid],s.pos[2*pid+1]],actions=[[4,0],[4,0]];actions[pid]=[move,0];s.step(actions);
 const delta=[s.pos[2*pid]-start[0],s.pos[2*pid+1]-start[1]];
 const expected=[[-1,0],[1,0],[0,-1],[0,1],[0,0]][move];
 if(!delta.every((v,i)=>Math.abs(v)<1e-8?expected[i]===0:Math.sign(v)===expected[i]))throw Error('seat action direction mismatch');
 directions.push({seat:pid,move,delta,expected});
}
fs.writeFileSync(process.argv[3],JSON.stringify({rows,directions,engine:'web/app.js -> TransformerModel._getLaggedInputs/forward; stock-window-v1',
 mapping:'absolute spatial coordinates and up/down/left/right/stay; exchange identities/origins/cages, no spatial rotation'}));
