'use strict';
// Test harness only. Production Sim and its random stream are never patched.
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'../..');
const Module=require('node:module');
const simPath=path.join(root,'web/sim.js');
let testSource=fs.readFileSync(simPath,'utf8');
const brickOriginal='if (this.rng() < this.crateRate && !this.wall[i]) {\n            const rolled = this._rollCrateType();';
const brickHook='const parityOutcome = this.__parityBrickOutcome ? this.__parityBrickOutcome(i, this.t) : null;\n          if ((parityOutcome ? parityOutcome.type >= 0 : this.rng() < this.crateRate) && !this.wall[i]) {\n            const rolled = parityOutcome || this._rollCrateType();';
if(testSource.split(brickOriginal).length!==2)throw Error('brick hook source signature changed');
testSource=testSource.replace(brickOriginal,brickHook);
const pickupOriginal='const type = this.crateType[cell];';
if(testSource.split(pickupOriginal).length!==2)throw Error('pickup hook source signature changed');
testSource=testSource.replace(pickupOriginal,'let type = this.crateType[cell];\n      if (type < 0 && this.__parityPickupOutcome) type = this.__parityPickupOutcome(player, cell, this.t);');
// Additional optional callbacks exist only in this compiled test module.
const destructionOriginal='const type = this.crateType[i] >= 0 ? this.crateType[i] : Math.floor(this.rng() * 3);';
if(testSource.split(destructionOriginal).length!==2)throw Error('destruction hook signature changed');
testSource=testSource.replace(destructionOriginal,'const type = this.crateType[i] >= 0 ? this.crateType[i] : (this.__parityDestroyAttr ? this.__parityDestroyAttr(i,this.t) : Math.floor(this.rng() * 3));');
const airOriginal='const cell = cells[Math.floor(this.rng() * cells.length)];';
if(testSource.split(airOriginal).length!==2)throw Error('airdrop hook signature changed');
testSource=testSource.replace(airOriginal,'const cell = this.__parityAirCell ? this.__parityAirCell(cells,this.t) : cells[Math.floor(this.rng() * cells.length)];');
const expiryOriginal='if (this.pendingCrateType[i] >= 0) {';
if(testSource.split(expiryOriginal).length!==2)throw Error('expiry hook signature changed');
testSource=testSource.replace(expiryOriginal,'if(this.__parityBrickExpiry){const outcome=this.__parityBrickExpiry(i,this.t);this.pendingCrateType[i]=outcome.type;this.pendingSuperCrate[i]=outcome.isSuper?1:0;}\n          '+expiryOriginal);
const testModule=new Module(simPath,module);testModule.filename=simPath;
testModule.paths=module.paths;testModule._compile(testSource,simPath);
const Q=testModule.exports;
const levels=JSON.parse(fs.readFileSync(path.join(root,'web/assets/maps/levels.json')));
const level=levels.find(x=>x.qqt_id===806);
if(process.argv[2]==='--validate-disabled'){
  const production=require(simPath);let ticks=0;
  for(const seed of [17,2026122600,2026132600]){
    const a=new production.Sim(seed),b=new Q.Sim(seed);
    a.reset(level,{nativeTrap:true,nativeItems:true});b.reset(level,{nativeTrap:true,nativeItems:true});
    for(let t=0;t<900;t++){
      if(JSON.stringify(a.snapshotReplay())!==JSON.stringify(b.snapshotReplay()))throw Error('disabled hook changed state at '+t);
      const actions=[[t%5,t%23===0?1:0,t%13===0?1:0],[(t+2)%5,t%19===0?1:0,t%17===0?1:0]];
      a.step(actions);b.step(actions);ticks++;
    }
  }
  process.stdout.write(JSON.stringify({passed:true,seeds:[17,2026122600,2026132600],ticks,production_code_modified:false})+'\n');process.exit(0);
}
const input=JSON.parse(fs.readFileSync(process.argv[2]));
const sim=new Q.Sim(input.seed);
sim.reset(level,{nativeTrap:true,nativeItems:true});
sim.restoreReplay(input.initial_frame);
sim.initialHp=input.initial_frame.hp[0];
const hookLog=[];
let currentUnalignedRngCalls=0,totalUnalignedRngCalls=0;
const originalRng=sim.rng;
sim.rng=()=>{currentUnalignedRngCalls++;totalUnalignedRngCalls++;hookLog.push({event:'unhooked_rng',tick:sim.t,stack:new Error().stack.split('\n').slice(2,5).map(line=>line.replaceAll(root+'/', '').replace(/\/[^ ()]+\/([^/ ()]+:\d+:\d+)/g,'$1'))});return originalRng();};
const unalignedRngLog=[];
// Passive instance instrumentation records events without changing return values.
const eventNames=['bomb_place','explosion_chain','control/trap','death','respawn','item_spawn',
  'item_pickup','banana_use','glue_use','bag_drop','bag_pickup','bag_transfer','delivery'];
const eventLog=[];
let observed={};
function observeMethod(name,observer){
  const original=sim[name];
  sim[name]=function(...args){return observer.call(this,original,args);};
}
observeMethod('_bunDrop',function(original,args){
  if(this.bunCarried[args[0]]>=0)observed.bag_drop=true;
  return original.apply(this,args);
});
observeMethod('_collectCrate',function(original,args){
  const p=args[0],before=[this.bombsCap[p],this.blastCap[p],this.spdG[p]],
    inventory=this.itemSlots[p].reduce((sum,x)=>sum+x.count,0);
  const result=original.apply(this,args);
  if(result)observed.item_pickup=true;
  return result;
});
observeMethod('_bunUpdate',function(original,args){
  const carried=this.bunCarried.slice(),stored=this.bunStored.map(x=>x.slice());
  const result=original.apply(this,args);
  for(let p=0;p<2;p++){
    if(carried[p]<0&&this.bunCarried[p]>=0)observed.bag_pickup=true;
    if(this.bunStored[p][1-p]>stored[p][1-p])observed.delivery=true;
  }
  return result;
});
observeMethod('_trapStep',function(original,args){
  const carried=this.bunCarried.slice(),alive=this.alive.slice();
  const result=original.apply(this,args);
  for(let victim=0;victim<2;victim++){
    const p=1-victim;
    if(alive[victim]&&!this.alive[victim]&&carried[victim]>=0&&carried[p]<0&&
       this.bunCarried[p]===carried[victim])observed.bag_transfer=true;
  }
  return result;
});
if(input.rng_tape){
  sim.__parityBrickExpiry=(cell,t)=>{
    const outcome={type:input.rng_tape.brick_drop_kind[t][cell],isSuper:!!input.rng_tape.brick_drop_super[t][cell]};
    hookLog.push({event:'brick_expiry',tick:t+1,cell,outcome});return outcome;
  };
  sim.__parityBrickOutcome=(cell,t)=>{
    const expiry=t+4;
    if(expiry>=input.rng_tape.brick_drop_kind.length)return {type:-1,isSuper:false};
    const type=input.rng_tape.brick_drop_kind[expiry][cell];
    const outcome={type,isSuper:!!input.rng_tape.brick_drop_super[expiry][cell]};
    hookLog.push({event:'brick_drop',tick:t+1,expiry_tick:expiry+1,cell,outcome});return outcome;
  };
  sim.__parityPickupOutcome=(player,cell,t)=>{
    const idx=Math.max(0,t-1),attr=input.rng_tape.unknown_pickup_attr[idx][player];
    hookLog.push({event:'unknown_pickup',tick:t,cell,player,attr});return attr;
  };
}
if(input.rng_tape && input.rng_tape.native_destroyed_unknown_attr){
  sim.__parityDestroyAttr=(cell,t)=>{
    const attr=input.rng_tape.native_destroyed_unknown_attr[t][cell];
    hookLog.push({event:'destroy_unknown',tick:t+1,cell,attr});return attr;
  };
  let airTick=-1,airIndex=0;
  sim.__parityAirCell=(cells,t)=>{
    if(t!==airTick){airTick=t;airIndex=0;}
    const cell=input.rng_tape.native_air_launch_cells[t-1][airIndex++];
    if(!cells.includes(cell)){const error=new Error('air launch candidates disagree');error.parity={tick:t,field:'rng.air_launch_candidates',jax:cell,js:cells,within_tolerance:false,tolerance:0,class:'PHYSICS_MISMATCH'};throw error;}
    hookLog.push({event:'air_launch',tick:t,cell,candidates:cells.length});return cell;
  };
}
const flat=x=>Array.from(x).flat(Infinity).map(Number);
function state(){
  const d={t:[sim.t],pos:flat(sim.pos),alive:flat(sim.alive),hp:flat(sim.hp),
    bun_carried:flat(sim.bunCarried),bun_stored:flat(sim.bunStored),bun_loose:flat(sim.bunLoose),
    bun_respawn:flat(sim.bunRespawn),held_item:flat(sim.heldItem),field_item:flat(sim.fieldItem),
    field_owner:flat(sim.fieldOwner),field_armed:flat(sim.fieldArmed),move_status:flat(sim.movementStatus),
    status_ticks:flat(sim.movementStatusTicks),slide_dir:flat(sim.slideDir),last_move_dir:flat(sim.lastMoveDir),
    trapped:flat(sim.trapped),bun_spawn_pos:flat(sim.bunSpawnPos),
    bomb_blast:flat(sim.bombBlast),blast_linger:Array.from(sim.horzLinger,(h,i)=>(h<<4)|sim.vertLinger[i]),brick_linger:flat(sim.brickLinger),
    push_t:flat(sim.pushT),rec_crate:flat(sim.recycle),invuln:flat(sim.invuln),
    bombs_cap:flat(sim.bombsCap),blast_cap:flat(sim.blastCap),spd_g:flat(sim.spdG)};
  for(const n of ['fuse','owner','wall','brick','pushable','bush'])d[n]=flat(sim[n]);
  d.item_slots=[];d.item_counts=[];
  for(let p=0;p<2;p++)for(let i=0;i<7;i++){
    const slot=sim.itemSlots[p][i];d.item_slots.push(slot?slot.item:0);d.item_counts.push(slot?slot.count:0);
  }
  d.crate=[];d.tactical_crate=[];
  for(let i=0;i<Q.H*Q.W;i++){
    const exists=sim.crate[i]>0,type=sim.crateType[i];
    d.tactical_crate.push(exists&&type>=3&&type<=4?type-2:0);
    d.crate.push(!exists||type>=3?0:type<0?7:1+type+(sim.superCrate[i]?3:0));
  }
  if(input.states.native_crate_counts)d.native_crate_counts=flat(sim.crateCount);
  if(input.states.spawn_protection)d.spawn_protection=flat(sim.spawnProtection);
  return d;
}
const floats=new Set(['pos','bun_spawn_pos','push_t','bombs_cap','blast_cap','spd_g']);
const randomFields=new Set(['crate','tactical_crate','hp','bombs_cap','blast_cap','spd_g','item_slots','item_counts','held_item']);
const fields=Object.keys(state());
let first=null,maxPositionError=0,ticks=0,last=null,firstHidden=null;
for(let t=0;t<=input.actions.length;t++){
  const actual=state();ticks++;
  for(const field of fields){
    const expected=input.states[field][t],got=actual[field];
    if(expected.length!==got.length)throw Error('ABI shape mismatch: '+field);
    const atol=floats.has(field)?2e-5:0;
    for(let i=0;i<got.length;i++){
      const error=Math.abs(Number(expected[i])-got[i]);
      if(field==='pos')maxPositionError=Math.max(maxPositionError,error);
      if(error>atol){
        first={tick:t,field,index:i,jax:expected[i],js:got[i],within_tolerance:false,
          absolute_error:error,tolerance:atol,class:randomFields.has(field)&&(!input.rng_tape||currentUnalignedRngCalls>0)?'RNG_UNALIGNED':'PHYSICS_MISMATCH'};
        break;
      }
    }
    if(first)break;
  }
  if(first){first.context={expected:Object.fromEntries(['move_status','status_ticks','slide_dir','last_move_dir','trapped','invuln','pos'].map(n=>[n,input.states[n][t]])),js:Object.fromEntries(['move_status','status_ticks','slide_dir','last_move_dir','trapped','invuln','pos'].map(n=>[n,actual[n]]))};}
  if(first||t===input.actions.length)break;
  observed={};
  currentUnalignedRngCalls=0;
  const before=state(),held=sim.heldItem.slice();
  // JAX ability 2 maps to the production JS item-release slot.
  const joint=input.actions[t].map(a=>[a[0],a[1]===1?1:0,a[1]===2?1:0]);
  try {last=sim.step(joint);} catch(error){if(!error.parity)throw error;first=error.parity;ticks++;break;}
  if(currentUnalignedRngCalls>0)unalignedRngLog.push({tick:t+1,calls:currentUnalignedRngCalls});
  if(input.native_debug&&!firstHidden){
    const pack=q=>Array.from({length:16},(_,i)=>{const d=q[i];return d?[d.type,d.isSuper?1:0,d.count||0]:[0,0,0];}).flat();
    const actualHidden=[sim.graveyard.length,sim._airdropQueue.length,sim.airdropTotal,sim.airdropDropped,...pack(sim.graveyard),...pack(sim._airdropQueue)];
    const expect=input.native_debug[t];
    // Unused queue storage after size is deliberately ignored.
    const indices=[0,1,2,3,...Array.from({length:Math.min(16,expect[0])*3},(_,i)=>i+4),...Array.from({length:Math.min(16,expect[1])*3},(_,i)=>i+52)];
    if(indices.some(i=>expect[i]!==actualHidden[i])){firstHidden={tick:t+1,jax:expect,js:actualHidden};first={tick:t+1,field:'native_item_queue',jax:expect,js:actualHidden,within_tolerance:false,tolerance:0,class:'PHYSICS_MISMATCH'};ticks++;break;}
  }
  if(input.event_flags){
    const after=state(),some=x=>x.some(Boolean);
    observed.bomb_place=some(last.placed);
    observed.explosion_chain=Array.from(last.triggered).some((v,i)=>v&&(before.fuse[i]>1||last.placed.some((placed,p)=>placed&&Math.floor(before.pos[2*p])*Q.W+Math.floor(before.pos[2*p+1])===i)));
    observed['control/trap']=sim.trapped.some((v,p)=>v>0&&before.trapped[p]===0);
    observed.death=some(last.died);
    observed.respawn=sim.alive.some((v,p)=>v&&!before.alive[p]);
    observed.item_spawn=after.crate.some((v,i)=>v>0&&before.crate[i]===0)||
      after.tactical_crate.some((v,i)=>v>0&&before.tactical_crate[i]===0);
    observed.banana_use=last.itemReleased.some((v,p)=>v&&held[p]===1);
    observed.glue_use=last.itemReleased.some((v,p)=>v&&held[p]===2);
    const expected=input.event_flags[t];
    const jsFlags=eventNames.map(n=>Boolean(observed[n]));
    if(expected.some(Boolean)||jsFlags.some(Boolean))eventLog.push({tick:t+1,jax:expected.map(Boolean),js:jsFlags});
    for(let i=0;i<eventNames.length;i++)if(Boolean(expected[i])!==Boolean(observed[eventNames[i]])){
      first={tick:t+1,field:'event.'+eventNames[i],jax:Boolean(expected[i]),js:Boolean(observed[eventNames[i]]),
        context:{before,after,expectedAfter:Object.fromEntries(Object.keys(after).map(n=>[n,input.states[n][t+1]])),joint},within_tolerance:false,tolerance:0,class:['item_spawn','item_pickup'].includes(eventNames[i])&&currentUnalignedRngCalls>0?'RNG_UNALIGNED':'PHYSICS_MISMATCH'};
      break;
    }
    if(first){ticks++;break;}
  }
}
fs.writeFileSync(process.argv[3],JSON.stringify({schema:'qqt.js-event-parity-comparison/v1',
  event_log:eventLog,first_hidden_divergence:firstHidden,ticks_compared:ticks,first_divergence:first,max_position_error:maxPositionError,
  compared_fields:fields,compared_event_types:input.event_flags?eventNames:[],
  rng_override_hook:!!input.rng_tape,rng_hook_log:hookLog,
  unaligned_rng_calls:totalUnalignedRngCalls,unaligned_rng_call_log:unalignedRngLog,
  initial_state_via:'existing Sim.restoreReplay at tick 0 only',
  production_code_modified:false,hidden_final:{graveyard:sim.graveyard,pending_deaths:sim._pendingDeathDrops,falls:sim.airdropFalls}},null,2)+'\n');
