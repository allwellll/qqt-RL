'use strict';
// P7 public-attribute finite constant-action witness. No Sim.step, RNG, center projection or BFS.
const Q=require('../web/sim-bun-native-v2');
const MAX_HORIZON=64, N=195, W=15, H=13;
const REASONS=['ok','missing_public_metadata','unsupported_rules','inactive_or_status',
 'opponent_not_absent','invalid_bomb','horizon_out_of_bound','opponent_respawns',
 'dynamic_pushable','unproven_death_drop_queue_clear','airdrop_random_window','episode_boundary'];
const arr=a=>Array.from(a,Number);
function metadata(sim){
 return {schema:3,verifiedRules:sim instanceof Q.Sim&&sim.isBun&&sim.nativeItems===true&&sim.nativeTrap===true,
  newSteering:(sim.playerModes||[]).length===0||(sim.playerModes||[]).every(x=>x==='new'),
  respawn:arr(sim.bunRespawn),pushable:arr(sim.pushable),visibleCover:arr(sim.cover),
  airdropFalls:(sim.airdropFalls||[]).map(x=>x.cell),
  crateRate:Number(sim.crateRate),maxSteps:Number(sim.maxSteps),growth:arr(sim.spdG),bombsCap:arr(sim.bombsCap),blastCap:arr(sim.blastCap),superCrate:arr(sim.superCrate),crateCount:arr(sim.crateCount),caps:[Number(sim.bombsMax),Number(sim.blastMax),Number(sim.speedMax)],speedStep:Number(sim.speedStep||Q.CFG.growthSpeedStep)};
}
function eligibility(s,pid){
 const m=s.knownThreatWitness,me=s.players[pid],foe=s.players[1-pid];
 const no=(reason,horizon=0)=>({applicable:false,reason,horizon});
 if(!m||m.schema!==3)return no('missing_public_metadata');
 if(!m.verifiedRules||!m.newSteering||s.height!==H||s.width!==W||s.players.length!==2||s.players.some((p,i)=>(p.team??i)!==i))return no('unsupported_rules');
 if(!me.alive||me.trapped>0||me.invuln>0||me.carrying<0||me.movementStatus!==0||!Number.isFinite(me.speed)||me.speed<=0||me.speed>2.4)return no('inactive_or_status');
 if(foe.alive)return no('opponent_not_absent');
 if(!m.growth||!m.caps||!m.superCrate||!m.crateCount||!Number.isFinite(m.speedStep)||Math.abs(m.speedStep-1.1/7)>1e-12||m.growth.some(x=>!Number.isFinite(x)||x<=0))return no('missing_attribute_metadata');
 if(s.bombs.some(b=>!Number.isInteger(b.fuse)||b.fuse<1||b.owner<0||b.owner>1||b.cell<0||b.cell>=N||!Number.isInteger(b.blast)||b.blast<1||b.blast>8))return no('invalid_bomb');
 const horizon=Math.max(1,...s.bombs.map(b=>b.fuse+Q.CFG.brickLingerTicks),...s.blastLinger,...s.brickLinger);
 if(!Number.isInteger(horizon)||horizon>MAX_HORIZON)return no('horizon_out_of_bound',horizon);
 if(!(m.respawn[1-pid]>horizon))return no('opponent_respawns',horizon);
 if(m.pushable.some(Boolean))return no('dynamic_pushable',horizon);
 // At a completed logical tick, _releaseDeathDrops drains every pending drop
 // whenever an available cell exists. Later pickups are only on the live
 // player's cell (not available), and bird events cannot create an available
 // cell. Thus one currently free public cell proves no pending queue remains.
 // Do not inspect that hidden queue or its count.
 const occupied=new Set(s.players.filter(p=>p.alive).map(p=>p.cell));
 const bombs=new Set(s.bombs.map(b=>b.cell)),falls=new Set(m.airdropFalls);
 const free=s.tick>0&&Array.from({length:N},(_,i)=>i).some(i=>!s.wall[i]&&!s.brick[i]&&!m.pushable[i]&&!m.visibleCover[i]&&!s.crate[i]&&!s.fieldItem[i]&&!bombs.has(i)&&!s.blastLinger[i]&&!falls.has(i)&&!occupied.has(i));
 if(!free)return no('unproven_death_drop_queue_clear',horizon);
 // Public clock alone: never inspect graveyard/flight payload to pass a gate.
 if(Array.from({length:horizon},(_,k)=>(s.tick+k+1)%300).some(t=>t>=270))return no('airdrop_random_window',horizon);
 if(!(m.maxSteps>s.tick+horizon))return no('episode_boundary',horizon);
 return {applicable:true,reason:'ok',horizon};
}
function timeline(s,horizon){
 const f={wall:Uint8Array.from(s.wall),brick:Uint8Array.from(s.brick),fuse:new Int16Array(N),owner:new Int16Array(N).fill(-1),bombBlast:new Int16Array(N),
  _rays:Q.Sim.prototype._rays};
 const linger=Int16Array.from(s.brickLinger),uncertain=new Uint8Array(N),rows=[];
 const crates=Uint8Array.from(s.crate),fields=Uint8Array.from(s.fieldItem);
 for(const b of s.bombs){f.fuse[b.cell]=b.fuse;f.owner[b.cell]=b.owner;f.bombBlast[b.cell]=b.blast;}
 for(let i=0;i<N;i++)if(s.brickLinger[i]>0)uncertain[i]=1;
 for(const c of s.knownThreatWitness.airdropFalls)uncertain[c]=1;
 for(let tick=1;tick<=horizon;tick++){
  for(let i=0;i<N;i++)if(f.fuse[i]>0)f.fuse[i]--;
  const blocked=Uint8Array.from(f.wall,(w,i)=>!!(w||f.brick[i]||f.fuse[i]>0));
  const {covered,triggered}=Q.Sim.prototype._resolveExplosions.call(f),coveredBrick=new Uint8Array(N);
  for(let i=0;i<N;i++){
   coveredBrick[i]=covered[i]&&f.brick[i]?1:0;
   // Newly rolled loot is unknown, but its only possible position is public.
   // Reject every path entering that cell; no RNG is read or predicted.
   if(coveredBrick[i]&&linger[i]===0&&s.knownThreatWitness.crateRate>0)uncertain[i]=1;
  }
  const fieldBefore=fields.slice();
  for(let i=0;i<N;i++)if(covered[i]){crates[i]=0;fields[i]=0;}
  rows.push({tick,blocked,covered,triggered,uncertain:uncertain.slice(),crates:crates.slice(),fieldBefore});
  for(let i=0;i<N;i++){
   const old=linger[i];linger[i]=coveredBrick[i]?Q.CFG.brickLingerTicks:Math.max(0,old-1);
   if(old>0&&linger[i]===0&&!coveredBrick[i])f.brick[i]=0;
   if(triggered[i]){f.fuse[i]=0;f.owner[i]=-1;f.bombBlast[i]=0;}
  }
 }
 return rows;
}
function contactCells(pos){return Q.Sim.prototype._explosionContactCells.call({pos},0);}
function hit(pos,covered){return contactCells(pos).every(c=>!!covered[c]);}
function movement(pos,move,blocked,speed){
 const ctx={_tryMove:Q.Sim.prototype._tryMove,playerModes:['new'],pushable:new Uint8Array(N)};
 const p=Q.Sim.prototype._steer.call(ctx,pos[0],pos[1],move,blocked,Q.CFG.stepLen*speed,0);
 return [Math.min(Math.max(p[0],Q.CFG.radius),H-Q.CFG.radius),Math.min(Math.max(p[1],Q.CFG.radius),W-Q.CFG.radius)];
}
function carryDist(growth){return Q.CFG.stepLen*growth*(72/(Q.CFG.speed*growth*Q.NATIVE_CELL_PX));}
function movementWithGrowth(pos,move,blocked,growth){
 const ctx={_tryMove:Q.Sim.prototype._tryMove,playerModes:['new'],pushable:new Uint8Array(N)};
 const p=Q.Sim.prototype._steer.call(ctx,pos[0],pos[1],move,blocked,carryDist(growth),0);
 return [Math.min(Math.max(p[0],Q.CFG.radius),H-Q.CFG.radius),Math.min(Math.max(p[1],Q.CFG.radius),W-Q.CFG.radius)];
}
function collectAttribute(attributes,type,isSuper,count,caps,speedStep){
 if(!Number.isInteger(type)||type<0||type>2||!Number.isInteger(count)||count<0||count>1024)return null;
 const a=attributes.slice(),times=count||1,add=isSuper?5:1;
 // Same addition order as native _grow, including the speed cap tolerance.
 for(let k=0;k<times;k++){
  if(type===2){if(a[2]<caps[2]-1e-4)a[2]=Math.min(a[2]+add*speedStep,caps[2]);}
  else if(a[type]<caps[type])a[type]=Math.min(a[type]+add,caps[type]);
 }
 return a;
}
function witness(s,pid,{trajectories=false}={}){
 const e=eligibility(s,pid),result={...e,certifiedMask:0,selectedMove:-1,paths:[]};
 if(!e.applicable)return result;
 const frames=timeline(s,e.horizon),me=s.players[pid],m=s.knownThreatWitness,base=s.bunBases[me.team??pid];
 for(let move=0;move<5;move++){
  let p=[me.y,me.x],attributes=[m.bombsCap[pid],m.blastCap[pid],m.growth[pid]],reason='safe_known_horizon',firstHit=-1;const path=[],picked=new Uint8Array(N);
  for(const f of frames){
   const before=p.slice();p=movementWithGrowth(p,move,f.blocked,attributes[2]);
   const r=Math.floor(p[0]),c=Math.floor(p[1]),cell=r*W+c,startCell=Math.floor(before[0])*W+Math.floor(before[1]);
   let unsupported=f.fieldBefore[startCell]||f.fieldBefore[cell]?'field_status':f.uncertain[cell]?'unknown_future_drop':r>=base[0]&&r<base[0]+3&&c>=base[1]&&c<base[1]+3?'carry_settlement':null;
   const struck=hit(p,f.covered);let pickup=null;
   if(!unsupported&&!struck&&f.crates[cell]&&!picked[cell]){
    const next=collectAttribute(attributes,s.crateType[cell],m.superCrate[cell]===1,m.crateCount[cell],m.caps,m.speedStep);
    if(!next)unsupported='unmodeled_crate';
    else{pickup={cell,type:s.crateType[cell],super:m.superCrate[cell]===1,count:m.crateCount[cell]||1,before:attributes.slice(),after:next.slice()};attributes=next;picked[cell]=1;}
   }
   if(trajectories)path.push({tick:f.tick,pos:p.slice(),contact:contactCells(p),hit:struck,unsupported,pickup,attributes:attributes.slice(),carrySpeed:attributes[2]*(72/(Q.CFG.speed*attributes[2]*Q.NATIVE_CELL_PX))});
   if(unsupported){reason=unsupported;break;}
   if(struck){reason='known_bomb_hit';firstHit=f.tick;break;}
  }
  const safe=reason==='safe_known_horizon';if(safe)result.certifiedMask|=1<<move;
  result.paths.push({move,safe,reason,firstHit,finalPos:p,finalAttributes:attributes,...(trajectories?{trajectory:path}:{})});
 }
 if(result.certifiedMask)result.selectedMove=result.paths.find(x=>x.safe).move;
 return result;
}
module.exports={MAX_HORIZON,REASONS,metadata,eligibility,timeline,contactCells,hit,movement,carryDist,movementWithGrowth,collectAttribute,witness};
