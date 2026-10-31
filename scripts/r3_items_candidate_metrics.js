'use strict';
const base=require('./r3_items_metrics');
const crypto=require('node:crypto');
function install(Q){
 base.install(Q);const P=Q.Sim.prototype,explode=P._resolveExplosions;
 P._resolveExplosions=function(...args){
  const brick=Array.from(this.brick),crate=Array.from(this.crate),pos=Array.from(this.pos),alive=Array.from(this.alive),r=explode.apply(this,args);
  if(this.__btIdentity){this.__r3Resolved ||= [[],[]];for(const b of r.sources||[])if(b.owner>=0&&b.owner<2){
   const enemy=1-b.owner,cell=Math.floor(pos[2*enemy])*15+Math.floor(pos[2*enemy+1]);
   this.__r3Resolved[b.owner].push({tick:this.t,cell:b.cell,empty:!Array.from(b.covered).some((v,i)=>v&&(brick[i]||crate[i]||(alive[enemy]&&i===cell))),cause_mask:b.causeMask});}}
  return r;
 };
 const step=P.step;P.step=function(...args){if(this.__btIdentity){this.__r3Actions ||= [];this.__r3Actions.push(args[0].map(a=>Array.from(a)));}
  const r=step.apply(this,args);if(this.done&&this.__bt?.r3){this.__bt.r3.resolved_bombs=this.__r3Resolved||[[],[]];this.__bt.r3.joint_actions_sha256=crypto.createHash('sha256').update(JSON.stringify(this.__r3Actions)).digest('hex');}return r;};
}
module.exports={install};
