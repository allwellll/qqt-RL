'use strict';
// Explicit schema dispatch. A seven-action policy always uses its own consumer.
const fs=require('node:fs'),assert=require('node:assert/strict');
const seven=require('./five_item_slots_inference'),legacy=require('./r3_items_inference');
async function loadModel(file){
 const meta=JSON.parse(fs.readFileSync(file)).meta;
 if(meta.ability_actions===7)return seven.loadModel(file);
 assert.equal(meta.ability_actions,3);assert.equal(meta.state_dim,37);
 const loaded=await legacy.loadModel(file),m=loaded.model;
 const probability=(v,mask)=>{const mx=Math.max(...Array.from(v,(x,i)=>mask[i]?x:-Infinity)),p=Array.from(v,(x,i)=>mask[i]?Math.exp(x-mx):0),z=p.reduce((a,b)=>a+b,0);assert(z>0);return p.map(x=>x/z);};
 m.act=async(s,p,rng)=>{const x=legacy.inputs(s,p),mask=s.legalMask(),v=await m.forward(x.obs,x.state),am=m._abilityMask(s,mask,p);
  const move=m._sampleMasked(v.move,mask.mm[p],rng),ability=m._sampleMasked(v.bomb,am,rng);
  m.lastDecision={move,ability,move_mask:Array.from(mask.mm[p]),ability_mask:Array.from(am),move_probability:probability(v.move,mask.mm[p]),ability_probability:probability(v.bomb,am)};
  return m._decodeAbility(move,ability);};
 loaded.session={release:async()=>{}};return loaded;
}
module.exports={loadModel};
