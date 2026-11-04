'use strict';
(function(root,factory){const node=typeof module==='object'&&module.exports;const api=factory(node?require('./five-item-slots-v1'):root.QQTFiveSlots);if(node)module.exports=api;else root.QQTFiveSlotsModel=api;})(typeof globalThis!=='undefined'?globalThis:this,function(S){
function validate(meta){
 if(meta.schema_version!=='five-item-slots-v1'||meta.action_schema!=='move-v1+five-item-slots-v1'||meta.ability_actions!==7||meta.state_dim!==52||meta.global_dim!==52||JSON.stringify(meta.obs_shape)!=='[25,13,15]'||JSON.stringify(meta.ability_encoding)!==JSON.stringify(S.ACTION_NAMES))throw Error('explicit five-slot metadata required');
 return meta;
}
function inputs(sim,p){
 const obs=new Float32Array(25*195);obs.set(sim.encodeObsJAX(p,24));
 const bombs=sim.__r3?.bombs||sim.__r3Visible?.bombs||new Map(),live=[];
 for(let i=0;i<195;i++)if(sim.fuse[i]>0&&sim.owner[i]===p)live.push(i);
 const batch=live.map(i=>bombs.get(i)?.batch??-1),ids=batch.filter(x=>x>=0),minimum=ids.length?Math.min(...ids):0;
 for(let k=0;k<live.length;k++)if(batch[k]>=0)obs[24*195+live[k]]=(batch[k]-minimum+1)/8;
 const state=new Float32Array(52),base=sim.encodeStateJAX(p);base[9]+=sim.trapped[p]/60;base[10]+=sim.trapped[1-p]/60;state.set(base);
 let k=24;const enemy=1-p;
 state[k++]=Math.min(1,sim.itemSlots[p].reduce((a,b)=>a+b.count,0)/50);
 for(const type of [1,2])state[k++]=Number(sim.itemSlots[enemy].some(x=>x.item===type&&x.count>0));
 for(const player of [p,enemy])for(let type=0;type<4;type++)state[k++]=Number(sim.movementStatus[player]===type);
 for(const player of [p,enemy])state[k++]=sim.movementStatusTicks[player]/100;
 for(let i=0;i<5;i++)state[k++]=(sim.itemSlots[p][i]?.item||0)/8;
 for(let i=0;i<5;i++)state[k++]=Math.min(32,sim.itemSlots[p][i]?.count||0)/32;
 for(let i=0;i<5;i++)state[k++]=Number(!sim.itemSlots[p][i]||sim.itemSlots[p][i].item<=0||sim.itemSlots[p][i].count<=0);
 if(k!==52)throw Error('inventory input width');return {obs,state};
}
function maskedProbability(logits,mask){const mx=Math.max(...Array.from(logits,(x,i)=>mask[i]?x:-Infinity));const p=Array.from(logits,(x,i)=>mask[i]?Math.exp(x-mx):0),z=p.reduce((a,b)=>a+b,0);if(!(z>0))throw Error('empty action mask');return p.map(x=>x/z);}
function attach(model,meta){
 validate(meta);model.abilityActions=7;
 model._abilityMask=(sim,masks,p)=>S.abilityMask(sim,p,masks);
 model._decodeAbility=(move,ability)=>S.engineAction(move,ability);
 model._getLaggedInputs=inputs;
 model.act=async function(sim,p,rng){const x=inputs(sim,p),mask=sim.legalMask(),v=await this.forward(x.obs,x.state),bm=S.abilityMask(sim,p,mask);
  if(v.move.length!==5||v.bomb.length!==7)throw Error('five-slot output dimension');
  const move=this._sampleMasked(v.move,mask.mm[p],rng),ability=this._sampleMasked(v.bomb,bm,rng);
  this.lastDecision={move,ability,move_mask:Array.from(mask.mm[p]),ability_mask:bm,move_probability:maskedProbability(v.move,mask.mm[p]),ability_probability:maskedProbability(v.bomb,bm)};
  return S.engineAction(move,ability);
 };
 return model;
}
function create(Q,document){validate(document.meta);return attach(new Q.TransformerModel(document),document.meta);}
function dispatch(Q,document,legacyLoader){
 if(document.meta?.schema_version==='five-item-slots-v1')return create(Q,document);
 if((document.meta?.ability_actions??2)<=3)return legacyLoader?legacyLoader(document):new Q.TransformerModel(document);
 throw Error('unknown action schema; shape inference prohibited');
}
function jointActions(sim,p){const masks=sim.legalMask(),bm=S.abilityMask(sim,p,masks),out=[];for(let move=0;move<5;move++)for(let a=0;a<7;a++)if(masks.mm[p][move]&&bm[a])out.push({move,ability:a,engine_action:S.engineAction(move,a)});return out;}
return {validate,inputs,attach,create,dispatch,maskedProbability,jointActions};
});
