'use strict';
(function(root,factory){const node=typeof module==='object'&&module.exports;const api=factory(node?require('./review_protocol'):root.QQTReview,node?require('./sim-bun-native-v2'):root.QQT,node?require('./five-item-slots-v1'):root.QQTFiveSlots);if(node)module.exports=api;else root.QQTFiveSlotsReview=api;})(typeof globalThis!=='undefined'?globalThis:this,function(P,Q,S){
const SCHEMA='qqt.diagnostic-replay/five-item-slots-v1';
function compatible(d){return {...d,schema:P.SCHEMA,actions:d.actions.map(row=>row.map((v,i)=>i%2&&v>=2?2:v))};}
function validate(d){
 if(d?.schema!==SCHEMA||d.meta?.action_schema!=='move-v1+five-item-slots-v1'||d.meta?.ability_actions!==7||d.meta?.rule_version!=='bun-native-v2')throw Error('explicit five-slot Review metadata');
 for(const a of d.actions||[])if(a.length!==4||a.some((v,i)=>!Number.isInteger(v)||v<0||v>(i%2?6:4)))throw Error('explicit slot replay action');
 P.validate(compatible(d));return d;
}
function unpack(row){return [S.engineAction(row[0],row[1]),S.engineAction(row[2],row[3])];}
function setup(meta,level){
 const s=new Q.Sim(meta.seed);s.reset(level,{nativeTrap:true,nativeItems:true});s.maxSteps=meta.max_ticks;
 if(meta.frozen_birth){const cells=meta.frozen_birth.cells;for(let p=0;p<2;p++){s.pos[2*p]=cells[p][0]+.5;s.pos[2*p+1]=cells[p][1]+.5;}s.bunSpawnPos=cells.map(x=>x.map(v=>v+.5));s._gen++;}
 return s;
}
class Player extends P.Player{
 constructor(d,level){validate(d);super(compatible(d),level);this.document=d;this.cache.clear();this.reset();}
 reset(){this.sim=setup(this.document.meta,this.level);this.info=null;this.countRandom(this.sim,this.sim.rng);if(!this.cache.has(0))this.checkpoint();}
 restore(tick){const cached=this.cache.get(tick),fresh=setup(this.document.meta,this.level);for(let i=0;i<cached.draws;i++)fresh.rng();this.sim=Object.assign(Object.create(Object.getPrototypeOf(fresh)),structuredClone(cached.data));this.info=structuredClone(cached.info);this.countRandom(this.sim,fresh.rng,cached.draws);}
 step(){if(this.sim.t>=this.document.actions.length)return false;this.info=this.sim.step(unpack(this.document.actions[this.sim.t]));if(this.sim.t%200===0)this.checkpoint();return true;}
}
return {...P,SCHEMA,validate,unpack,Player,setup,replaySetup:setup};
});
