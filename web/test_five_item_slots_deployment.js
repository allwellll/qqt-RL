'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const Q=require('./sim-bun-native-v2'),S=require('./five-item-slots-v1'),A=require('./five-item-slots-model-v1'),R=require('./r3_model_inputs'),Bots=require('./bot_contract');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
R.installSim(Q);
function fresh(){const s=new Q.Sim(21);s.reset(level,{nativeItems:true,nativeTrap:true});s.wall.fill(0);s.brick.fill(0);s.pushable.fill(0);s.fuse.fill(0);s.crate.fill(0);s.pos.set([6.5,5.5,10.5,11.5]);s.itemSlots[0]=[{item:1,count:2},{item:2,count:3},{item:1,count:4},{item:2,count:5},{item:1,count:6}];s._syncHeldItem(0);return s;}
const meta={schema_version:'five-item-slots-v1',action_schema:'move-v1+five-item-slots-v1',ability_actions:7,state_dim:52,global_dim:52,obs_shape:[25,13,15],ability_encoding:S.ACTION_NAMES};
(async()=>{
 for(let slot=0;slot<5;slot++){
  const s=fresh(),mock={meta,greedy:true,forward:async()=>({move:[-9,-9,-9,-9,9],bomb:Array.from({length:7},(_,i)=>i===slot+2?9:-9)}),_sampleMasked:Q.TransformerModel.prototype._sampleMasked};
  A.attach(mock,meta);const raw=await mock.act(s,0,()=>.5);assert.equal(raw[6],slot);
  const action=Bots.validateAction({move:raw[0],ability:raw[2]?2:raw[1],itemSlot:raw[6]});
  const engine=[action.move,Number(action.ability===1),Number(action.ability===2),0,-1,-1,action.itemSlot];
  s.step([engine,S.engineAction(4,0)]);assert.equal(s.fieldItem[95],slot%2?2:1);assert.equal(s.itemSlots[0][slot].count,slot+1);
 }
 const s=fresh();for(let t=0;t<90;t++){
  s.step([[4,Number(t%10===0),0],[4,0,0]]);
  for(let p=0;p<2;p++){const x=A.inputs(s,p),base=R.inputs(s,p);assert.deepEqual(x.obs,base.obs);assert.deepEqual(x.state.slice(0,37),base.state);}
 }
 for(const field of ['wall','brick','fuse','fieldItem']){const s=fresh();s[field][95]=1;assert(S.abilityMask(s,0,s.legalMask()).slice(2).every(x=>x===0));}
 for(const field of ['trapped','alive']){const s=fresh();s[field][0]=field==='trapped'?2:false;assert.deepEqual(S.abilityMask(s,0,s.legalMask()),[1,0,0,0,0,0,0]);}
 assert.throws(()=>A.validate({...meta,ability_encoding:['none','bomb','use_item']}));assert.throws(()=>A.dispatch(Q,{meta:{ability_actions:7}}));
 const legacy={meta:{},obsShape:[24,13,15],_getLaggedInputs:()=>42};assert.equal(R.attach(legacy)._getLaggedInputs(),42);
 console.log('Five-slot deployment: all five slots reach native physics; inventory52, R3 batch channel, blocked/dead masks and legacy compatibility PASS');
})().catch(e=>{console.error(e);process.exitCode=1;});
