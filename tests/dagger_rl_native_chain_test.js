'use strict';
const assert=require('node:assert/strict');
const {pickPlan,publicRequest}=require('../scripts/dagger_rl_native_chain');
const Q=require('../web/sim'),authority=require('../scripts/eval_web_hunter'),P=require('../web/review_protocol');
const response={mask:Array.from({length:5},()=>[false,false,false]),diagnostic:[1,0],field:[3,4],
  plans:[{direction:0,first:1,valid:true,gain:9},{direction:1,first:0,valid:true,gain:3}]};
assert.equal(pickPlan(response),null,'unaccepted forecast cannot become a committed teacher plan');
response.mask[0][2]=true;
assert.equal(pickPlan(response).direction,1,'highest raw gain is insufficient without a legal accepted joint action');
response.mask[1][2]=true;
assert.equal(pickPlan(response).direction,0);
const s=new Q.Sim(13);s.reset(authority.loadLevel(),{nativeItems:true,nativeTrap:true});
const before=P.state(s),model={_abilityMask:(sim,mask,p)=>mask.am[p]};
const request=publicRequest(s,{p:1},model);
assert.deepEqual(request.state.pos,[[s.pos[2],s.pos[3]],[s.pos[0],s.pos[1]]]);
assert.deepEqual(request.state.bun_carried,[-1,-1]);assert.equal(request.real_custody,false);
assert.deepEqual(request.move_mask,[s.legalMask().mm[1],s.legalMask().mm[0]]);
assert.deepEqual(P.state(s),before,'public forecast adapter cannot edit native physics');
console.log('native mechanism masks, seat orientation, no invented custody and passive request tests passed');
