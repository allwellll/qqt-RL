'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),crypto=require('crypto');
const {columns}=require('../scripts/finalize_v36_current_review');
const P=require('./review_protocol'),level=require('../scripts/eval_web_hunter').loadLevel();
const specs=[['h8-v36-g32-carry_home-2026101785',107,'22f0359884838fff2a5d57c8fcbd12952b2547d44d0042e80f6f5499e1309114'],['h8-v36-g32-open-2026101782',436,'04a572ad739b70acff36cba744817bc185b854f1617f571b30b439e2a17ec5cb']];
for(const [id,t,hash]of specs){const b=fs.readFileSync('web/review-replays/'+id+'.json');assert.equal(crypto.createHash('sha256').update(b).digest('hex'),hash);const d=JSON.parse(b),raw=d.diagnostic_ticks[t-1].offline_recomputed_current_reward;
 if(id.includes('open')){const c=columns([{tick:t,route:raw.route}],d.meta.seed);assert.equal(c.distance_before[0],null,'NONE has no target distance; default home lookup must not leak into UI');}
 else {const p=new P.Player(d,level);p.seek(t-1);assert.throws(()=>columns([{tick:t,route:raw.route}],d.meta.seed,[p.sim]),/target|phase/,'D_DELIVERY must not accept a point in the home half outside the actual cage');}
}
console.log('Route target export GREEN');
