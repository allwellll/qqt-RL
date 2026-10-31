'use strict';
const t=require('node:test'),a=require('node:assert/strict');
const {bfs,rayThreat,classify}=require('../scripts/bomb_threat_low_bomb_metrics');
t('BFS distinguishes bricks and hard wall without interpreting numeric INF',()=>{
 const wall=new Uint8Array(195).fill(1),brick=new Uint8Array(195);for(let c=0;c<5;c++)wall[15+c]=0;brick[17]=1;
 const s={wall,brick,pushable:new Uint8Array(195),crate:new Uint8Array(195),fuse:new Uint8Array(195)};
 a.equal(bfs(s,15)[19],-1);a.equal(bfs(s,15,true)[19],4);wall[18]=1;a.equal(bfs(s,15,true)[19],-1);
});
t('ray opportunity needs alignment/range and cannot pass walls or bricks',()=>{
 const s={centerCell:p=>p===0?[3,3]:[3,5],blastCap:[3,3],wall:new Uint8Array(195),brick:new Uint8Array(195),fuse:new Uint8Array(195),pushable:new Uint8Array(195),crate:new Uint8Array(195)};
 a.equal(rayThreat(s,0),true);s.brick[3*15+4]=1;a.equal(rayThreat(s,0),false);s.brick.fill(0);s.blastCap[0]=1;a.equal(rayThreat(s,0),false);
});
t('predeclared descriptive priority and high-control exclusion',()=>{
 const z={group:'low',ticks:2400,active_both_ticks:2000,dead_ticks:0,trapped_ticks:0,first_death_tick:null,brick_only_ticks:1500,connected_active_ticks:0,near5_active_ticks:0,own_half_ticks:2000,opportunity_ticks:0,rank2_mid_opportunity_ticks:0};
 a.equal(classify(z),'碰不到敌人');a.equal(classify({...z,first_death_tick:10}),'被对手压制');a.equal(classify({...z,group:'high'}),'对照');
 a.equal(classify({...z,brick_only_ticks:0,connected_active_ticks:1600}),'能碰到但躲着不打');
 a.equal(classify({...z,brick_only_ticks:0,opportunity_ticks:20,rank2_mid_opportunity_ticks:12}),'去打了但argmax选不到放泡');
});
t('rejection snapshot uses post-decrement bomb capacity and preserves controls',()=>{
 const {rejectionSnapshot}=require('../scripts/bomb_threat_low_bomb_metrics');
 const s={centerCell:()=>[1,1],_actionCell:(a,b)=>b,owner:new Int8Array(195).fill(-1),fuse:new Int16Array(195),alive:[1,1],trapped:[0,0],brick:new Uint8Array(195),isBun:true,bunCarried:[-1,-1],bombsCap:[1,1]};
 s.owner[17]=0;s.fuse[17]=1;let r=rejectionSnapshot(s,[[4,1],[4,0]],0);a.equal(r.live_after_decrement,0);a.deepEqual(r.why,[]);
 s.fuse[17]=2;r=rejectionSnapshot(s,[[4,1],[4,0]],0);a.deepEqual(r.why,['capacity_after_decrement']);
 s.bunCarried[0]=1;s.trapped[0]=3;r=rejectionSnapshot(s,[[4,1],[4,0]],0);a.ok(r.why.includes('carrying')&&r.why.includes('native_sugar'));
});
