'use strict';
const t=require('node:test'),a=require('node:assert/strict');
const M=require('../scripts/bomb_threat_p0_jitter_metrics');
t('positive oscillating path is distinct from zero-motion blocked state',()=>{
 const z=Array.from({length:51},(_,i)=>[2,i%2?1.2:1]);
 a.ok(M.kinematics(z).oscillation);a.equal(M.kinematics(z.map(()=>[2,1])).oscillation,false);
 a.equal(M.kinematics(z.map((_,i)=>[2,i])).oscillation,false);
});
t('direct reversals require adjacent controlled requested LR actions',()=>{
 const r=M.reversals([2,3,2,4,3,2],[1,1,1,1,0,1]);
 a.deepEqual(r.events,[1,2]);a.deepEqual(r.intervals,[1]);
});
t('home distance treats live bombs and bricks as separate instantaneous blockers',()=>{
 const wall=new Uint8Array(195).fill(1);for(let i=15;i<=19;i++)wall[i]=0;
 const s={wall,brick:new Uint8Array(195),pushable:new Uint8Array(195),crate:new Uint8Array(195),fuse:new Int16Array(195)};
 s.fuse[17]=3;a.equal(M.homeDistance(s,15,[19],false,true),-1);a.equal(M.homeDistance(s,15,[19],false,false),4);
 s.brick[18]=1;a.equal(M.homeDistance(s,15,[19],false,false),-1);a.equal(M.homeDistance(s,15,[19],true,false),4);
});
t('walk-over crate pickup does not block Sim.step movement graph',()=>{
 const wall=new Uint8Array(195).fill(1);for(let i=15;i<=19;i++)wall[i]=0;
 const s={wall,brick:new Uint8Array(195),pushable:new Uint8Array(195),crate:new Uint8Array(195),fuse:new Int16Array(195)};
 s.crate[17]=1;a.equal(M.homeDistance(s,15,[19]),4);
});
t('exclusive categories preserve carry priority and disallow passive zero path oscillation',()=>{
 const base={eligible:true,carry:true,net_home_progress:0,bfs_progress:false,path:8,ratio:0,legal:4,prob:.01};
 a.equal(M.category(base),'持包停滞');a.equal(M.category({...base,carry:false}),'原地振荡');
 a.equal(M.category({...base,carry:false,ratio:.5}),'正常游走，只是概率低');a.equal(M.category({...base,eligible:false}),'其他');
 a.equal(M.category({...base,carry:false,path:0}),'其他');
});
