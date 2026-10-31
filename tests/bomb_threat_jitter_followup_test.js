'use strict';
const t=require('node:test'),a=require('node:assert/strict'),{holdController}=require('../scripts/bomb_threat_jitter_hold');
t('H0 returns exact movement and ability without mutation',()=>{
 const h=holdController(1),s={t:0,legalMask:()=>({mm:[[1,1,1,1,1]]})};
 for(let i=0;i<10;i++){s.t=i;const ac=[i%5,i%2,(i+1)%2];a.deepEqual(h.apply(s,0,ac),ac);}
 a.equal(h.stats().overrides,0);
});
t('H2 holds movement but recomputes ability; fixed blocks and masked request recorded',()=>{
 const h=holdController(2),s={t:0,legalMask:()=>({mm:[[1,1,1,0,1]]})};
 a.deepEqual(h.apply(s,0,[3,0,0]),[3,0,0]);s.t=1;a.deepEqual(h.apply(s,0,[2,1,0]),[3,1,0]);s.t=2;a.deepEqual(h.apply(s,0,[0,0,1]),[0,0,1]);a.equal(h.stats().masked_held_ticks,2);
});
t('new episode state cannot inherit previous movement',()=>{
 const h=holdController(4),mk=()=>({t:0,legalMask:()=>({mm:[[1,1,1,1,1]]})}),s=mk();h.apply(s,0,[3,0,0]);a.deepEqual(h.apply(mk(),0,[1,0,0]),[1,0,0]);
});
t('H4 is fixed four-tick blocks and never holds ability',()=>{
 const h=holdController(4),s={t:0,legalMask:()=>({mm:[[1,1,1,1,1]]})};
 const moves=[];for(let i=0;i<8;i++){s.t=i;const x=h.apply(s,0,[i%5,i%2,Number(i%3===0)]);moves.push(x[0]);a.equal(x[1],i%2);a.equal(x[2],Number(i%3===0));}a.deepEqual(moves,[0,0,0,0,4,4,4,4]);
});
t('production encoded inputs omit last requested direction but include speed growth',()=>{
 const Q=require('../web/sim-stock-window-v1'),A=require('../scripts/eval_web_hunter'),{encodeState}=require('../scripts/je05_bun_encoding');
 const s=new Q.Sim(2026122900);s.reset(A.loadLevel(),{nativeTrap:true,nativeItems:true});
 const obs=Array.from(s.encodeObsJAX(0,24)),state=Array.from(encodeState(s,0));s.lastMoveDir[0]=(s.lastMoveDir[0]+2)%4;s.slideDir[0]=(s.slideDir[0]+1)%4;
 a.deepEqual(Array.from(s.encodeObsJAX(0,24)),obs);a.deepEqual(Array.from(encodeState(s,0)),state);
 s.spdG[0]+=.1;a.notEqual(encodeState(s,0)[5],state[5]);
});
