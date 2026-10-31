'use strict';
const assert=require('node:assert/strict'),P=require('./review_protocol'),Q=require('./sim');
const level=require('../scripts/eval_web_hunter').loadLevel();
async function main(){
 const s=new Q.Sim(2026101800);s.reset(level,{nativeTrap:true,nativeItems:true});s.brick.fill(0);s.initialHp=1;s.hp=[1,1];s.trapped[1]=60;s.pos.set([5.5,5.5,5.5,9.5]);s.maxSteps=10;
 const initial=P.state(s),meta={seed:2026101800,scene:'native',max_ticks:10,native_fixture:{schema:'qqt.native-reset-fixture/v1',task:'native_sugar_stock',initial_hp:1,frame:initial,frame_sha256:await P.hash(initial)}};
 const restored=P.replaySetup(meta,level);
 assert.deepEqual(P.state(restored),initial,'Review must restore exact declared reset fixture');
 for(let t=0;t<10;t++){s.step([[4,0,0],[4,0,0]]);restored.step([[4,0,0],[4,0,0]]);assert.deepEqual(P.state(restored),P.state(s));}
 console.log('JE10 fixture Review exact state and ten-step replay passed');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
