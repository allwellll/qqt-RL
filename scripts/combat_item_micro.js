'use strict';
// Synthetic mechanism fixtures only; never a training state bank.
const assert=require('assert'),fs=require('fs'),Q=require('../web/sim.js');
const authority=require('./eval_web_hunter.js');
function fresh(){const s=new Q.Sim(20261023);s.reset(authority.loadLevel(),{nativeItems:true,nativeTrap:true});
 for(const k of ['wall','brick','fuse','crate','pushable','blastLinger'])s[k].fill(0);
 s.pos.set([5.5,5.5,9.5,10.5]);s.invuln.fill(0);s.spawnProtection.fill(0);s.bunCarried=[-1,-1];return s;}
const s=fresh(),speed=()=>s.playerMoveScale(0)*40*Q.CFG.speed*s.spdG[0];
const free=speed();s.bunCarried[0]=1;const carried=speed();s._setMovementStatus(0,Q.MOVE_STATUS_SLOW);const slowCarried=speed();
s._setMovementStatus(0,Q.MOVE_STATUS_SLIDE,0);s.slideDir[0]=3;const slide=speed();
assert(Math.abs(carried-72)<1e-5&&Math.abs(slowCarried-72)<1e-5&&Math.abs(slide-268)<1e-5);
const x=s.pos[1];s.step([[2,0],[4,0]]);assert(s.pos[1]>x&&s.movementStatus[0]===Q.MOVE_STATUS_SLIDE);
s.wall[5*Q.W+7]=1;let ticks=0;while(s.movementStatus[0]===Q.MOVE_STATUS_SLIDE&&ticks<40){s.step([[2,0],[4,0]]);ticks++;}
assert(ticks<40&&s.pos[1]<7);
const g=fresh();g.fieldItem[5*Q.W+5]=Q.ITEM_SLOW_GLUE;g.fieldOwner[5*Q.W+5]=1;g.fieldArmed[5*Q.W+5]=1;g._contactFieldItem(0,3);
assert(g.movementStatus[0]===Q.MOVE_STATUS_SLOW&&g.trapped[0]===0&&g.fieldItem[5*Q.W+5]===0);
const out={passed:true,scope:'fixed mechanism fixture only; no artificial carrier training',engine:'production JS',speed_px_per_second:{free,carried,slowCarried,slide},forced_direction:true,collision_stops_slide:true,collision_ticks:ticks,glue_not_native_window:true,glue_consumed_once:true};
fs.writeFileSync('runs/combat_item_20261023/v1/JS_MECHANISM.json',JSON.stringify(out,null,2));console.log(out);
