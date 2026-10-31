'use strict';
const assert=require('node:assert/strict'),H=require('../web/bun_hunter_bot'),I=require('../web/bun_interceptor_bot');
const C=process.env.INTERCEPTOR_TEST_BASELINE?I.BunInterceptorBot:require('./interceptor_timing_v2').BunInterceptorBot;
// A useful forward target must not be replaced just because the carrier is near.
{
 const bot=new C(),state={players:[{team:0,row:3,col:3,cell:48,carrying:-1},{team:1,row:3,col:4,cell:49,carrying:0}]};
 bot.routePlan=()=>({best:{plant:63}});bot.attackSeeds=()=>[{cell:99}];
 const g=bot.chooseGoal(state,{},0,null,()=>true);assert.deepEqual(g.seeds,[{cell:63,cost:0}]);
}
// A late static-detour candidate cannot displace a future fuse-feasible one.
{
 const prior=I.BunInterceptorBot.prototype.routePlan;
 I.BunInterceptorBot.prototype.routePlan=()=>({candidates:[{plant:11,timed:false},{plant:12,timed:true}],best:{plant:11,timed:false}});
 try{const plan=new C().routePlan({}, {}, 0);assert.deepEqual(plan.candidates.map(x=>x.plant),[12]);assert.equal(plan.best.plant,12);}
 finally{I.BunInterceptorBot.prototype.routePlan=prior;}
}
// While intercepting, no timing opportunity means reserve inventory; don't
// fall through to a late ordinary attack bomb. No-plan combat is unchanged.
{
 const parent=H.BunHunterBot.prototype.considerBomb;H.BunHunterBot.prototype.considerBomb=()=>({move:4,reason:'parent_called'});
 try{const bot=new C();bot.interception={};
  assert.equal(bot.considerBomb({players:[{bombsLeft:0,liveBombs:0,cell:0}]},{},0,null,null,null),null);
  bot.interception=null;assert.equal(bot.considerBomb({players:[{}]}, {},0).reason,'parent_called');
 }finally{H.BunHunterBot.prototype.considerBomb=parent;}
}
// A correctly timed proposal must still pass both escape and robust masks.
{
 const footprint=I.footprint;I.footprint=()=>new Uint8Array([1]);
 try{const bot=new C(),me={cell:0,bombsLeft:1,liveBombs:0,blast:2},state={players:[me]},g={bombAt:new Int16Array([-1]),N:1};
  bot.interception={near:[0],foe:{slideTicks:0},from:[0]};bot.firstTicks=()=>31;bot.moveTicks=()=>1;
  bot.predict=()=>({});bot.escape=()=>({surv:0});bot.hypoSurvivors=()=>31;bot.alliesOf=()=>[];
  bot.pickMove=()=>4;bot.dangerCells=()=>new Uint8Array(1);
  assert.equal(bot.considerBomb(state,g,0,null,()=>true,null),null);
  bot.escape=()=>({surv:31});bot.hypoSurvivors=()=>0;
  assert.equal(bot.considerBomb(state,g,0,null,()=>true,null),null);
  bot.hypoSurvivors=()=>31;
  assert.equal(bot.considerBomb(state,g,0,null,()=>true,null).reason,'bomb_intercept_timed');
 }finally{I.footprint=footprint;}
}
// Risk thresholds and all old Hunter difficulty parameters remain identical.
assert.deepEqual(new C().cfg,new I.BunInterceptorBot().cfg);
console.log(JSON.stringify({status:'PASS',checks:['target_priority','fuse_feasible_candidates','no_untimed_fallback','escape_and_robust_veto','same_safety_config']}));
