'use strict';
const assert=require('node:assert/strict'),H=require('../web/bun_hunter_bot');
const C=process.env.INTERCEPTOR_TEST_BASELINE?require('../web/bun_interceptor_bot').BunInterceptorBot:require('./interceptor_timing_v2').BunInterceptorBot;
const previous=H.BunHunterBot.prototype.attackSeeds;
try{
 H.BunHunterBot.prototype.attackSeeds=()=>[{cell:99}];
 const bot=new C(),state={players:[{team:0,row:3,col:3,cell:48,carrying:-1},{team:1,row:3,col:4,cell:49,carrying:0}]};
 bot.routePlan=()=>({best:{plant:63}});
 assert.deepEqual(bot.chooseGoal(state,{},0,null,()=>true).seeds,[{cell:63,cost:0}]);
 console.log(JSON.stringify({status:'PASS',target_priority:true}));
}finally{H.BunHunterBot.prototype.attackSeeds=previous;}
