'use strict';
// Isolated evaluator entry; no production registry or old opponent edits.
const Q=require('../web/sim-bun-native-v2');
require('./r4_pressure_metrics').install(Q);
require('./r4_carrier_metrics').install(Q);
if(process.env.R4_PRESSURE_OPPONENT==='interceptor'){
 const authority=require('./eval_web_hunter'),{BunInterceptorBot,VERSION}=require('../web/bun_interceptor_bot');
 authority.createHunter=()=>{
  const bot=new BunInterceptorBot();
  return {adapter:{bot,reset:ctx=>bot.reset(ctx.seed),act:(o,p)=>{const a=bot.act(o.metadata.sim,p);return {move:a[0],ability:a[1],itemSlot:0};}},
   spec:{id:'bun.interceptor',version:VERSION},config:{difficulty:'hard_intercept'},resolved:{...bot.cfg}};
 };
}
require('./r3_items_candidate_eval');
