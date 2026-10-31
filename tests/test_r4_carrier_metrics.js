'use strict';
const assert=require('node:assert/strict'),M=require('../scripts/r4_carrier_metrics');
function fixture(){return {wall:Array(195).fill(0),brick:Array(195).fill(0),pushable:Array(195).fill(0),
 pos:[6.5,6.5,6.5,8.5],team:[0,1],bunBases:[[4,0],[4,12]],bunCarried:[-1,0],
 bunStored:[[0,0],[0,1]],alive:[true,true],trapped:[0,0],t:0};}
const s=fixture();let b=M.beforeStep(s);assert.equal(M.init(s).interception_ticks[0],1);
assert.equal(M.init(s).own_actionable_ticks[0],1);M.beforeStep(s);assert.equal(M.init(s).exposed_carrier_lives[0],1);
s.trapped[0]=2;M.beforeStep(s);assert.equal(M.init(s).own_actionable_ticks[0],2);
s.trapped[0]=0;M.init(s).cycle[0]={pickup_tick:0,killed:false};
M.afterStep(s,b,{died:[false,true],causalKill:[true,false]});assert.equal(M.init(s).carrier_causal_kills[0],1);
s.bunCarried=[0,-1];M.stockUpdate(s,()=>{s.bunStored[0][0]++;s.bunCarried[0]=-1;});
assert.equal(M.init(s).completed_antifarm_cycles[0],1);
// Losing the package to an opponent delivery clears a pending loop.
s.bunCarried=[-1,0];M.init(s).cycle[0]={pickup_tick:1,killed:false};
M.stockUpdate(s,()=>{s.bunStored[1][0]++;s.bunCarried[1]=-1;});assert.equal(M.init(s).enemy_deliveries[0],1);
assert.equal(M.init(s).cycle[0],null);
// A disconnected route is missing exposure, never a fabricated interception.
const z=fixture();z.wall.fill(1);M.beforeStep(z);assert.equal(M.init(z).reachable_path_ticks[0],0);
assert.equal(M.init(z).own_actionable_ticks[0],1);
assert.equal(M.exportMetrics(s).cache,undefined);
console.log('PASS carrier denominators, life dedup, causal cycle, delivery, unreachable path');
