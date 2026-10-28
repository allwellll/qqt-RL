'use strict';
const assert=require('assert');
const {StockAcquisitionFunnel}=require('../scripts/stock_acquisition_funnel');
function before(window='free'){
  return {alive:[true,window!=='dead'],trap:[0,window==='sugar'?10:0],carry:[-1,-1],stock:[[1,0],[0,1]]};
}
function after(b){return {alive:b.alive.slice(),trapped:b.trap.slice(),bunCarried:[1,-1],bunStored:[[1,0],[0,0]]};}
for(const window of ['free','dead','sugar']){
  const f=new StockAcquisitionFunnel(0),b=before(window),a=after(b);
  f.observe(b,a,7);
  const deposited={...a,bunCarried:[-1,-1],bunStored:[[1,1],[0,0]]};
  f.observe({alive:a.alive,trap:a.trapped,carry:a.bunCarried,stock:a.bunStored},deposited,20);
  assert.equal(f.metrics.stock_pickup_free_alive,Number(window==='free'));
  assert.equal(f.metrics.window_stock_pickup,Number(window!=='free'));
  assert.equal(f.metrics.window_stock_delivery,Number(window!=='free'));
  assert.equal(f.metrics.free_stock_delivery,Number(window==='free'));
}
// The enemy can become controlled during the action, before Bun interaction.
{
  const f=new StockAcquisitionFunnel(0),b=before(),a=after(before('sugar'));
  f.observe(b,a,1,{alive:[true,true],trapped:[0,10],carried:[-1,-1],stock:b.stock});
  assert.equal(f.metrics.window_stock_pickup,1);
}
// A stale historical window never grants current stock acquisition authority.
{
  const f=new StockAcquisitionFunnel(0),dead=before('dead');
  f.observe(dead,{alive:[true,true],trapped:[0,0],bunCarried:dead.carry,bunStored:dead.stock},2);
  const b=before(),a=after(b);f.observe(b,a,100);
  assert.equal(f.metrics.stock_pickup_expired_window,1);
  assert.equal(f.metrics.window_stock_pickup,0);
}
// Losing cargo clears qualification. A later delivery cannot reuse it.
{
  const f=new StockAcquisitionFunnel(0),b=before('dead'),a=after(b);f.observe(b,a,1);
  f.observe({carry:a.bunCarried,stock:a.bunStored,alive:a.alive,trap:a.trapped},{...a,bunCarried:[-1,-1]},2);
  f.observe(before(),{...after(before()),bunCarried:[-1,-1],bunStored:[[1,1],[0,1]]},3);
  assert.equal(f.metrics.window_stock_delivery,0);
  assert.equal(f.metrics.delivery_without_tracked_acquisition,1);
}
// Actual dropped cargo is distinct from stock, regardless of enemy recovery.
{
  const f=new StockAcquisitionFunnel(0),b=before(),a=after(b);a.bunStored=b.stock;
  f.observe(b,a,1);assert.equal(f.metrics.nonstock_pickup,1);assert.equal(f.metrics.stock_pickup_free_alive,0);
}
console.log('stock acquisition qualification: 7 cases passed');
