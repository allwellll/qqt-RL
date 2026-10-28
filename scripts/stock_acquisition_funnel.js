'use strict';
// Passive attribution only. Does not change legal actions, inference or RNG.
class StockAcquisitionFunnel {
  constructor(team){
    this.team=team;this.enemy=1-team;this.cargo=null;this.lastWindowEnd=null;
    this.metrics={window_stock_pickup:0,stock_pickup_dead:0,stock_pickup_native_sugar:0,
      stock_pickup_free_alive:0,stock_pickup_expired_window:0,stock_pickup_free_no_history:0,
      window_stock_delivery:0,free_stock_delivery:0,nonstock_pickup:0,nonstock_delivery:0,
      delivery_without_tracked_acquisition:0};
  }
  observe(before,after,tick,interaction=null){
    const t=this.team,e=this.enemy,m=this.metrics;
    const wasWindow=!before.alive[e]||before.trap[e]>0;
    const nowWindow=!after.alive[e]||after.trapped[e]>0;
    if(wasWindow&&!nowWindow)this.lastWindowEnd=tick;
    // Use the native Bun interaction stage when available. This differs from
    // the decision state if a bomb creates control during the same action.
    const at=interaction||{alive:before.alive,trapped:before.trap,carried:before.carry,stock:before.stock};
    const picked=before.carry[t]<0&&after.bunCarried[t]===e;
    let acquisition=null;
    if(picked){
      const stock=at.carried[t]<0&&at.stock[e][e]>after.bunStored[e][e];
      const window=!at.alive[e]?'DEAD':at.trapped[e]>0?'NATIVE_SUGAR':'FREE_ALIVE';
      acquisition={source:stock?'STOCK':'NONSTOCK',window,tick,
        qualified_stock:stock&&window!=='FREE_ALIVE',last_window_end:this.lastWindowEnd};
      this.cargo=acquisition;
      if(!stock)m.nonstock_pickup++;
      else if(window==='DEAD'){m.stock_pickup_dead++;m.window_stock_pickup++;}
      else if(window==='NATIVE_SUGAR'){m.stock_pickup_native_sugar++;m.window_stock_pickup++;}
      else{m.stock_pickup_free_alive++;if(this.lastWindowEnd===null)m.stock_pickup_free_no_history++;else m.stock_pickup_expired_window++;}
    }
    const delivery=after.bunStored[t][e]>before.stock[t][e];
    if(delivery){
      if(this.cargo?.qualified_stock)m.window_stock_delivery++;
      else if(this.cargo?.source==='STOCK')m.free_stock_delivery++;
      else if(this.cargo)m.nonstock_delivery++;
      else m.delivery_without_tracked_acquisition++;
    }
    if(after.bunCarried[t]!==e||!after.alive[t])this.cargo=null;
    return acquisition;
  }
}
module.exports={StockAcquisitionFunnel};
