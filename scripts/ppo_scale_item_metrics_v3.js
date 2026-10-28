'use strict';
// Passive whole-placement chains. v2 contact-to-slide proxies stay separately named.
const {ItemMetrics:V2}=require('./dagger_rl_metrics_v2');
const Q=require('../web/sim');
class ItemMetrics extends V2 {
  constructor(s,p){
    super(s,p);this.chainSerial=0;this.chains=new Map();this.cellChain=new Map();this.chainEvents=[];
    const place=s._placeHeldItem,me=this;
    s._placeHeldItem=function(player,requested,slot){
      const index=Number.isInteger(slot)&&slot>=0?slot:0,item=this.itemSlots[player][index]?.item;
      const [r,c]=this.centerCell(player),cell=this._actionCell(requested,r*15+c);
      const ok=place.call(this,player,requested,slot);
      if(ok&&player===p){
        const previous=me.cellChain.get(cell);if(previous)me.finish(previous,'replaced',true);
        const id=++me.chainSerial;
        me.chains.set(id,{id,item,cell,phase:me.phase0||me.phase(),tick:me.beforeTick??s.t,
          origin:me.d??me.distance(),carry:me.carried===me.enemy,valid:true,contact:false,
          delay:0,sliding:false});me.cellChain.set(cell,id);
        me.row(me.phase0||me.phase(),item).chain_placements++;
      }
      return ok;
    };
  }
  row(phase,item){const r=super.row(phase,item);for(const key of [
    'chain_placements','chain_owned_contacts','chain_complete','chain_censored','chain_beneficial',
    'chain_harm','chain_total_ticks','chain_preparation_ticks','chain_net_home_distance',
    'chain_actual_chase_delay_cells'])if(r[key]===undefined)r[key]=0;return r;}
  before(action){super.before(action);this.beforeTick=this.s.t;
    this.beforeTopology=Array.from(this.s.wall,(v,i)=>Number(!!(v||this.s.brick[i]||this.s.pushable[i])));
  }
  finish(id,reason,censored=false){const e=this.chains.get(id);if(!e)return;
    const ticks=Math.max(1,this.s.t-e.tick),net=e.origin-this.distance()-.18*ticks;
    const r=this.row(e.phase,e.item);r.chain_total_ticks+=ticks;
    r.chain_preparation_ticks+=e.contactTick===undefined?ticks:Math.max(0,e.contactTick-e.tick);
    r.chain_censored+=Number(censored);r.chain_complete+=Number(!censored);
    if(!censored){r.chain_net_home_distance+=net;r.chain_actual_chase_delay_cells+=e.delay;
      const beneficial=e.valid&&e.carry&&e.contact&&(e.item===1?net>0:e.delay>0&&this.distance()<e.origin);
      r.chain_beneficial+=Number(beneficial);r.chain_harm+=Number(!e.valid);}
    this.chainEvents.push({id:e.id,item:e.item,phase:e.phase,reason,censored,ticks,
      preparation_ticks:e.contactTick===undefined?null:e.contactTick-e.tick,
      owned_contact:e.contact,valid:e.valid,started_with_real_cargo:e.carry,
      net_home_distance:censored?null:net,actual_chase_delay_cells:e.delay});
    this.chains.delete(id);if(this.cellChain.get(e.cell)===id)this.cellChain.delete(e.cell);
  }
  after(){const s=this.s,p=this.p,enemy=this.enemy;
    const topologyChanged=this.beforeTopology.some((v,i)=>v!==Number(!!(s.wall[i]||s.brick[i]||s.pushable[i])));
    for(const hit of this.contacts){const id=this.cellChain.get(hit.cell),e=this.chains.get(id);
      if(!e||hit.owner!==p||hit.item!==e.item)continue;
      if((e.item===1&&hit.player===p)||(e.item===2&&hit.player===enemy)){
        for(const other of [...this.chains.values()])if(other.id!==id&&other.contact&&other.item===e.item)
          this.finish(other.id,'overlapping_contact',true);
        e.contact=true;e.contactTick=s.t;e.sliding=e.item===1;
        this.row(e.phase,e.item).chain_owned_contacts++;
      }else this.finish(id,'consumed_by_other',true);
    }
    for(const e of [...this.chains.values()]){
      e.valid&&=Boolean(s.alive[p]&&s.trapped[p]===0&&(!e.carry||s.bunCarried[p]===enemy||s.bunStored[p][enemy]>this.stock));
      if(topologyChanged){this.finish(e.id,'topology_changed',true);continue;}
      if(e.contact&&e.item===2){const dy=s.pos[enemy*2]-this.pos[enemy*2],dx=s.pos[enemy*2+1]-this.pos[enemy*2+1];
        const chase=dy*(this.pos[p*2]-this.pos[enemy*2])+dx*(this.pos[p*2+1]-this.pos[enemy*2+1])>1e-6;
        if(chase&&s.movementStatus[enemy]===1&&s.bunCarried[enemy]<0)e.delay+=Math.max(0,Q.CFG.stepLen*s.spdG[enemy]-.18);
      }
      if(e.contact&&((e.item===1&&s.movementStatus[p]!==2)||
          (e.item===2&&(s.movementStatus[enemy]!==1||!s.alive[enemy]||s.trapped[enemy]>0))))
        this.finish(e.id,'completed');
      else if(!e.valid)this.finish(e.id,'carrier_harm');
      else if(s.done||s.t>=s.maxSteps)this.finish(e.id,'terminal_or_horizon',true);
      else if(!e.contact&&(!s.fieldItem[e.cell]||s.fieldOwner[e.cell]!==p))this.finish(e.id,'expired',true);
    }
    super.after();
  }
  flush(reason='explicit_end'){for(const id of [...this.chains.keys()])this.finish(id,reason,true);}
}
module.exports={ItemMetrics};
