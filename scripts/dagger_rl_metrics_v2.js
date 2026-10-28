'use strict';
// Metric v2: resolve native fallback placement cell; action/RNG/physics unchanged.
const {ItemMetrics:Base}=require('./combat_item_metrics');
class ItemMetrics extends Base{
 constructor(s,p){super(s,p);this.owned=new Map();this.serial=0;this.ownedSlide=null;
  const original=s._placeHeldItem,me=this;
  s._placeHeldItem=function(player,requested,slot){const index=Number.isInteger(slot)&&slot>=0?slot:0;const item=this.itemSlots[player][index]?.item;const [r,c]=this.centerCell(player),cell=this._actionCell(requested,r*15+c);const ok=original.call(this,player,requested,slot);if(ok&&player===p)me.owned.set(cell,{item,id:++me.serial,phase:me.phase0||me.phase(),cell});return ok;};
 }
 row(phase,item){const r=super.row(phase,item);for(const k of ['first_slot_legal_ticks','release_probability_sum','greedy_release_ticks','owned_self_contacts','other_self_contacts','owned_slide_completed','owned_slide_beneficial','owned_slide_harm','owned_net_home_distance'])if(r[k]===undefined)r[k]=0;return r;}
 before(a){super.before(a);const s=this.s,decision=s._daggerItemDecision?.[this.p];const item=this.inventory[0]?.item;
  if(item&&decision?.legal[2]){const r=this.row(this.phase0,item);r.first_slot_legal_ticks++;r.release_probability_sum+=decision.probabilities[2];r.greedy_release_ticks+=Number(decision.greedy===2);}
 }
 after(){const s=this.s,p=this.p;
  for(const hit of this.contacts){if(hit.player!==p)continue;const record=this.owned.get(hit.cell);const own=hit.owner===p&&record?.item===hit.item;this.row(this.phase0,hit.item)[own?'owned_self_contacts':'other_self_contacts']++;
   if(own&&hit.item===1)this.ownedSlide={id:record.id,phase:record.phase,start:this.d,ticks:0,valid:this.carried===this.enemy};
   this.owned.delete(hit.cell);
  }
  if(this.ownedSlide){const e=this.ownedSlide;e.ticks++;e.valid&&=s.alive[p]&&s.trapped[p]===0&&s.bunCarried[p]===this.enemy;
   if(s.movementStatus[p]!==2){const gain=e.start-this.distance()-.18*e.ticks,r=this.row(e.phase,1);r.owned_slide_completed++;r.owned_net_home_distance+=gain;r.owned_slide_beneficial+=Number(e.valid&&gain>0);r.owned_slide_harm+=Number(!e.valid||gain<=0);this.ownedSlide=null;}
  }
  super.after();
 }
}
module.exports={ItemMetrics};
