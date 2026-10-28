'use strict';
const Q=require('../web/sim.js'),topology=require('./je08_web_diagnostics.js');
const DIR=[[-1,0],[1,0],[0,-1],[0,1],[0,0]];
class ItemMetrics{
 constructor(s,p){this.s=s;this.p=p;this.enemy=1-p;this.rows={};this.slide=null;this.glue=null;this.contacts=[];
  const place=s._placeHeldItem,me=this;s._placeHeldItem=function(player,cell,slot){const item=this.itemSlots[player][slot||0]?.item;const ok=place.call(this,player,cell,slot);if(ok&&player===me.p)me.row(me.phase0||me.phase(),item).legal_releases++;return ok;};
  const old=s._contactFieldItem;s._contactFieldItem=function(player,requested,contactCell){
   const [r,c]=this.centerCell(player),cell=contactCell==null?r*15+c:contactCell;
   const item=this.fieldItem[cell],owner=this.fieldOwner[cell];
   old.call(this,player,requested,contactCell);
   if(item&&this.fieldItem[cell]===0)me.contacts.push({player,item,owner,cell,tick:this.t});
  };
 }
 phase(){const s=this.s;return s.bunCarried[this.p]===this.enemy?'carry':!s.alive[this.enemy]||s.trapped[this.enemy]>0?'window':'combat';}
 row(phase,item){const key=`${phase}_${item===1?'banana':'glue'}`;return this.rows[key]||(this.rows[key]={opportunity_ticks:0,inventory_ticks:0,release_attempts:0,legal_releases:0,self_contacts:0,enemy_hits:0,completed:0,net_home_distance:0,pursuit_delay_cells:0,beneficial:0,harm:0,deliveries:0});}
 distance(){const s=this.s,b=s.bunBases[this.p],goals=[];for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)goals.push(r*15+c);
  const block=Array.from(s.wall,(v,i)=>Number(!!(v||s.brick[i]||s.pushable[i]))),d=topology.distance(block,new Uint8Array(195),goals);
  const y=s.pos[this.p*2],x=s.pos[this.p*2+1],r=Math.floor(y),c=Math.floor(x);let best=32767;for(const [dr,dc]of DIR){const rr=r+dr,cc=c+dc,i=rr*15+cc;if(rr>=0&&rr<13&&cc>=0&&cc<15&&!block[i]&&d[i]<32767)best=Math.min(best,d[i]+Math.abs(y-rr-.5)+Math.abs(x-cc-.5));}return best;}
 before(a){const s=this.s;this.phase0=this.phase();this.inventory=s.itemSlots[this.p].map(x=>x?{item:x.item,count:x.count}:null);this.contacts=[];
  this.pos=Array.from(s.pos);this.d=this.distance();this.alive=s.alive[this.p];this.carried=s.bunCarried[this.p];this.stock=s.bunStored[this.p][this.enemy];
  for(const item of [1,2]){const row=this.row(this.phase0,item);row.opportunity_ticks+=Number(s.alive[this.p]&&s.trapped[this.p]===0);row.inventory_ticks+=Number(this.inventory.some(x=>x?.item===item&&x.count>0));}
  const item=this.inventory[0]?.item;if(item&&a[2])this.row(this.phase0,item).release_attempts++;
 }
 after(){const s=this.s,p=this.p,enemy=this.enemy;
  for(const item of [1,2]){const pre=this.inventory.reduce((n,x)=>n+(x?.item===item?x.count:0),0),post=s.itemSlots[p].reduce((n,x)=>n+(x?.item===item?x.count:0),0);/* Actual native placement hook owns releases; inventory death-drops are not releases. */}
  for(const hit of this.contacts){
   if(hit.player===p)this.row(this.phase0,hit.item).self_contacts++;
   if(hit.player===enemy&&hit.owner===p)this.row(this.phase0,hit.item).enemy_hits++;
   if(hit.item===1&&hit.player===p){this.slide={phase:this.phase0,start:this.d,ticks:0,valid:hit.owner===p&&this.carried===enemy};}
   if(hit.item===2&&hit.player===enemy&&hit.owner===p){this.glue={phase:this.phase0,start:this.d,delay:0,valid:this.carried===enemy&&s.bunCarried[enemy]<0};}
  }
  const bad=!s.alive[p]||s.trapped[p]>0||s.bunCarried[p]!==enemy;
  if(this.slide){const e=this.slide;e.ticks++;e.valid&&=!bad;
   if(s.movementStatus[p]!==2){const row=this.row(e.phase,1),gain=e.start-this.distance()-.18*e.ticks;row.completed++;row.net_home_distance+=gain;row.beneficial+=Number(e.valid&&gain>0);row.harm+=Number(!e.valid||gain<=0);this.slide=null;}}
  if(this.glue){const e=this.glue;e.valid&&=!bad;
   const dy=s.pos[enemy*2]-this.pos[enemy*2],dx=s.pos[enemy*2+1]-this.pos[enemy*2+1];
   const chase=dy*(this.pos[p*2]-this.pos[enemy*2])+dx*(this.pos[p*2+1]-this.pos[enemy*2+1])>1e-6;
   if(chase&&s.movementStatus[enemy]===1&&s.bunCarried[enemy]<0)e.delay+=Math.max(0,Q.CFG.stepLen*s.spdG[enemy]-.18);
   if(s.movementStatus[enemy]!==1||!s.alive[enemy]||s.trapped[enemy]>0){const row=this.row(e.phase,2);row.completed++;row.pursuit_delay_cells+=e.delay;row.beneficial+=Number(e.valid&&e.delay>0&&this.distance()<e.start);row.harm+=Number(!e.valid);this.glue=null;}}
  if(s.bunStored[p][enemy]>this.stock)for(const item of [1,2])this.row(this.phase0,item).deliveries++;
 }
}
module.exports={ItemMetrics};
