'use strict';
function snapshot(s){return {pos:Array.from(s.pos),alive:Array.from(s.alive),carried:Array.from(s.bunCarried),stored:s.bunStored.map(x=>x.slice())};}
function nearPosition(b,pos){const r=Math.floor(pos[0]),c=Math.floor(pos[1]);
 return Math.max(b[0]-r,0,r-b[0]-2)+Math.max(b[1]-c,0,c-b[1]-2)<=3;}
function near(s,p,team){return nearPosition(s.bunBases[team],s.pos.slice(p*2,p*2+2));}
function inCage(s,p,t){return s._bunBaseTeam(...s.centerCell(p))===t;}
function init(s){if(!s.__anti)s.__anti={version:1,alive_ticks:[0,0],actionable_ticks:[0,0],near_home_ticks:[0,0],enemy_cage_entries:[0,0],
 stock_pickups:[0,0],deliveries:[0,0],drop_or_contact_pickups:[0,0],events:[],holding_start:[null,null]};return s.__anti;}
function stockUpdate(s,fn){
 if(!s.__btIdentity)return fn();
 const before=snapshot(s),x=init(s),r=fn();
 for(let p=0;p<2;p++){
  const team=s.team[p],origin=s.bunCarried[p];
  if(before.carried[p]<0&&origin>=0){
   const stock=before.stored[origin][origin]>s.bunStored[origin][origin]&&inCage(s,p,origin);
   x[stock?'stock_pickups':'drop_or_contact_pickups'][p]++;
   x.holding_start[p]=s.t;x.events.push({tick:s.t,type:stock?'stock_pickup':'loose_pickup',player:p,origin});
  }
  if(before.carried[p]>=0&&s.bunCarried[p]<0&&s.bunStored[team][before.carried[p]]>before.stored[team][before.carried[p]]){
   x.deliveries[p]++;x.events.push({tick:s.t,type:'delivery',player:p,origin:before.carried[p],pickup_to_delivery_ticks:x.holding_start[p]===null?null:s.t-x.holding_start[p]});x.holding_start[p]=null;
  }
 }
 return r;
}
function observe(s,actions,fn){
 if(!s.__btIdentity)return fn();
 const x=init(s),b=snapshot(s),cage=[0,1].map(p=>inCage(s,p,1-s.team[p]));
 for(let p=0;p<2;p++)if(s.alive[p]){x.alive_ticks[p]++;x.near_home_ticks[p]+=Number(near(s,p,s.team[p]));x.actionable_ticks[p]+=Number(s.trapped[p]===0);}
 const r=fn();
 for(let p=0;p<2;p++){
  if(b.alive[p]&&s.alive[p]&&!cage[p]&&inCage(s,p,1-s.team[p]))x.enemy_cage_entries[p]++;
  if(b.carried[p]>=0&&s.bunCarried[p]<0&&r.died?.[p]){x.events.push({tick:s.t,type:'death_drop_or_transfer',player:p,origin:b.carried[p]});x.holding_start[p]=null;}
  if(b.carried[p]<0&&s.bunCarried[p]>=0&&x.holding_start[p]===null){x.holding_start[p]=s.t;x.drop_or_contact_pickups[p]++;x.events.push({tick:s.t,type:'contact_transfer',player:p,origin:s.bunCarried[p]});}
 }
 if(s.__bt)s.__bt.antifarm=x;
 return r;
}
module.exports={observe,stockUpdate,near,inCage,nearPosition};
