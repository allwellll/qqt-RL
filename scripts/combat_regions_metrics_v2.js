'use strict';
// Passive O(map-size) measurements. No inference, RNG, action or reward changes.
const observers=new WeakMap();
function region(y,x){return y>=4&&y<9&&x>=5&&x<10?'center':(y<6?'top_':'bottom_')+(x<7?'left':'right');}
function empty(){return {alive_ticks:0,actionable_ticks:0,contact_ticks:0,legal_capacity_ticks:0,
 legal_repeat_opportunities:0,placed:0,contact_placed:0,empty_placed:0,repeats:0,contact_repeat30:0,
 placement_gaps:[],eligible_gaps:[],route_cut_proxy:0,causal_controls:0,causal_kills:0,
 repeated_attack_controls:0,repeated_attack_kills:0,physical_self_control:0,self_deaths:0,deaths:0};}
function component(s,start,extra=-1){
 const blocked=i=>Boolean(s.wall[i]||s.brick[i]||s.pushable[i]||s.fuse[i]>1||i===extra);
 const seen=new Set([start]),queue=[start];
 for(let k=0;k<queue.length;k++){const i=queue[k],r=Math.floor(i/15),c=i%15;
  for(const n of [r>0?i-15:-1,r<12?i+15:-1,c>0?i-1:-1,c<14?i+1:-1])
   if(n>=0&&!seen.has(n)&&!blocked(n)){seen.add(n);queue.push(n);}}
 return seen.size;
}
function legalCapacity(s,p){
 const [r,c]=s.centerCell(p),i=r*15+c;
 let bombs=0;for(let n=0;n<195;n++)bombs+=Number(s.owner[n]===p&&s.fuse[n]>1);
 return Boolean(s.alive[p]&&s.trapped[p]===0&&s.bunCarried[p]<0&&!s.brick[i]&&s.fuse[i]<=1&&bombs<s.bombsCap[p]);
}
function attach(s,p){
 const value={p,last:-1,eligible:0,lastRepeat:false,control:null,selfControlled:false,
  groups:Object.fromEntries(['top_left','top_right','bottom_left','bottom_right','center'].map(k=>[k,empty()])),
  placements:[],contact:false};observers.set(s,value);return value;
}
function install(Q){
 const step=Q.Sim.prototype.step;
 Q.Sim.prototype.step=function(actions){
  const x=observers.get(this);if(!x||this.done)return step.call(this,actions);
  const p=x.p,q=1-p,s=this,at=region(s.pos[2*p],s.pos[2*p+1]),g=x.groups[at];
  const active=Boolean(s.alive[p]&&s.trapped[p]===0),enemy=Boolean(s.alive[q]&&s.trapped[q]===0);
  const contact=active&&enemy&&Math.abs(s.pos[2*p]-s.pos[2*q])+Math.abs(s.pos[2*p+1]-s.pos[2*q+1])<=6;
  const legal=legalCapacity(s,p),oldEnemyTrapped=s.trapped[q],oldOwnTrapped=s.trapped[p];
  g.alive_ticks+=Number(s.alive[p]);g.actionable_ticks+=Number(active);g.contact_ticks+=Number(contact);x.contact ||= contact;
  g.legal_capacity_ticks+=Number(legal);if(legal&&x.last>=0){x.eligible++;g.legal_repeat_opportunities++;}
  const [r,c]=s.centerCell(p),[er,ec]=s.centerCell(q),cell=r*15+c;
  // Difference uses identical pre-action geometry and existing bubbles; it is
  // a one-cell graph obstruction proxy, never a forced kill or safety proof.
  let cut=false;if(contact&&legal&&actions[p][1]===1)cut=component(s,er*15+ec)>component(s,er*15+ec,cell);
  const info=step.call(s,actions),now=region(s.pos[2*p],s.pos[2*p+1]);
  if(info.placed[p]){
   const repeat=x.last>=0,gap=repeat?s.t-x.last:null;
   g.placed++;g.contact_placed+=Number(contact);g.empty_placed+=Number(!contact);g.repeats+=Number(repeat);
   g.contact_repeat30+=Number(contact&&repeat&&gap<=30);g.route_cut_proxy+=Number(cut);
   if(repeat){g.placement_gaps.push(gap);g.eligible_gaps.push(x.eligible);}
   x.lastRepeat=Boolean(contact&&repeat&&gap<=30);
   x.placements.push({tick:s.t,region:at,contact,gap,eligible_gap:repeat?x.eligible:null,route_cut_proxy:cut});
   x.last=s.t;x.eligible=0;
  }
  if(oldEnemyTrapped===0&&s.trapped[q]>0){
   const src=info.causalDamageSource?.[q],unique=src?.filter(Boolean).length===1&&src[p];
   x.control=unique?{region:now,repeat:x.lastRepeat&&s.t-x.last<=60}:null;
   if(unique){x.groups[now].causal_controls++;x.groups[now].repeated_attack_controls+=Number(x.control.repeat);}
  }
  if(oldOwnTrapped===0&&s.trapped[p]>0){
   const src=info.physicalDamageSource?.[p];x.selfControlled=Boolean(src?.filter(Boolean).length===1&&src[p]);
   x.groups[now].physical_self_control+=Number(x.selfControlled);
  }
  if(info.died[q]){
   if(x.control){const h=x.groups[x.control.region];h.causal_kills++;h.repeated_attack_kills+=Number(x.control.repeat);}
   else if(oldEnemyTrapped===0&&info.causalKill?.[p])x.groups[now].causal_kills++;
   x.control=null;
  }else if(oldEnemyTrapped>0&&s.trapped[q]===0)x.control=null;
  if(info.died[p]){x.groups[now].deaths++;x.groups[now].self_deaths+=Number(x.selfControlled||info.ownBombDefeat?.[p]);
   x.last=-1;x.eligible=0;x.lastRepeat=false;x.selfControlled=false;
  }else if(oldOwnTrapped>0&&s.trapped[p]===0)x.selfControlled=false;
  return info;
 };
}
function result(s){const x=observers.get(s);return {groups:x.groups,placements:x.placements,contact:x.contact,
 definitions:{route_cut_proxy:'marginal reachable-component reduction from the actual new bubble cell in otherwise identical geometry; not a safety or forced-control claim',
 repeated_attack_conversion:'unique-other initial control after a contact repeat within 30 ticks, with most recent own placement <=60 ticks; temporal attack-chain association, not per-bubble causal attribution',
 eligible_gap:'legal-capacity ticks between actual own placements; reset on own death',
 empty_placed:'outside the both-actionable distance<=6 contact definition, not automatically tactically useless'}};}
module.exports={install,attach,result,region,legalCapacity,component};
