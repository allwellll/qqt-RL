'use strict';
// Passive opportunity measurements. The model, rules and both RNGs are intact.
const crypto=require('node:crypto');
const cell=(s,p)=>[Math.floor(s.pos[2*p]),Math.floor(s.pos[2*p+1])];
function rayOpportunity(s,p){
 const q=1-p,[r,c]=cell(s,p),[er,ec]=cell(s,q),range=s.blastCap[p];
 if(!s.alive[p]||!s.alive[q]||s.trapped[p]>0||s.trapped[q]>0)return false;
 if(r!==er&&c!==ec)return false;
 const distance=Math.abs(r-er)+Math.abs(c-ec);
 if(distance>range)return false;
 const dr=Math.sign(er-r),dc=Math.sign(ec-c);
 for(let d=1;d<distance;d++){
  const at=(r+dr*d)*15+c+dc*d;
  if(s.wall[at]||s.brick[at]||s.pushable[at]||s.fuse[at]>0)return false;
 }
 const masks=s.legalMask();
 return Boolean(masks.bm[p][1]);
}
function install(Q){
 const proto=Q.Sim.prototype,step=proto.step;
 if(step.__r4Passive)throw Error('duplicate R4 passive installation');
 function wrapped(actions){
  if(!this.__btIdentity)return step.call(this,actions);
  if(!this.__r4)this.__r4={version:1,opportunity:[0,0],requests:[0,0],placements:[0,0],
   probability_sum:[0,0],probability_count:[0,0],phase:[{},{}],events:[],
   action_digest:crypto.createHash('sha256'),ticks:0};
  const x=this.__r4,op=[rayOpportunity(this,0),rayOpportunity(this,1)];
  const before={alive:this.alive.slice(),carried:this.bunCarried.slice(),trapped:this.trapped.slice()};
  const phases=[0,1].map(p=>this.bunCarried[p]>=0?'carrying':!this.alive[1-p]||this.trapped[1-p]>0?'window':'combat');
  const probabilities=[0,1].map(p=>this._expertActStats?.[p]?.abilityProbabilities?.[1]);
  const result=step.call(this,actions);x.ticks++;x.action_digest.update(JSON.stringify(actions));
  for(let p=0;p<2;p++){
   const phase=x.phase[p][phases[p]] ||= {ticks:0,opportunities:0,bomb_requests:0,bomb_placed:0};phase.ticks++;
   if(op[p]){
    x.opportunity[p]++;phase.opportunities++;
    if(actions[p][1]){x.requests[p]++;phase.bomb_requests++;}
    if(result.placed[p]){x.placements[p]++;phase.bomb_placed++;}
    if(Number.isFinite(probabilities[p])){x.probability_sum[p]+=probabilities[p];x.probability_count[p]++;}
   }
   if(result.died?.[p])x.events.push({tick:this.t,type:'death',player:p,self:Boolean(result.ownBombDefeat?.[p]),
    victim_carried:before.carried[p],trapped_before:before.trapped[p]});
  }
  if(this.done&&this.__bt){
   const {action_digest,...exported}=x;
   this.__bt.r4={...exported,joint_actions_sha256:action_digest.copy().digest('hex'),
    opportunity_definition:'pre-action both alive/uncontrolled, bomb mask legal with capacity, enemy cell in unobstructed new-bomb cross at current blast range; not forced-kill/Q',
    coverage:{training_reward:'unavailable_in_JS',pending_clawback:'unrecorded',reward_fraction:'unavailable_without_native_reward_ledger'},
    limitations:['production Sim.step only','opportunity is a geometric legal-action exposure, not guaranteed kill','ability probability recorded for learner only','existing resolved bombs/chain/item metrics remain observational']};
  }
  return result;
 }
 wrapped.__r4Passive=true;proto.step=wrapped;
}
module.exports={install,rayOpportunity};
