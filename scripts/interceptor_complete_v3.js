'use strict';
// Opt-in engineering version. Production Hunter, timing-v2 and rules untouched.
const H=require('../web/bun_hunter_bot');
const T=require('../tests/interceptor_timing_v2');
const VERSION='bun-interceptor-complete-v3';

class BunInterceptorBot extends T.BunInterceptorBot {
  constructor(options={}) {
    super(options);
    this.allowFallback=options.allowFallback===true;
  }
  considerBomb(state,g,pid,pred,perceive,field) {
    const timed=super.considerBomb(state,g,pid,pred,perceive,field);
    if(timed || !this.allowFallback || !this.interception)return timed;
    // Explicit v3 intervention, disabled for the complete-v2 parity contract.
    // A carrier arriving before31ticks is not a reason to veto the ordinary
    // Hunter's safely escapable immediate kill/space-reduction attack.
    const attack=H.BunHunterBot.prototype.considerBomb.call(this,state,g,pid,pred,perceive,field);
    return attack&&attack.reason!=='bomb_dig'?attack:null;
  }
}

function stateFromSim(sim) {
  const state=H.hunterStateFromSim(sim);
  state.players.forEach((p,i)=>Object.assign(p,{
    movementStatus:Number(sim.movementStatus?.[i]||0),
    slideDir:Number(sim.slideDir?.[i]??1),
    postSlideSpeed:sim.nativeItems?.valueOf()?.6:Number(sim.spdG[i])*(p.carrying>=0?Number(sim.bunCarrySpeedScale||.5):1)
  }));
  state.itemSlots=(sim.itemSlots||[[],[]]).map(a=>a.map(x=>({...x})));
  return state;
}

function publicInput(state) {
  if(state.height!==13||state.width!==15||state.players.length!==2||state.players.some((p,i)=>(p.team??i)!==i))
    throw Error('complete-v3 accepted domain is13x15 two-player opposing teams');
  const n=195,ints=a=>Array.from(a,Number),zeros=()=>new Array(n).fill(0);
  const fuse=zeros(),owner=new Array(n).fill(-1),bomb_blast=new Array(n).fill(2);
  for(const b of state.bombs){fuse[b.cell]=b.fuse;owner[b.cell]=b.owner;bomb_blast[b.cell]=b.blast;}
  const slots=state.itemSlots||state.players.map((p,i)=>state.heldItem?.[i]>0?[{item:state.heldItem[i],count:1}]:[]);
  return {
    wall:ints(state.wall).map(Boolean),brick:ints(state.brick).map(Boolean),
    brick_linger:ints(state.brickLinger),blast_linger:ints(state.blastLinger),fuse,owner,bomb_blast,
    pos:state.players.map(p=>[p.y,p.x]),speed:state.players.map(p=>p.speed),
    blast:state.players.map(p=>p.blast),bombs_cap:state.players.map(p=>p.bombsCap),
    alive:state.players.map(p=>p.alive),trapped:state.players.map(p=>p.trapped||0),
    invuln:state.players.map(p=>p.invuln||0),carry:state.players.map(p=>p.carrying),
    status:state.players.map(p=>p.movementStatus||0),slide_dir:state.players.map(p=>p.slideDir??1),
    post_slide_speed:state.players.map(p=>p.postSlideSpeed??.6),bases:state.bunBases,
    stock:state.bunStored,loose:Array.from({length:n},(_,i)=>[state.bunLoose[2*i],state.bunLoose[2*i+1]]),
    crate:ints(state.crate).map(Boolean),crate_type:ints(state.crateType),field_item:ints(state.fieldItem),
    held_type:slots.map(a=>Array.from({length:7},(_,i)=>a[i]?.item||0)),
    held_count:slots.map(a=>Array.from({length:7},(_,i)=>a[i]?.count||0))
  };
}

// Explicit future7-action collector/opponent adapter. The reference Hunter
// currently requests only0/1; no held-item heuristic is fabricated here.
function engineAction(decision) {
  const [move,ability]=decision.action;
  if(!Number.isInteger(move)||move<0||move>4||!Number.isInteger(ability)||ability<0||ability>6)
    throw Error('invalid explicit five-slot bot action');
  if(ability<=1)return [move,ability,0];
  return [move,0,1,0,null,null,ability-2];
}

module.exports={BunInterceptorBot,VERSION,stateFromSim,publicInput,engineAction};
