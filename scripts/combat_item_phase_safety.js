'use strict';
// Passive decision-stage safety ledger; never writes game state or consumes RNG.
class PhaseSafety {
 constructor(team){this.team=team;this.rows={};this.everPicked=false;this.pre_first_acquisition_deaths=0;}
 before(s){const p=this.team,e=1-p;this.phase=s.bunCarried[p]===e?'carry':!s.alive[e]||s.trapped[e]>0?'window':'combat';
  this.trapped=s.trapped[p];this.pickedBefore=this.everPicked;
  const r=this.rows[this.phase]||(this.rows[this.phase]={ticks:0,alive_free_ticks:0,deaths:0,self_defeats:0,trap_entries:0,self_trap_entries:0,enemy_trap_entries:0});
  r.ticks++;r.alive_free_ticks+=Number(s.alive[p]&&s.trapped[p]===0);
 }
 after(s,info,picked){const p=this.team,e=1-p,r=this.rows[this.phase];
  r.deaths+=Number(info.died[p]);r.self_defeats+=Number(info.ownBombDefeat?.[p]||false);
  const entry=this.trapped===0&&s.trapped[p]>0,source=info.causalDamageSource?.[p]||[];
  r.trap_entries+=Number(entry);r.self_trap_entries+=Number(entry&&source[p]&&!source[e]);r.enemy_trap_entries+=Number(entry&&source[e]&&!source[p]);
  this.pre_first_acquisition_deaths+=Number(!this.pickedBefore&&info.died[p]);this.everPicked||=picked;
 }
}
module.exports={PhaseSafety};
