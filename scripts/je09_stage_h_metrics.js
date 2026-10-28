'use strict';
// Passive additive fields on existing protocols; no new scenes or actions.
const fs = require('fs'), Q = require('../web/sim.js');
const inference = require('./je05_web_hunter.js');
const topology = require('./je08_web_diagnostics.js');
const originalLoad = inference.loadModel;
const observations = [];
inference.loadModel = async function (...args) {
  const loaded = await originalLoad(...args), act = loaded.model.act.bind(loaded.model);
  loaded.model.act = async function (sim, team, ...rest) {
    sim._hLearnerSeat = team;
    return act(sim, team, ...rest);
  };
  return loaded;
};
function homeDistance(s, team, extraBlocked = false) {
  const base = s.bunBases[team], goals = [], blocked = Array.from(s.wall, (v, i) =>
    Number(Boolean(v || s.brick[i] || (extraBlocked && (s.fuse[i] > 0 || s.blastLinger[i] > 0 || s.fieldItem[i] === 2)))));
  for (let r = base[0]; r < base[0]+3; r++) for (let c = base[1]; c < base[1]+3; c++) goals.push(r*Q.W+c);
  return topology.distance(blocked, new Uint8Array(Q.H*Q.W), goals);
}
const originalStep = Q.Sim.prototype.step;
Q.Sim.prototype.step = function (actions) {
  const team = this._hLearnerSeat;
  if (team !== 0 && team !== 1) return originalStep.call(this, actions);
  const enemy = 1-team;
  if (!this._hItems) {
    this._hItems = {team, custody:false, glue:new Map(), slide:null,
      glue_placed:0, glue_attributed_hit:0, banana_events:[], captures:0};
    observations.push(this._hItems);
  }
  const h = this._hItems, cell = this.centerCell(team), index = cell[0]*Q.W+cell[1];
  const before = {tick:this.t, carry:this.bunCarried.slice(), alive:this.alive.slice(),
    trap:this.trapped.slice(), status:this.movementStatus.slice(), field:Array.from(this.fieldItem),
    owners:Array.from(this.fieldOwner), armed:Array.from(this.fieldArmed),
    brick:Array.from(this.brick), ownStock:this.bunStored[team][enemy],
    item:this.itemSlots[team]?.[0]?.item, distance:h.custody ? homeDistance(this, team)[index] : null};
  const info = originalStep.call(this, actions), now = this.centerCell(team), at = now[0]*Q.W+now[1];
  const foe = this.centerCell(enemy), foeCell = foe[0]*Q.W+foe[1];
  const holding = this.alive[team] && this.bunCarried[team] === enemy && this.trapped[team] === 0;
  const continuity = h.custody && before.carry[team] === enemy && holding;
  if (continuity && actions[team]?.[2] && before.item === 2 && !before.field[index] &&
      this.fieldItem[index] === 2 && this.fieldOwner[index] === team) {
    h.glue_placed++;
    h.glue.set(index, {tick:before.tick, owner:team});
  }
  if (continuity && h.glue.has(foeCell) && before.field[foeCell] === 2 &&
      before.owners[foeCell] === team && before.armed[foeCell] && this.fieldItem[foeCell] === 0 &&
      before.alive[enemy] && this.alive[enemy] && before.trap[enemy] === 0 && this.trapped[enemy] === 0 &&
      before.status[enemy] !== 1 && this.movementStatus[enemy] === 1) h.glue_attributed_hit++;
  for (const key of h.glue.keys()) if (this.fieldItem[key] !== 2 || this.fieldOwner[key] !== team) h.glue.delete(key);
  const banana = before.field[at] === 1 && before.armed[at] && this.fieldItem[at] === 0 && this.movementStatus[team] === 2;
  if (banana && continuity && !h.slide) h.slide = {tick:before.tick, source_cell:at,
    source_owner:before.owners[at], before_distance:before.distance, valid:true};
  if (h.slide) {
    h.slide.valid &&= continuity && !before.brick.some((v, i) => v !== this.brick[i]);
    if (this.movementStatus[team] !== 2 || !holding) {
      const distance = homeDistance(this, team)[at];
      h.banana_events.push({...h.slide, end_tick:this.t, after_distance:distance,
        net_home_progress:h.slide.before_distance-distance, alive:this.alive[team], still_carrying:holding,
        successful:h.slide.valid && holding && distance < h.slide.before_distance});
      h.slide = null;
    }
  }
  if (this.bunStored[team][enemy] > before.ownStock) h.captures++;
  const acquired = before.carry[team] < 0 && this.bunCarried[team] === enemy;
  h.custody = (h.custody || acquired) && holding;
  return info;
};
const originalWrite = fs.writeFileSync;
fs.writeFileSync = function (path, data, ...args) {
  if (typeof data === 'string' && data.startsWith('{')) {
    try {
      const value = JSON.parse(data);
      if (value.results && observations.length) {
        value.carry_item_observations = {passive:true, existing_protocol_unchanged:true,
          initialized_carry_excluded:true, episodes:observations.map(h => ({...h, glue:undefined, slide:undefined}))};
        data = JSON.stringify(value);
      }
    } catch (_) { /* Not a result JSON. */ }
  }
  return originalWrite.call(this, path, data, ...args);
};
