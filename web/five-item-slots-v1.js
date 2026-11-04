'use strict';
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.QQTFiveSlots=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){

// Isolated browser ABI for models with seven skill actions.  The production
// simulator is unchanged; callers pass the returned requestedSlot to its
// existing _placeHeldItem(player, cell, requestedSlot) primitive.
const ACTION_NAMES = Object.freeze(['none', 'bomb', 'use_slot1', 'use_slot2', 'use_slot3', 'use_slot4', 'use_slot5']);

function itemCount(slot) { return slot && Number(slot.count) > 0 ? Number(slot.count) : 0; }

function abilityMask(sim, pid, masks) {
  const out = new Array(7).fill(0);
  out[0] = 1;
  out[1] = masks && masks.bm && masks.bm[pid] && masks.bm[pid][1] ? 1 : 0;
  const alive = !!(sim.alive && sim.alive[pid]);
  const trapped = !!(sim.trapped && sim.trapped[pid] > 0);
  const slots = (sim.itemSlots && sim.itemSlots[pid]) || [];
  if (!alive || trapped) out[1]=0;
  let place=true;
  if (sim.pos && sim.wall && sim.brick && sim.fuse && sim.fieldItem) {
    const row=Math.max(0,Math.min(12,Math.floor(sim.pos[2*pid]))),col=Math.max(0,Math.min(14,Math.floor(sim.pos[2*pid+1]))),cell=row*15+col;
    place=!sim.wall[cell]&&!sim.brick[cell]&&!sim.fuse[cell]&&!sim.fieldItem[cell];
  }
  if (alive && !trapped && place) {
    for (let i = 0; i < 5; i++) out[2 + i] = itemCount(slots[i]) > 0 && slots[i].item > 0 ? 1 : 0;
  }
  return out;
}

function decodeAbility(move, ability) {
  if (!Number.isInteger(move) || move < 0 || move > 4) throw new Error('invalid move');
  if (!Number.isInteger(ability) || ability < 0 || ability > 6) throw new Error('invalid five-slot ability');
  return { move, ability, bomb: ability === 1 ? 1 : 0, use: ability >= 2 ? 1 : 0,
    requestedSlot: ability >= 2 ? ability - 2 : null };
}

function useSlot(slots, index) {
  if (!Array.isArray(slots) || !Number.isInteger(index) || index < 0 || index >= 5) {
    return { accepted: false, slots: Array.isArray(slots) ? slots.map(x => ({...x})) : [] };
  }
  const out = slots.map(x => ({...x}));
  const slot = out[index];
  if (!slot || itemCount(slot) <= 0) return { accepted: false, slots: out };
  slot.count -= 1;
  if (slot.count <= 0) out.splice(index, 1);
  return { accepted: true, slots: out };
}

function engineAction(move, ability, requestedCell) {
  const d = decodeAbility(move, ability);
  // [move,bomb,use,controlled,requestedCell,requestedCell,requestedSlot]
  return [d.move, d.bomb, d.use, 0, requestedCell == null ? -1 : requestedCell,
    requestedCell == null ? -1 : requestedCell, d.requestedSlot == null ? -1 : d.requestedSlot];
}

function metadata(parentCheckpointSha256) {
  return { schema_version: 'five-item-slots-v1', action_schema: 'move-v1+five-item-slots-v1',
    ability_actions: 7, ability_encoding: ACTION_NAMES.slice(), max_selectable_slots: 5,
    parent_checkpoint_sha256: parentCheckpointSha256 || null,
    legacy_compatibility: { ability_actions: 3, use_item_maps_to: 'use_slot1' } };
}

return { ACTION_NAMES, abilityMask, decodeAbility, useSlot, engineAction, metadata };
});
