'use strict';
// Genuine seeded item-rule game states, both seats; exports parity inputs, not performance evidence.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const Q = process.env.QQT_PARITY_RULE === 'bun-native-v4-facing' ? require('../web/sim-bun-native-v4-facing') : require('../web/sim-bun-native-v3-items');
const R = require('../web/r3_model_inputs');
const F = require('../web/five-item-slots-model-v1');
const {buildTacticalArena} = require('../web/tactical_arena');
const [modelFile, outputFile] = process.argv.slice(2);
const doc = JSON.parse(fs.readFileSync(modelFile));
assert(doc.meta.rule_version === 'bun-native-v3-items' || (doc.meta.r4_mode === 0 && /^[0-9a-f]{64}$/.test(doc.meta.r4_full_checkpoint_sha256)), 'hash-verified native items export required');
R.installSim(Q);
const model = F.dispatch(Q, doc, d => R.attach(new Q.TransformerModel(d)));
const levels = JSON.parse(fs.readFileSync(require('node:path').join(__dirname, '../web/assets/maps/levels.json')));
const normal = levels.find(x => x.qqt_id === 806), samples = [];
const plain = value => ({move: Array.from(value.move), bomb: Array.from(value.bomb), value: value.value});
const zero = plain(model.forward(new Float32Array(25 * 195), new Float32Array(doc.meta.state_dim)));
for (const [name, level] of [['normal', normal], ['duel', buildTacticalArena(normal)]]) {
  const sim = new Q.Sim(20261107), rng = Q.mulberry32(20261107);
  sim.reset(level, {nativeItems: true, nativeTrap: true});
  for (let t = 0; t <= 36 && !sim.done; t++) {
    if ([0, 12, 36].includes(t)) for (const seat of [0, 1]) {
      const {obs, state} = model._getLaggedInputs(sim, seat);
      const masks = sim.legalMask();
      const ability = model._abilityMask ? model._abilityMask(sim, masks, seat) : masks.bm[seat];
      const result = plain(model.forward(obs, state));
      assert([...result.move, ...result.bomb, result.value].every(Number.isFinite));
      samples.push({map: name, tick: t, seat, obs: Array.from(obs), state: Array.from(state),
        masks: {move: Array.from(masks.mm[seat]), bomb: Array.from(ability)}, js: result});
    }
    const masks = sim.legalMask();
    const actions = [0, 1].map(seat => {
      const legal = Array.from(masks.mm[seat]).flatMap((v, i) => v ? [i] : []);
      return [legal[Math.floor(rng() * legal.length)], Number(masks.bm[seat][1] && rng() < .2), 0];
    });
    sim.step(actions);
  }
}
fs.writeFileSync(outputFile, JSON.stringify({ruleset: Q.RULE_VERSION, reference_zero: zero, samples}) + '\n');
console.log(JSON.stringify({status: 'PASS', parity_states: samples.length, purpose: 'export integration only'}));
