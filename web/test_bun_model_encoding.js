'use strict';
const assert = require('node:assert/strict');
const QQT = require('./sim.js');
const level = require('../scripts/eval_web_hunter.js').loadLevel();
const sim = new QQT.Sim(1536);
sim.reset(level, {nativeTrap:true,nativeItems:true});
sim.trapped = [30, 12];
sim.movementStatus = [1, 0];
for (const seat of [0, 1]) {
  const old = sim.encodeStateJAX(seat);
  const encoded = sim.encodeStateJAX(seat, 'bun24-trap-timer-v2');
  assert.equal(encoded.length, 24);
  for (let i = 0; i < 24; i++) {
    const extra = i === 9 ? sim.trapped[seat]/60 : i === 10 ? sim.trapped[1-seat]/60 : 0;
    assert.equal(encoded[i], old[i]+extra, 'Only declared native timer slots change');
  }
  const model = {meta:{bun_state_encoding:'bun24-trap-timer-v2'},inferEvery:1,obsShape:[24,13,15]};
  const input = QQT.TransformerModel.prototype._getLaggedInputs.call(model, sim, seat);
  assert.deepEqual(input.state, encoded);
  model.meta = {};
  assert.deepEqual(QQT.TransformerModel.prototype._getLaggedInputs.call(model, sim, seat).state, old);
}
sim.trapped = [0, 0];
assert.deepEqual(sim.encodeStateJAX(0, 'bun24-trap-timer-v2'), sim.encodeStateJAX(0), 'Slow glue is not sugar trap');
assert.throws(() => sim.encodeStateJAX(0, 'unknown-encoding'), /encoding/);
console.log('Versioned native Bun encoding, both seats and legacy compatibility passed');
