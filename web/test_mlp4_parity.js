'use strict';

// Python↔JS 固定入力 parity：web/sim.js::MLP4Model.forward が
// scripts/gen_mlp4_parity_fixture.py の float64 参照実装と数値一致（許容誤差内）
// かつ move/bomb の argmax が完全一致することを確認する。fixture はエクスポータ
// 契約（extract_mlp4 → pack_tensors、[out,in] 転置）を通した実モデルと同一の
// 前向き経路をコンパクトに再現している。
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const QQT = require('./sim.js');

const fixture = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'test_fixtures', 'mlp4_parity.json'), 'utf8'));

const model = new QQT.MLP4Model(fixture.doc);
const TOL = 1e-6;

let maxErr = 0;
fixture.cases.forEach((testCase, index) => {
  const obs = new Float32Array(testCase.obs);
  const out = model.forward(obs);

  const compare = (label, actual, expected) => {
    assert.strictEqual(actual.length, expected.length,
      `case ${index} ${label} length ${actual.length} != ${expected.length}`);
    for (let i = 0; i < expected.length; i++) {
      const err = Math.abs(actual[i] - expected[i]);
      if (err > maxErr) maxErr = err;
      assert.ok(err <= TOL,
        `case ${index} ${label}[${i}] err ${err} > ${TOL} `
        + `(js ${actual[i]} vs py ${expected[i]})`);
    }
  };

  compare('move', out.move, testCase.move);
  compare('bomb', out.bomb, testCase.bomb);

  const valueErr = Math.abs(out.value - testCase.value);
  if (valueErr > maxErr) maxErr = valueErr;
  assert.ok(valueErr <= TOL,
    `case ${index} value err ${valueErr} > ${TOL} (js ${out.value} vs py ${testCase.value})`);

  const argmax = (arr) => {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
  };
  assert.strictEqual(argmax(out.move), testCase.move_argmax,
    `case ${index} move argmax mismatch`);
  assert.strictEqual(argmax(out.bomb), testCase.bomb_argmax,
    `case ${index} bomb argmax mismatch`);
});

console.log(`网页 MLP4Model 数值/argmax parity 通过（${fixture.cases.length} 例，最大误差 ${maxErr.toExponential(2)}）`);
