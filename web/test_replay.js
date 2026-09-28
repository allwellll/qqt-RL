'use strict';

const assert = require('assert');
const replay = require('./replay.js');

const doc = {
  schema: 'qqt.replay/v1',
  meta: { id: 'demo', model_id: 'm', opponent: 'bun.tactical_v2', seed: 7, tick_ms: 100, source_sha256: 'a'.repeat(64) },
  summary: { ticks: 2, model_bombs: 1, rule_bombs: 0, model_unique_cells: 2, rule_unique_cells: 1, winner: null },
  actions: [[0, 1, 0, 4, 0, 0], [3, 0, 1, 4, 0, 0]],
};
const parsed = replay.validateReplay(doc);
assert.strictEqual(parsed.actions.length, 2);
assert.strictEqual(replay.frameIndexAt(0, 2), 0);
assert.strictEqual(replay.frameIndexAt(100, 2), 1);
assert.strictEqual(replay.frameIndexAt(300, 2), 1);
assert.throws(() => replay.validateReplay({ ...doc, actions: [] }), /actions/);
assert(JSON.stringify(doc).length < 600, '动作流回放应紧凑，不能逐帧重复完整状态');
console.log('网页离线对局回放紧凑格式与时间轴契约通过');
