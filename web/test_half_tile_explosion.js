'use strict';

const assert = require('assert');
const Q = require('./sim.js');

const N = Q.H * Q.W;
const IDLE = [Q.MOVE_IDLE, 0, 0, 1];
const cell = (r, c) => r * Q.W + c;

function scene(y, x) {
  const sim = new Q.Sim(731);
  sim.reset('open', { nativeItems: true, nativeTrap: true });
  for (const key of ['wall', 'brick', 'cover', 'bush', 'crate', 'fuse', 'blastLinger',
    'horzLinger', 'vertLinger', 'fieldItem']) sim[key].fill(0);
  sim.pos.set([y, x, 12.5, 12.5]);
  sim.invuln.fill(0);
  sim.spawnProtection.fill(0);
  return sim;
}

function masks(cells) {
  const covered = new Uint8Array(N);
  for (const [r, c] of cells) covered[cell(r, c)] = 1;
  return covered;
}

function hitTick(sim, groups, reverse = false) {
  const ordered = reverse ? groups.slice().reverse() : groups.slice();
  const covered = masks(ordered.flatMap((group) => group.cells));
  const sources = ordered.map((group) => {
    const sourceCovered = masks(group.cells);
    return {
      cell: cell(group.cells[0][0], group.cells[0][1]),
      owner: group.owner,
      causeMask: 1 << group.owner,
      covered: sourceCovered,
      horzCovered: sourceCovered,
      vertCovered: sourceCovered,
    };
  });
  sim._resolveExplosions = () => ({ covered, triggered: new Uint8Array(N), sources });
  return sim.step([IDLE, IDLE]);
}

function survived(sim, message) {
  assert(sim.alive[0] && sim.trapped[0] === 0, message);
}

function trapped(sim, message) {
  assert(sim.alive[0] && sim.trapped[0] > 0, message);
}

function realBomb(sim, r, c, owner, fuse = 1, blast = 1) {
  const i = cell(r, c);
  sim.fuse[i] = fuse;
  sim.owner[i] = owner;
  sim.bombBlast[i] = blast;
  sim.bombStyle[i] = owner === 0 ? sim.playerBombStyle : -1;
  return i;
}

function infoShape(info) {
  return {
    covered: Array.from(info.covered),
    triggered: Array.from(info.triggered),
    died: Array.from(info.died),
    physicalDamageSource: info.physicalDamageSource.map((row) => row.slice()),
    causalDamageSource: info.causalDamageSource.map((row) => row.slice()),
  };
}

// 横向边界：9%/10% 仍属于半身豁免；11% 已离开豁免带，单侧覆盖命中。
for (const [x, grace, centerSide] of [[5.91, true, null], [5.90, true, null], [5.89, false, 0], [6.0, true, null], [6.09, true, null], [6.11, false, 1]]) {
  for (const side of [[[5, 5]], [[5, 6]]]) {
    const sim = scene(5.5, x);
    hitTick(sim, [{ owner: 1, cells: side }]);
    const sideIndex = side[0][1] === 5 ? 0 : 1;
    if (grace || sideIndex !== centerSide) survived(sim, `horizontal ${x} tile boundary coverage ${JSON.stringify(side)} must survive`);
    else trapped(sim, `horizontal ${x} tile center-cell coverage must hit outside the 0.10-tile exemption`);
  }
  if (grace) {
    const sim = scene(5.5, x);
    hitTick(sim, [{ owner: 1, cells: [[5, 5]] }, { owner: 1, cells: [[5, 6]] }]);
    trapped(sim, `horizontal half-tile x=${x} must be hit when both sides are covered in one tick`);
  }
}

// 纵向边界与横向完全对称。
for (const [y, grace, centerSide] of [[5.91, true, null], [5.90, true, null], [5.89, false, 0], [6.0, true, null], [6.09, true, null], [6.11, false, 1]]) {
  for (const side of [[[5, 5]], [[6, 5]]]) {
    const sim = scene(y, 5.5);
    hitTick(sim, [{ owner: 1, cells: side }]);
    const sideIndex = side[0][0] === 5 ? 0 : 1;
    if (grace || sideIndex !== centerSide) survived(sim, `vertical ${y} tile boundary coverage ${JSON.stringify(side)} must survive`);
    else trapped(sim, `vertical ${y} tile center-cell coverage must hit outside the 0.10-tile exemption`);
  }
  if (grace) {
    const sim = scene(y, 5.5);
    hitTick(sim, [{ owner: 1, cells: [[5, 5]] }, { owner: 1, cells: [[6, 5]] }]);
    trapped(sim, `vertical half-tile y=${y} must be hit when both sides are covered in one tick`);
  }
}

// Just beyond the threshold is not covered by the numeric tolerance.
for (const axis of [0, 1]) for (const sign of [-1, 1]) for (const extra of [0.000001, 0.00005]) {
  const point = [5.5, 5.5]; point[axis] = 6 + sign * (0.10 + extra);
  const sim = scene(...point);
  const center = [Math.floor(point[0]), Math.floor(point[1])];
  hitTick(sim, [{ owner: 1, cells: [center] }]);
  trapped(sim, `just outside 10% axis=${axis} sign=${sign} extra=${extra} receives a normal hit`);
}

// 同 tick 聚合不依赖爆炸源遍历顺序，且两侧来源都保留物理归因。
for (const reverse of [false, true]) {
  const sim = scene(5.5, 6.0);
  const info = hitTick(sim, [
    { owner: 0, cells: [[5, 5]] },
    { owner: 1, cells: [[5, 6]] },
  ], reverse);
  trapped(sim, `same-tick result must not depend on source order reverse=${reverse}`);
  assert.deepStrictEqual(info.physicalDamageSource[0], [true, true], 'both covering owners contribute to the hit');
}

// 错开 tick 不累计：上一 tick 的单侧覆盖/余焰不能与下一 tick 的另一侧拼成命中。
{
  const sim = scene(5.5, 6.0);
  hitTick(sim, [{ owner: 1, cells: [[5, 5]] }]);
  survived(sim, 'first one-sided tick survives');
  hitTick(sim, [{ owner: 1, cells: [[5, 6]] }]);
  survived(sim, 'opposite side on a later tick must not accumulate');
}

// 完整站在格中心时，所在格覆盖仍然命中。
{
  const sim = scene(5.5, 5.5);
  hitTick(sim, [{ owner: 1, cells: [[5, 5]] }]);
  trapped(sim, 'centered player must be hit by the centered cell');
}

// 角落规则：同时明显跨行、跨列时占四格；四种象限都必须覆盖四格才命中，少一格仍安全。
for (const [y, x] of [[6.0, 6.0], [6.0, 5.0], [5.0, 6.0], [5.0, 5.0]]) {
  const rows = y === 6.0 ? [5, 6] : [4, 5];
  const cols = x === 6.0 ? [5, 6] : [4, 5];
  const cornerCells = rows.flatMap((r) => cols.map((c) => [r, c]));
  for (let omitted = 0; omitted < cornerCells.length; omitted++) {
    const sim = scene(y, x);
    hitTick(sim, [{ owner: 1, cells: cornerCells.filter((_, i) => i !== omitted) }]);
    survived(sim, `four-cell corner ${y},${x} must survive when corner cell ${omitted} is not covered`);
  }
  const sim = scene(y, x);
  hitTick(sim, [{ owner: 1, cells: cornerCells }]);
  trapped(sim, `four-cell corner ${y},${x} must be hit when all four cells are covered in one tick`);
}

// 出生保护优先于完整的双侧覆盖。
{
  const sim = scene(5.5, 6.0);
  sim.invuln[0] = 2;
  sim.spawnProtection[0] = 2;
  hitTick(sim, [{ owner: 1, cells: [[5, 5], [5, 6]] }]);
  survived(sim, 'spawn protection must still reject a complete half-tile hit');
}

// 真实物理链路：两颗相邻泡从边界两侧同 tick 爆炸，不能依赖替换
// _resolveExplosions；两位来源都必须出现在物理归因中。
{
  const sim = scene(5.5, 6.0);
  const left = realBomb(sim, 5, 5, 0);
  const right = realBomb(sim, 5, 6, 1);
  const info = sim.step([IDLE, IDLE]);
  trapped(sim, 'real adjacent bombs must hit a boundary-spanning player');
  assert(info.triggered[left] && info.triggered[right], 'both real boundary bombs trigger in one tick');
  assert.deepStrictEqual(info.physicalDamageSource[0], [true, true],
    'real same-tick boundary bombs preserve both physical sources');
  assert.deepStrictEqual(info.causalDamageSource[0], [true, true],
    'real same-tick boundary bombs preserve both causal sources');
}

// 真实连锁：第一颗泡触发第二颗泡。第二颗的物理所有者仍保留，
// 但因果来源应追溯到第一颗泡的 owner。
{
  const sim = scene(5.5, 6.0);
  const anchor = realBomb(sim, 5, 5, 0, 1, 1);
  const chained = realBomb(sim, 5, 6, 1, 8, 1);
  const info = sim.step([IDLE, IDLE]);
  trapped(sim, 'real chained boundary bombs must hit on the chain tick');
  assert(info.triggered[anchor] && info.triggered[chained], 'real chain triggers the pending adjacent bomb');
  assert(info.physicalDamageSource[0][0] && info.physicalDamageSource[0][1],
    'real chain records both physical bomb owners');
  assert(info.causalDamageSource[0][0], 'real chain records the initiating causal owner');
  assert(!info.causalDamageSource[0][1], 'real chain does not misattribute the chained bomb as initiator');
}

// 真实四格角落：四颗来源覆盖四个接触格，缺一格时不会被判为完整命中。
{
  const cells = [[5, 5], [5, 6], [6, 5], [6, 6]];
  const sim = scene(6.0, 6.0);
  for (let i = 0; i < cells.length; i++) realBomb(sim, cells[i][0], cells[i][1], i & 1, 1, 1);
  const info = sim.step([IDLE, IDLE]);
  trapped(sim, 'real four-source corner coverage must hit');
  assert(info.physicalDamageSource[0][0] && info.physicalDamageSource[0][1],
    'real four-source corner retains both owner classes');
  assert(info.causalDamageSource[0][0] && info.causalDamageSource[0][1],
    'real four-source corner retains both causal owner classes');
}

// 待爆真实泡的 snapshot/restore：先保存 fuse>0，再逐步到同一爆炸 tick，
// 覆盖、触发和两类归因必须逐项一致。
{
  const original = scene(5.5, 6.0);
  realBomb(original, 5, 5, 0, 2, 1);
  realBomb(original, 5, 6, 1, 9, 1);
  const frame = JSON.parse(JSON.stringify(original.snapshotReplay(null)));
  const restored = new Q.Sim(99).restoreReplay(frame);
  const firstA = original.step([IDLE, IDLE]);
  const firstB = restored.step([IDLE, IDLE]);
  assert.deepStrictEqual(infoShape(firstA), infoShape(firstB),
    'snapshot/restore keeps pending real-bomb step deterministic');
  const secondA = original.step([IDLE, IDLE]);
  const secondB = restored.step([IDLE, IDLE]);
  assert.deepStrictEqual(infoShape(secondA), infoShape(secondB),
    'snapshot/restore keeps real chain explosion deterministic');
  assert(original.trapped[0] > 0 && restored.trapped[0] > 0,
    'restored pending real bombs hit the same boundary player');
}

// 快照恢复后不引入跨 tick 隐状态；同一帧、同一覆盖得到相同结果。
{
  const cornerCells = [[5, 5], [5, 6], [6, 5], [6, 6]];
  const original = scene(6.0, 6.0);
  const frame = JSON.parse(JSON.stringify(original.snapshotReplay(null)));
  const restored = new Q.Sim(99).restoreReplay(frame);
  const a = hitTick(original, [{ owner: 1, cells: cornerCells }]);
  const b = hitTick(restored, [{ owner: 1, cells: cornerCells }]);
  assert.deepStrictEqual(
    { trapped: original.trapped, died: a.died, physical: a.physicalDamageSource },
    { trapped: restored.trapped, died: b.died, physical: b.physicalDamageSource },
    'snapshot/replay restoration must preserve deterministic half-tile damage',
  );
}

// Boundary tolerance survives snapshot/replay and does not widen after restore.
for (const [x, grace] of [[5.9, true], [5.89, false]]) {
  const original = scene(5.5, x);
  const frame = JSON.parse(JSON.stringify(original.snapshotReplay(null)));
  const restored = new Q.Sim(99).restoreReplay(frame);
  for (const sim of [original, restored]) {
    hitTick(sim, [{ owner: 1, cells: [[5, 5]] }]);
    if (grace) survived(sim, `replay preserves the ${x} tile half-tile exemption`);
    else trapped(sim, `replay preserves the ${x} tile hard hit`);
  }
}

console.log('half-tile explosion aggregation, symmetry, corner, protection and replay checks passed');
