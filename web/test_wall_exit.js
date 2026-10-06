'use strict';
const assert = require('assert');
const baseline = process.argv[2] === '--baseline' ? process.argv[3] : null;
let Q = require('./sim');
if (baseline) {
  const Module = require('module'), { execFileSync } = require('child_process');
  const module = new Module(__filename);
  module._compile(execFileSync('git', ['show', `${baseline}:web/sim.js`], { encoding: 'utf8' }), __filename);
  Q = module.exports;
}
const level = require('./assets/maps/levels.json').find(l => l.qqt_id === 806);
const directions = [[Q.MOVE_UP, -1, 0], [Q.MOVE_DOWN, 1, 0], [Q.MOVE_LEFT, 0, -1], [Q.MOVE_RIGHT, 0, 1]];
const offsets = [0, 1, 6, 19, 20, 34, 39];
function scene(row = 5, col = 5, oy = 20, ox = 20) {
  const sim = new Q.Sim(19);
  sim.reset(level, { nativeItems: true, nativeTrap: true });
  for (const key of ['wall', 'brick', 'fuse', 'pushable', 'crate']) sim[key].fill(0);
  sim.wall[row * Q.W + col] = 1;
  sim.pos[0] = row + oy / 40; sim.pos[1] = col + ox / 40;
  sim.spdG[0] = 1; sim.nativeMove = null;
  return sim;
}
function reach(sim, move, target) {
  for (let frame = 0; frame < 60; frame++) {
    sim.frameStep(0, move, .02);
    const axis=move<Q.MOVE_LEFT?0:1;
    if (sim.centerCell(0).join(',') === target.join(',') &&
        Math.abs(sim.pos[axis]-(target[axis]+.5))<.025) return true;
  }
  return false;
}
let exits = 0;
const reproduction = directions.map(([move]) => ({ move, localCases: 0, stuck: 0, sideways: 0,
  mapCases: 0, mapStuck: 0, mapSideways: 0 }));
// Every neighboring wall layout, including diagonal walls intersecting leading corners.
const neighbors = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]];
for (const [move, dy, dx] of directions) for (let mask = 0; mask < 256; mask++) {
  if (mask & (1 << neighbors.findIndex(([r,c]) => r === dy && c === dx))) continue;
  for (const oy of offsets) for (const ox of offsets) {
    const sim = scene(5, 5, oy, ox), before = Array.from(sim.pos.slice(0,2));
    neighbors.forEach(([r,c], i) => { if (mask & (1 << i)) sim.wall[(5+r)*Q.W+5+c] = 1; });
    const reached=reach(sim, move, [5+dy,5+dx]), changed=sim.pos[dy ? 1 : 0]!==before[dy ? 1 : 0];
    const row=reproduction[move];row.localCases++;row.stuck+=!reached;row.sideways+=changed;
    if(!baseline){
      assert(reached, `wall exit move=${move} mask=${mask} offset=${oy},${ox}`);
      assert(!changed, 'exit stays on the requested axis');
    }
    exits++;
  }
}
// Real map wall element types, preserving all surrounding unbroken bricks and walls.
const wallTypes = new Set(); let mapExits = 0;
for (let cell = 0; cell < Q.N; cell++) if (level.wall[cell]) {
  const row = Math.floor(cell / Q.W), col = cell % Q.W;
  for (const [move, dy, dx] of directions) {
    const tr = row+dy, tc = col+dx, target = tr*Q.W+tc;
    if (tr<0 || tr>=Q.H || tc<0 || tc>=Q.W || level.wall[target] || level.brick[target]) continue;
    for (const oy of offsets) for (const ox of offsets) {
      const sim = scene(row,col,oy,ox);
      sim.wall.set(level.wall); sim.brick.set(level.brick);
      const before=Array.from(sim.pos.slice(0,2)), reached=reach(sim,move,[tr,tc]);
      const stats=reproduction[move];stats.mapCases++;stats.mapStuck+=!reached;
      stats.mapSideways+=sim.pos[dy ? 1 : 0]!==before[dy ? 1 : 0];
      if(!baseline)assert(reached, `806 wall ${cell}, direction ${move}, offset ${oy},${ox}`);
      wallTypes.add(level.layers[0][cell] || level.layers[1][cell]); mapExits++;
    }
  }
}
if(baseline){console.log(JSON.stringify({baseline,reproduction,wallTypes:[...wallTypes].sort()},null,2));process.exit(0);}
for (const [move,dy,dx] of directions) for (const kind of ['wall','brick','bomb','pushable','debris']) {
  for (const active of [false,true]) for (const oy of offsets) for (const ox of offsets) {
    const sim=scene(5,5,oy,ox), target=(5+dy)*Q.W+5+dx, before=Array.from(sim.pos.slice(0,2));
    if (kind==='bomb') sim.fuse[target]=30;
    else if(kind==='debris') { sim.brick[target]=1; sim.brickLinger[target]=2; }
    else sim[kind][target]=1;
    sim._nativeState(0).passActive=active; sim._nativeState(0).passDuration=10000;
    for(let n=0;n<20;n++) sim.frameStep(0,move,.02);
    assert.deepStrictEqual(Array.from(sim.pos.slice(0,2)),before, `blocked ${kind}, move=${move}, pass=${active}`);
  }
}
for(const [move,dy,dx]of directions) {
  const row=dy<0?0:dy>0?Q.H-1:5, col=dx<0?0:dx>0?Q.W-1:5;
  for(const oy of offsets)for(const ox of offsets){
    const sim=scene(row,col,oy,ox), before=Array.from(sim.pos.slice(0,2));
    for(let n=0;n<20;n++)sim.frameStep(0,move,.02);
    assert.deepStrictEqual(Array.from(sim.pos.slice(0,2)),before, 'wall exit cannot leave the map');
  }
}
for (const [move,dy,dx] of directions) {
  for (const active of [false,true]) for (const speed of [72,120,268,480,1600]) {
    const sim=scene(), st=sim._nativeState(0);
    sim._nativeSpeedPx=()=>speed; st.passActive=active; st.passDuration=10000;
    const beyond=(5+2*dy)*Q.W+5+2*dx; sim.brick[beyond]=1;
    assert(reach(sim,move,[5+dy,5+dx]), 'speed and expired/active passage never prevent wall exit');
    for(let n=0;n<30;n++) sim.frameStep(0,move,.1);
    assert(!sim.brick[sim.centerCell(0)[0]*Q.W+sim.centerCell(0)[1]], 'fast exit never crosses the next brick');
  }
  const sim=scene(), target=(5+dy)*Q.W+5+dx;
  sim.frameStep(0,move,.025);
  assert(sim._nativeState(0).wallExit, 'exit remains active during the first part of movement');
  const frame=JSON.parse(JSON.stringify(sim.snapshotReplay())), restored=new Q.Sim(5);
  restored.restoreReplay(frame);
  for(let n=0;n<14;n++){sim.frameStep(0,move,.025);restored.frameStep(0,move,.025);}
  assert.deepStrictEqual(Array.from(restored.pos),Array.from(sim.pos), 'replay restores exit progression');
  assert(!restored._nativeState(0).wallExit, 'exit permission expires at target centre');
  const current=sim.centerCell(0);assert.deepStrictEqual(current,[5+dy,5+dx]);
  const opposite=[Q.MOVE_DOWN,Q.MOVE_UP,Q.MOVE_RIGHT,Q.MOVE_LEFT][move];
  for(let n=0;n<20;n++)sim.frameStep(0,opposite,.025);
  assert.deepStrictEqual(sim.centerCell(0),current,'ground actor cannot return into the source wall');
  const dynamic=scene();dynamic.frameStep(0,move,.025);
  dynamic.fuse[target]=30;const before=Array.from(dynamic.pos);
  dynamic.frameStep(0,move,.1);
  assert.deepStrictEqual(Array.from(dynamic.pos),before,'new target bubble cancels the exit without tunnelling');
  assert(!dynamic._nativeState(0).wallExit);
  dynamic.fuse[target]=0;assert(reach(dynamic,move,[5+dy,5+dx]),'exit can retry when target becomes open');
  for (const kind of ['wall','brick','pushable']) {
    const obstructed=scene();obstructed.frameStep(0,move,.025);obstructed[kind][target]=1;
    const previous=Array.from(obstructed.pos);obstructed.frameStep(0,move,.1);
    assert.deepStrictEqual(Array.from(obstructed.pos),previous,`new ${kind} cancels the exit`);
  }
  const vanished=scene();vanished.frameStep(0,move,.025);vanished.wall[5*Q.W+5]=0;
  vanished.frameStep(0,move,.025);
  assert(!vanished._nativeState(0).wallExit,'source removal cancels old permission');
  const changed=scene();changed.frameStep(0,move,.025);
  changed.frameStep(0,opposite,.025);
  assert.equal(changed._nativeState(0).wallExit.move,opposite,'changed direction cannot retain the old exit');
  const relocated=scene();relocated.frameStep(0,move,.025);relocated.pos[0]=9.5;relocated.pos[1]=9.5;
  relocated.frameStep(0,move,.025);
  assert(!relocated._nativeState(0).wallExit,'teleported ground actor cannot retain an unrelated exit');
  const dead=scene();dead.frameStep(0,move,.025);dead._killPlayer(0);
  assert(!dead._nativeState(0).wallExit,'death clears the transition');
  restored.restoreReplay(new Q.Sim(19).snapshotReplay());
  assert(!restored._nativeState(0).wallExit,'old/ordinary replay clears stale permission');
  const fractional=scene();fractional.spdG[0]=.7;fractional.frameStep(0,move,.02);
  const saved=fractional.snapshotReplay(), fractionalReplay=new Q.Sim(7);
  fractionalReplay.restoreReplay(saved);
  for(let n=0;n<10;n++){
    fractional.frameStep(0,move,.02);fractionalReplay.frameStep(0,move,.02);
    assert.deepStrictEqual(Array.from(fractionalReplay.pos),Array.from(fractional.pos),'fractional pixel replay must stay identical');
  }
  saved.nativeWallExits[0].to=0;
  assert.notEqual(fractionalReplay._nativeState(0).wallExit.to,0,'restored exit owns its state');
}
// Forced banana slides clear a single wall corner by the minimum perpendicular
// alignment, then keep their original direction. Exercise both sides and all
// four directions with the opposite input held to prove steering is ignored.
function slideCorner(direction, side) {
  const vectors = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const [dy, dx] = vectors[direction];
  const perp = direction < 2 ? 1 : 0;
  const sideOffset = side < 0 ? 5 : 35;
  const sim = scene(5, 5, direction < 2 ? 20 : sideOffset, direction < 2 ? sideOffset : 20);
  sim.wall[5 * Q.W + 5] = 0;
  const aheadRow = 5 + dy, aheadCol = 5 + dx;
  const sideRow = aheadRow + (direction < 2 ? 0 : side);
  const sideCol = aheadCol + (direction < 2 ? side : 0);
  sim.wall[sideRow * Q.W + sideCol] = 1;
  sim._setMovementStatus(0, Q.MOVE_STATUS_SLIDE, 0);
  sim.slideDir[0] = direction;
  const input = [Q.MOVE_DOWN, Q.MOVE_UP, Q.MOVE_RIGHT, Q.MOVE_LEFT][direction];
  const before = Array.from(sim.pos.slice(0, 2));
  for (let i = 0; i < 5; i++) sim.frameStep(0, input, .02);
  const perpendicular = sim.pos[perp] * 40;
  assert(Math.abs(perpendicular - 220) <= 1,
    `single-side corner direction=${direction} side=${side} aligns to cell center`);
  for (let i = 0; i < 15; i++) sim.frameStep(0, input, .02);
  const axis = direction < 2 ? 0 : 1;
  assert((sim.pos[axis] - before[axis]) * (dy || dx) > .5,
    `single-side corner direction=${direction} keeps sliding along slideDir`);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_SLIDE,
    `single-side corner direction=${direction} remains forced slide`);
  return sim;
}
for (const direction of [Q.MOVE_UP, Q.MOVE_DOWN, Q.MOVE_LEFT, Q.MOVE_RIGHT]) {
  for (const side of [-1, 1]) slideCorner(direction, side);
}

// Two blocked leading corners form a closed wall face: no perpendicular escape
// or tunnelling is allowed, and the forced status is cleared after the stop.
for (const direction of [Q.MOVE_UP, Q.MOVE_DOWN, Q.MOVE_LEFT, Q.MOVE_RIGHT]) {
  const [dy, dx] = [[-1, 0], [1, 0], [0, -1], [0, 1]][direction];
  const sim = scene(5, 5, 20, 20);
  sim.wall[5 * Q.W + 5] = 0;
  const aheadRow = 5 + dy, aheadCol = 5 + dx;
  const side = direction < 2 ? [[aheadRow, aheadCol - 1], [aheadRow, aheadCol + 1]]
    : [[aheadRow - 1, aheadCol], [aheadRow + 1, aheadCol]];
  side.push([aheadRow, aheadCol]);
  for (const [r, c] of side) sim.wall[r * Q.W + c] = 1;
  sim._setMovementStatus(0, Q.MOVE_STATUS_SLIDE, 0); sim.slideDir[0] = direction;
  const before = Array.from(sim.pos.slice(0, 2));
  const input = [Q.MOVE_DOWN, Q.MOVE_UP, Q.MOVE_RIGHT, Q.MOVE_LEFT][direction];
  sim.frameStep(0, input, .02);
  assert.deepStrictEqual(Array.from(sim.pos.slice(0, 2)), before, `closed corner direction=${direction} stays put`);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_NONE, `closed corner direction=${direction} clears slide`);
}

// A wall or bomb introduced after alignment stops the slide at the obstacle;
// it cannot tunnel through a dynamic target or change to the perpendicular axis.
for (const kind of ['wall', 'bomb']) {
  const sim = slideCorner(Q.MOVE_RIGHT, -1);
  const target = 5 * Q.W + 9;
  if (kind === 'wall') sim.wall[target] = 1;
  else sim.fuse[target] = 30;
  const alignedY = sim.pos[0];
  for (let i = 0; i < 30; i++) sim.frameStep(0, Q.MOVE_LEFT, .02);
  assert.equal(sim.pos[0], alignedY, `dynamic ${kind} keeps the aligned perpendicular coordinate`);
  assert(sim.pos[1] < 9, `dynamic ${kind} stops before the target cell`);
}

// The perpendicular alignment side must also remain traversable. Check both
// corners in each slide direction with obstacles inserted before alignment.
for (const direction of [Q.MOVE_UP, Q.MOVE_DOWN, Q.MOVE_LEFT, Q.MOVE_RIGHT]) {
  for (const side of [-1, 1]) for (const kind of ['wall', 'bomb']) {
    const [dy, dx] = [[-1, 0], [1, 0], [0, -1], [0, 1]][direction];
    const offset = side < 0 ? 4 : 36;
    const sim = scene(5, 5, direction < 2 ? 20 : offset, direction < 2 ? offset : 20);
    sim.wall[5 * Q.W + 5] = 0;
    sim.wall[(5 + dy) * Q.W + 5 + dx] = 1;
    const target = (5 + (direction < 2 ? 0 : side)) * Q.W + 5 + (direction < 2 ? side : 0);
    sim[kind === 'wall' ? 'wall' : 'fuse'][target] = kind === 'wall' ? 1 : 30;
    sim._setMovementStatus(0, Q.MOVE_STATUS_SLIDE, 0); sim.slideDir[0] = direction;
    const before = Array.from(sim.pos.slice(0, 2));
    const frame = JSON.parse(JSON.stringify(sim.snapshotReplay(null)));
    const restored = new Q.Sim(99); restored.restoreReplay(frame);
    for (const s of [sim, restored]) {
      s.frameStep(0, Q.MOVE_IDLE, .02);
      assert.deepStrictEqual(Array.from(s.pos.slice(0, 2)), before,
        `alignment side ${kind} direction=${direction} side=${side} blocks original and replay`);
      assert.equal(s.movementStatus[0], Q.MOVE_STATUS_NONE);
    }
  }
}

// A replay taken mid-slide resumes at the same aligned pixel and continues in
// the same direction, including the wall-corner permission state.
{
  const original = slideCorner(Q.MOVE_RIGHT, 1);
  const frame = JSON.parse(JSON.stringify(original.snapshotReplay(null)));
  const restored = new Q.Sim(99); restored.restoreReplay(frame);
  assert.deepStrictEqual(Array.from(restored.pos), Array.from(original.pos), 'slide corner replay restores aligned pixel');
  assert.equal(restored.movementStatus[0], original.movementStatus[0], 'slide corner replay restores status');
  assert.equal(restored.slideDir[0], original.slideDir[0], 'slide corner replay restores direction');
  for (let i = 0; i < 20; i++) {
    original.frameStep(0, Q.MOVE_LEFT, .02); restored.frameStep(0, Q.MOVE_LEFT, .02);
  }
  assert(restored.pos[1] > frame.pos[1] && original.pos[1] > frame.pos[1], 'slide corner replay continues forward');
}
{
  const sim=scene();sim._setMovementStatus(0,Q.MOVE_STATUS_SLIDE,0);sim.slideDir[0]=Q.MOVE_RIGHT;
  assert(reach(sim,Q.MOVE_LEFT,[5,6]),'forced banana motion uses its own direction when exiting');
  assert(!sim._nativeState(0).wallExit);
}
{
  const sim=scene();sim.reset(level,{nativeTrap:true,teams:[0,1,0,1]});
  sim._nativeState(3).wallExit={from:80,to:81,move:3,cross:220};
  sim.restoreReplay(new Q.Sim(19).snapshotReplay());
  assert(!('nativeWallExits' in sim.snapshotReplay()),'removed players cannot leak an exit into the next replay');
}
for (const [move,dy,dx] of directions) {
  const sim=scene();sim.wall[5*Q.W+5]=0;
  sim.wall[(5+dy)*Q.W+5+dx]=1;
  assert(!reach(sim,move,[5+dy,5+dx]),'ordinary ground movement cannot enter a wall');
  sim.wall[(5+dy)*Q.W+5+dx]=0;sim.brick[5*Q.W+5]=1;
  for(let n=0;n<4;n++)sim.frameStep(0,move,.025);
  assert(!sim._nativeState(0).wallExit,'unbroken brick occupancy does not grant wall exit permission');
}
console.log(JSON.stringify({ exits, mapExits, wallTypes: [...wallTypes].sort(), reproduction }));
console.log('Wall exits: four directions, all local wall layouts, 806 wall types and blocked targets passed');
