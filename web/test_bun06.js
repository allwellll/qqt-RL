#!/usr/bin/env node
'use strict';

const fs = require('fs');
const crypto = require('crypto');
const path = require('path');
const QQT = require('./sim.js');

function assert(ok, message) {
  if (!ok) throw new Error('ASSERT FAIL: ' + message);
}

const levels = JSON.parse(fs.readFileSync(path.join(__dirname, 'assets/maps/levels.json'), 'utf8'));
const elements = JSON.parse(fs.readFileSync(path.join(__dirname, 'assets/maps/elements.json'), 'utf8'));
const level = levels.find((item) => item.qqt_id === 806);
assert(level && level.bun && level.native_rule === 3, '806 抢包子规则元数据缺失');
assert(levels.length === 1 && level.id === 0 && level.qqt_id === 806,
  '纯 Bun 导出只含一图，内部数组 ID 归一化且官方地图 ID 保真');
assert(level.round_duration_ms === 240000, '原生对局时长必须是 240000ms');
assert(level.item_field === 1 && !level.no_items, '806 仅保留有道具版本');
assert(level.crate_rate === 0.582011, '806 道具掉落率');
assert(level.crate_super_fraction === 0.090909, '806 超级道具占比');
assert(level.tactical_item_fraction === 0.3, '806 战术道具占比');
assert(level.wall.reduce((sum, value) => sum + value, 0) === 50, '永久墙数量');
assert(level.brick.reduce((sum, value) => sum + value, 0) === 81, '可炸砖数量');

const map = fs.readFileSync(path.join(__dirname, 'assets/maps/bun06_8.map'));
assert(crypto.createHash('md5').update(map).digest('hex') === '13493e7cea40084836b0c3dc717b3279', 'map MD5');
assert(crypto.createHash('sha256').update(map).digest('hex') === '73572e6f07e4eae563b33bedf6beaa44b7edf2a6a2fe50df84af786594eb6c68', 'map SHA-256');
for (const elementId of [8001, 8002, 8003, 8004, 8005, 8006, 8009, 8010, 8011, 8012, 8013, 8028]) {
  const entry = elements[elementId];
  assert(entry, `元素 ${elementId} 元数据`);
  assert(fs.existsSync(path.join(__dirname, entry.file.replace(/^assets\//, 'assets/'))), `元素 ${elementId} PNG`);
}

const fresh = () => {
  const sim = new QQT.Sim(level.qqt_id);
  sim.reset(level);
  return sim;
};

const spawnKey = (row, column) => `${row},${column}`;
const redSpawns = new Set(level.bun_spawns[0].map(([row, column]) => spawnKey(row, column)));
const blueSpawns = new Set(level.bun_spawns[1].map(([row, column]) => spawnKey(row, column)));
for (let seed = 0; seed < 32; seed++) {
  const seeded = new QQT.Sim(seed);
  seeded.reset(level);
  const red = seeded.centerCell(0);
  const blue = seeded.centerCell(1);
  assert(redSpawns.has(spawnKey(red[0], red[1])), `红方出生组 seed=${seed}`);
  assert(blueSpawns.has(spawnKey(blue[0], blue[1])), `蓝方出生组 seed=${seed}`);
}

let sim = fresh();
assert(sim.isBun && sim.maxSteps === 2400, '规则/时长');
assert(sim.itemsEnabled && sim.crateRate === 0.582011, '有道具版本必须启用属性箱');
assert(sim.bunRespawnTicks === 100, '死亡后 10 秒复活');
assert(sim.bunBases.length === 2 && sim.bunBases[0][0] === 1 && sim.bunBases[0][1] === 4, '基地锚点');
assert(sim.bunStored[0][0] === 1 && sim.bunStored[1][1] === 1, '初始一人一包');
assert(sim.wall[1 * QQT.W + 4] === 1, '房屋实心角不能被规则层清空');
assert(sim.wall[2 * QQT.W + 5] === 0, '房屋十字通道必须可进入');

const canReach = (start, target) => {
  const queue = [start];
  const seen = new Set([spawnKey(start[0], start[1])]);
  while (queue.length) {
    const [row, column] = queue.shift();
    if (row === target[0] && column === target[1]) return true;
    for (const [dr, dc] of QQT.DIRS) {
      const nr = row + dr, nc = column + dc;
      const key = spawnKey(nr, nc);
      if (nr < 0 || nr >= QQT.H || nc < 0 || nc >= QQT.W || seen.has(key)) continue;
      if (sim.wall[nr * QQT.W + nc]) continue;
      seen.add(key);
      queue.push([nr, nc]);
    }
  }
  return false;
};
assert(canReach(sim.centerCell(0), [2, 9]), '清除可炸砖后红方可进入蓝方包子屋');
assert(canReach(sim.centerCell(1), [2, 5]), '清除可炸砖后蓝方可进入红方包子屋');

// 玩家0进入蓝方包子屋，自动取走蓝方基地包子；己方包子不能从己方基地取走。
sim.pos[0] = 2.5; sim.pos[1] = 9.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.bunCarried[0] === 1 && sim.bunStored[1][1] === 0, '从敌方基地取包');
assert(sim.playerMoveScale(0) === 0.5, '携包显著减速');
const carryBomb = sim.step([[QQT.MOVE_IDLE, 1], [QQT.MOVE_IDLE, 0]]);
assert(!carryBomb.placed[0] && sim.liveBombs(0) === 0, '携包时不能放泡');
sim.bunCarried[0] = -1;
sim.pos[0] = 2.5; sim.pos[1] = 5.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.bunCarried[0] === -1 && sim.bunStored[0][0] === 1, '不能拿己方基地库存');

// 把敌方包子带回己方包子屋，立即夺包获胜。
sim.bunStored[1][1] = 0;
sim.bunCarried[0] = 1;
sim.pos[0] = 2.5; sim.pos[1] = 5.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.done && sim.winner === 0 && sim.bunScore[0] === 1, '夺包结算');
assert(sim.bunStored[0][1] === 1 && sim._bunBaseTotal(0) === 2, '敌方包子进入己方库存');

// 携包死亡产生 BunID=1 散包；原属队伍可捡回并归还基地，不能增加夺包分。
sim = fresh();
sim.bunStored[1][1] = 0;
sim.bunCarried[0] = 1;
sim.pos[0] = 6.5; sim.pos[1] = 6.5;
sim._bunDrop(0);
const looseCell = 6 * QQT.W + 6;
assert(sim.bunLoose[looseCell * 2 + 1] === 1 && sim.bunCarried[0] === -1, '死亡散包');
sim.pos[0] = 7.5; sim.pos[1] = 0.5;
sim.pos[2] = 6.5; sim.pos[3] = 6.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.bunCarried[1] === 1 && sim.bunLoose[looseCell * 2 + 1] === 0, '原属队伍拾取散包');
sim.pos[2] = 2.5; sim.pos[3] = 9.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.bunStored[1][1] === 1 && sim.bunScore[1] === 0 && !sim.done, '己方包子归还不计夺包');

// 死亡不会结束对局，并在计时结束后复活。
sim.alive[0] = false;
sim.hp[0] = 0;
sim.bunRespawn[0] = 1;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.bunRespawn[0] === 0 && sim.alive[0] && !sim.done, '死亡复活');

// 原生隐藏格是可炸砖；爆炸后经过残骸时间开放通行并按掉落率生成道具。
sim = fresh();
const brickCell = 5 * QQT.W + 4;
const bombCell = 4 * QQT.W + 4;
assert(sim.brick[brickCell] === 1 && !sim.wall[bombCell] && !sim.brick[bombCell], '测试砖布局');
sim.pos[0] = 7.5; sim.pos[1] = 0.5;
sim.pos[2] = 7.5; sim.pos[3] = 14.5;
sim.fuse[bombCell] = 1;
sim.owner[bombCell] = 0;
sim.bombBlast[bombCell] = 2;
sim.tacticalItemFraction = 0;
sim.rng = () => 0;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.brick[brickCell] === 1 && sim.brickLinger[brickCell] > 0, '砖块进入炸毁残骸');
for (let i = 0; i < QQT.CFG.brickLingerTicks; i++) {
  sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
}
assert(sim.brick[brickCell] === 0, '砖块炸开后开放通行');
assert(sim.crate[brickCell] === 1, '有道具版本炸砖必须能掉落道具');
const bombsBeforePickup = sim.bombsCap[0];
sim.pos[0] = 5.5; sim.pos[1] = 4.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.crate[brickCell] === 0 && sim.bombsCap[0] > bombsBeforePickup, '玩家可以拾取炸出的道具');

// 香蕉皮和慢慢胶进入单格道具栏，按动作位放置；离开后才会武装，避免放置者原地触发。
sim = fresh();
const fieldCell = 7 * QQT.W + 8;
sim.pos[0] = 7.5; sim.pos[1] = 8.5;
sim.pos[2] = 4.5; sim.pos[3] = 14.5;
sim.crate[fieldCell] = 1;
sim.crateType[fieldCell] = QQT.CRATE_BANANA;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.heldItem[0] === QQT.ITEM_BANANA && sim.crate[fieldCell] === 0, '拾取香蕉皮进入道具栏');
sim.step([[QQT.MOVE_IDLE, 0, 1], [QQT.MOVE_IDLE, 0]]);
assert(sim.fieldItem[fieldCell] === QQT.ITEM_BANANA && !sim.fieldArmed[fieldCell], '香蕉皮放在脚下且不会立刻自触发');
sim.pos[0] = 7.5; sim.pos[1] = 6.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.fieldArmed[fieldCell] === 1, '放置者离开后香蕉皮武装');
sim.lastMoveDir[1] = QQT.MOVE_RIGHT;
sim.pos[2] = 7.5; sim.pos[3] = 8.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.movementStatus[1] === QQT.MOVE_STATUS_SLIDE && sim.fieldItem[fieldCell] === 0, '香蕉皮触发强制滑行');
const beforeSlideX = sim.pos[3];
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_LEFT, 0]]);
assert(sim.pos[3] > beforeSlideX, '强制滑行忽略反向输入并沿原朝向前进');
for (let i = 0; i < 8 && sim.movementStatus[1] === QQT.MOVE_STATUS_SLIDE; i++) {
  sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_LEFT, 0]]);
}
assert(sim.movementStatus[1] === QQT.MOVE_STATUS_NONE && sim.pos[3] < 10, '香蕉皮滑行撞到障碍后停止');

sim = fresh();
sim.pos[0] = 7.5; sim.pos[1] = 8.5;
sim.pos[2] = 4.5; sim.pos[3] = 14.5;
sim.crate[fieldCell] = 1;
sim.crateType[fieldCell] = QQT.CRATE_SLOW_GLUE;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.heldItem[0] === QQT.ITEM_SLOW_GLUE, '拾取慢慢胶进入道具栏');
sim.step([[QQT.MOVE_IDLE, 0, 1], [QQT.MOVE_IDLE, 0]]);
sim.pos[0] = 7.5; sim.pos[1] = 6.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
sim.pos[2] = 7.5; sim.pos[3] = 8.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.movementStatus[1] === QQT.MOVE_STATUS_SLOW && sim.movementStatusTicks[1] === 99, '慢慢胶减速 10 秒');
assert(sim.playerMoveScale(1) === 0.5, '慢慢胶速度倍率');
for (let i = 0; i < 99; i++) sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.movementStatus[1] === QQT.MOVE_STATUS_NONE && sim.playerMoveScale(1) === 1, '慢慢胶到期恢复速度');

sim = fresh();
sim.pos[0] = 7.5; sim.pos[1] = 8.5;
sim.crate[fieldCell] = 1;
sim.crateType[fieldCell] = QQT.CRATE_FAST_SHOE;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.movementStatus[0] === QQT.MOVE_STATUS_FAST && sim.movementStatusTicks[0] === 100, '超级鞋拾取后生效 10 秒');
assert(sim.playerMoveScale(0) === 1.6, '超级鞋速度倍率');

// 超时按双方基地现存包子总数比较，不按击杀或仅按夺包次数比较。
sim = fresh();
sim.bunStored[1][1] = 0;
sim.bunCarried[0] = 1;
sim.t = sim.maxSteps - 1;
sim.pos[0] = 6.5; sim.pos[1] = 6.5;
sim.pos[2] = 6.5; sim.pos[3] = 8.5;
sim.step([[QQT.MOVE_IDLE, 0], [QQT.MOVE_IDLE, 0]]);
assert(sim.done && sim.winner === 0, '超时按基地库存结算');

console.log('抢包子06 地图、碰撞、炸砖和规则状态测试通过');
