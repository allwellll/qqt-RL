'use strict';

(function tacticalArenaFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTTacticalArena = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildApi() {
  function clone(value) { return JSON.parse(JSON.stringify(value)); }

  function buildTacticalArena(base) {
    const level = clone(base);
    level.id = 240;
    level.name = 'Bun06 空砖近距竞技';
    level.category = '能力测试';
    level.crate_rate = 0;
    level.item_field = 0;
    level.no_items = true;
    level.brick = Array(level.brick.length).fill(0);
    if (level.layers && level.layers[1]) {
      level.layers[1] = level.layers[1].map((value, cell) => level.wall[cell] ? value : 0);
    }
    level.spawns = [
      [9.5, 4.5], [9.5, 4.5], [9.5, 4.5], [9.5, 4.5],
      [9.5, 8.5], [9.5, 8.5], [9.5, 8.5], [9.5, 8.5],
    ];
    level.bun_spawns = [level.spawns.slice(0, 4), level.spawns.slice(4, 8)];
    level.initial_stats = { hp: 1, bombs: 2, blast: 4, speed: 1.3 };
    return level;
  }

  return { buildTacticalArena };
});
