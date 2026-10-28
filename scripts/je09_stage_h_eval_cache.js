'use strict';
// Cache only a pure distance query; keys contain every argument byte.
const topology = require('./je08_web_diagnostics.js');
const original = topology.distance, cache = new Map();
topology.distance = function (wall, brick, goals, h, w) {
  const key = JSON.stringify([Array.from(wall), Array.from(brick), Array.from(goals), h || 13, w || 15]);
  if (!cache.has(key)) {
    if (cache.size > 256) cache.clear();
    cache.set(key, original(wall, brick, goals, h, w));
  }
  return cache.get(key).slice();
};
