'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const authority = require('../scripts/je03_original_hunter_20261011.js');

test('authority rejects battle models and observation adapters', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bun-authority-'));
  const file = path.join(dir, 'model.json');
  try {
    fs.writeFileSync(file, JSON.stringify({meta:{rule:'battle',arch:'transformer'}}));
    await assert.rejects(authority.loadModel(file), /native Bun rule/);
    fs.writeFileSync(file, JSON.stringify({meta:{rule:'bun',arch:'transformer',obs_shape:[14,13,15]}}));
    await assert.rejects(authority.loadModel(file), /native Bun24/);
    fs.writeFileSync(file, JSON.stringify({meta:{rule:'bun',arch:'transformer',obs_shape:[24,13,15],input_encoding:'native_battle_extended_v5'}}));
    await assert.rejects(authority.loadModel(file), /adapters/);
  } finally {
    fs.rmSync(dir, {recursive:true,force:true});
  }
});

test('async authority retains the real web episode implementation', () => {
  const converted = authority.asynchronousAuthority();
  assert.equal(converted.runEpisode.constructor.name, 'AsyncFunction');
  assert.equal(typeof converted.createHunter, 'function');
  assert.equal(typeof converted.loadLevel, 'function');
});
