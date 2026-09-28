#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const QQTBots = require('./bot_contract.js');
const { BunRuleTacticalBot } = require('./bun_rule_bot.js');

const fixture = JSON.parse(fs.readFileSync(
  path.join(__dirname, '..', 'tests', 'fixtures', 'bot_contract_cases.json'), 'utf8'));
const registry = QQTBots.createDefaultRegistry({ BunRuleTacticalBot });
const tactical = registry.describe('bun.tactical_v2', {});
assert.strictEqual(tactical.spec.identity_hash, fixture.tactical_identity_hash);
assert.strictEqual(tactical.fixture_hash, fixture.fixture_hash);
assert.strictEqual(tactical.spec.capabilities.frozen, true);
assert.strictEqual(tactical.spec.capabilities.batched, 'native_host');
assert.throws(() => registry.create('bun.tactical_v2', { unknown: 1 }), /unknown config/);
assert.throws(() => registry.validate('bun.tactical_v2', { horizon: '40' }), /invalid config type/);
assert.throws(() => registry.create('bun.tactical_v2', {}, { version: '3.0.0' }), /incompatible/);
assert.throws(() => registry.register(tactical.spec, () => ({})), /duplicate bot id/);

const bot = registry.create('bun.tactical_v2', {});
bot.reset({ schema: 'qqt.bot.context/v1', episode_id: 'fixture', seed: 7, ruleset: 'bun' });
const testCase = fixture.action_cases[0];
const action = bot.act({
  schema: 'qqt.bot.observation/v1', tick: 0, state: testCase.state,
  legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2],
}, testCase.player_id, 7);
QQTBots.validateAction(action, [0, 1, 2, 3, 4], [0, 1, 2]);
assert.deepStrictEqual(Object.keys(action).sort(), ['ability', 'move']);
assert.deepStrictEqual(action, testCase.expected_action);

const model = registry.describe('bun.browser_model', {});
assert.strictEqual(model.spec.capabilities.async, true);
assert.strictEqual(model.spec.runtime.includes('browser'), true);
console.log('Bot contract/registry cross-language fixtures passed');
