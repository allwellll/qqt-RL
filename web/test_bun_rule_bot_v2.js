#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { BunRuleTacticalBot, PHASE_NAMES } = require('./bun_rule_bot.js');

const fixtures = JSON.parse(fs.readFileSync(
  path.join(__dirname, '..', 'tests', 'fixtures', 'bun_rule_bot_v2_cases.json'), 'utf8'));

for (const testCase of fixtures.cases) {
  const bot = new BunRuleTacticalBot();
  bot.restorePhase(0, testCase.initial_phase);
  const state = Object.assign({ bun_bases: fixtures.bun_bases }, testCase.state);
  if (testCase.event) bot.observeTransition(testCase.event, state, 0);
  else bot.syncPhase(state, 0);
  const decision = bot.analyze(state, 0);
  assert.strictEqual(decision.phase, testCase.expected_phase, testCase.id);
  assert.deepStrictEqual(decision.action, testCase.expected_action, testCase.id);
}
assert.deepStrictEqual(PHASE_NAMES, [
  'COMBAT', 'KILL_CONFIRMED', 'OBJECTIVE_RUSH', 'CARRY_RETURN', 'DELIVER', 'RECOVER']);
console.log('Bun规则战术机器人 v2 FSM/目标路由跨语言 fixtures 通过');
