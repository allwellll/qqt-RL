#!/usr/bin/env node
'use strict';
// Cross-language safe-teacher regression. Replays the frozen self-play combat
// states in tests/fixtures/bun_rule_bot_selfplay_parity_cases.json through the
// JS rule bot and asserts byte-identical decisions to the Python reference,
// locking in Python/JS parity of the survival planner (escape-under-fire,
// doomed-max-survival, safe pressure attacks, space control).
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { BunRuleTacticalBot } = require('./bun_rule_bot.js');

const fixture = JSON.parse(fs.readFileSync(
  path.join(__dirname, '..', 'tests', 'fixtures',
    'bun_rule_bot_selfplay_parity_cases.json'), 'utf8'));

assert(fixture.cases.length > 0, 'fixture must not be empty');
for (const testCase of fixture.cases) {
  const bot = new BunRuleTacticalBot();
  const state = Object.assign({ bun_bases: fixture.bun_bases }, testCase.state);
  const decision = bot.analyze(state, testCase.player_id);
  const action = Array.from(decision.action).map((x) => Number(x));
  assert.deepStrictEqual(action, testCase.expected_action, testCase.id);
  assert.strictEqual(decision.reason, testCase.expected_reason, testCase.id);
  assert.strictEqual(decision.phase, testCase.expected_phase, testCase.id);
}
console.log(`Bun规则战术机器人自博弈战斗态跨语言 parity 通过 (${fixture.cases.length} 例)`);
