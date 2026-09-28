#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { BunRuleTacticalBot, TICK_HZ, HORIZON_STEPS } = require('./bun_rule_bot.js');

const fixtures = JSON.parse(fs.readFileSync(
  path.join(__dirname, '..', 'tests', 'fixtures', 'bun_rule_bot_cases.json'), 'utf8'));

assert.strictEqual(TICK_HZ, 10);
assert.strictEqual(HORIZON_STEPS, 40);
const bot = new BunRuleTacticalBot();
for (const testCase of fixtures.cases) {
  const decision = bot.analyze(testCase.state, 0);
  assert.deepStrictEqual(decision.action, testCase.expected_action, testCase.id);
  if (['own_safe_drop', 'bun_cage_chokepoint', 'safe_non_trade_kill'].includes(testCase.id)) {
    assert.strictEqual(decision.safeEscapeAfterBomb, true, testCase.id);
  }
  if (testCase.id === 'already_doomed') {
    assert.strictEqual(decision.doomed, true);
    assert.strictEqual(decision.claimedEscape, false);
  }
}

const chainCase = fixtures.cases.find((item) => item.id === 'chain_reaction_escape');
const chain = bot.analyze(chainCase.state, 0).predictedBombs.find((bomb) => bomb.row === 6 && bomb.col === 8);
assert.strictEqual(chain.explodeStep, 5);
assert.strictEqual(chain.physicalOwner, 0);
assert.strictEqual(chain.causalOwner, 1);

const perfCase = fixtures.cases.find((item) => item.id === 'safe_non_trade_kill');
for (let i = 0; i < 20; i++) bot.decide(perfCase.state, 0);
const times = [];
for (let i = 0; i < 300; i++) {
  const started = process.hrtime.bigint();
  bot.decide(perfCase.state, 0);
  times.push(Number(process.hrtime.bigint() - started) / 1e6);
}
times.sort((a, b) => a - b);
assert(times[times.length - 1] < 50, `max=${times[times.length - 1].toFixed(3)}ms`);
assert(times[Math.floor(times.length * 0.95)] < 5, `p95=${times[Math.floor(times.length * 0.95)].toFixed(3)}ms`);
console.log('Bun 规则战术机器人 JS fixtures/链爆/性能测试通过');
