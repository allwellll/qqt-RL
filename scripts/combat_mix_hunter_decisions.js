'use strict';
const fs = require('fs');
const {BunHunterBot, DIFFICULTIES} = require('../web/bun_hunter_bot.js');
const [input, output] = process.argv.slice(2);
const fixture = JSON.parse(fs.readFileSync(input));
const rows = [];
for (const row of fixture.states) {
  for (const difficulty of ['easy', 'normal', 'hard']) {
    const bot = new BunHunterBot({difficulty, seed: row.seed});
    const decision = bot.analyze(JSON.parse(JSON.stringify(row.hunter_state)), 1);
    rows.push({id: row.id, seed: row.seed, difficulty, ...decision});
  }
}
fs.writeFileSync(output, JSON.stringify({passed: true, source: 'production web/bun_hunter_bot.js.analyze',
  scope: 'same serialized observable state; independent documented PRNGs', parameters: DIFFICULTIES, rows}, null, 2));
