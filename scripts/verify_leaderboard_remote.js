'use strict';

// Live smoke test for the already-applied migration. Secrets stay in memory and
// are never included in stdout or the evidence file.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const vm = require('vm');
const { execFileSync } = require('child_process');

const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync('web/leaderboard_config.js', 'utf8'), sandbox);
const config = sandbox.window.QQTLeaderboardConfig;
const base = `${config.url}/rest/v1`;
const playerId = crypto.randomUUID();
const matchId = crypto.randomUUID();
const secret = crypto.randomBytes(32).toString('hex');
const payload = {
  player_id: playerId,
  client_match_id: matchId,
  player_secret: secret,
  nickname: 'Hermes远端验收',
  victory_message: '临时测试，验收后清理',
  result: 'win',
  game_duration_ms: 5000,
  wall_duration_ms: 5000,
  client_total_ms: 5000,
  opponent: 'bun.coop_hunter',
  difficulty: 'fixed',
  seed: 20261005,
  mode: '1v1',
  map_id: 'training806',
  completed_at: new Date().toISOString(),
  client_version: 'dev',
};

async function request(endpoint, body) {
  const text = execFileSync('curl', [
    '--silent', '--show-error', '--retry', '2', '--max-time', '30',
    '--request', 'POST', '--header', `apikey: ${config.publishableKey}`,
    '--header', 'Content-Type: application/json', '--data-binary', JSON.stringify(body),
    '--write-out', '\n%{http_code}', `${base}/${endpoint}`,
  ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  const split = text.lastIndexOf('\n');
  const status = Number(text.slice(split + 1));
  const bodyText = text.slice(0, split);
  let value;
  try { value = JSON.parse(bodyText); } catch (_) { value = { raw: bodyText.slice(0, 200) }; }
  return { status, value };
}

async function probe(endpoint) {
  const text = execFileSync('curl', [
    '--silent', '--show-error', '--retry', '2', '--max-time', '30',
    '--header', `apikey: ${config.publishableKey}`, '--write-out', '\n%{http_code}',
    `${base}/${endpoint}`,
  ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  const split = text.lastIndexOf('\n');
  const status = Number(text.slice(split + 1));
  const bodyText = text.slice(0, split);
  let value;
  try { value = JSON.parse(bodyText); } catch (_) { value = { raw: bodyText.slice(0, 200) }; }
  return { status, code: value && value.code || null };
}

(async () => {
  const before = await request('rpc/qqt_leaderboard', {});
  if (before.status !== 200 || !Array.isArray(before.value)) throw new Error('leaderboard RPC did not return 200 rows');
  const first = await request('rpc/qqt_submit_result', { p_payload: payload });
  if (first.status !== 200 || !first.value || first.value.games !== 1 || first.value.wins !== 1) throw new Error(`valid settlement failed (${first.status})`);
  const repeat = await request('rpc/qqt_submit_result', { p_payload: payload });
  if (repeat.status !== 200 || JSON.stringify(repeat.value) !== JSON.stringify(first.value)) throw new Error('identical settlement was not idempotent');
  const wrong = await request('rpc/qqt_submit_result', { p_payload: { ...payload, player_secret: '0'.repeat(64) } });
  if (wrong.status !== 401 || wrong.value.code !== '42501') throw new Error('wrong secret was not rejected');
  const board = await request('rpc/qqt_leaderboard', {});
  const testRow = board.value.find(row => row.nickname === payload.nickname && row.victory_message === payload.victory_message);
  if (board.status !== 200 || !testRow || testRow.games < 1 || testRow.wins < 1) throw new Error('submitted player is absent from leaderboard');
  const baseTables = {
    players: await probe('players?select=player_id&limit=1'),
    match_results: await probe('match_results?select=client_match_id&limit=1'),
  };
  if (Object.values(baseTables).some(result => result.status < 400)) throw new Error('base table is directly readable');
  const evidence = {
    initialized: true,
    independent: true,
    ddlExecutedByThisTask: false,
    player_id: playerId,
    client_match_id: matchId,
    firstStatus: first.status,
    first: first.value,
    repeatStatus: repeat.status,
    repeat: repeat.value,
    wrongSecretStatus: wrong.status,
    wrongSecretCode: wrong.value.code || null,
    boardStatus: board.status,
    testRow,
    baseTables,
    passed: true,
    cleanupSql: `delete from qqt_private.players where player_id = '${playerId}'::uuid;`,
  };
  const out = path.resolve('runs/leaderboard_20261005/remote-live-e2e-independent.json');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify(evidence, null, 2));
})().catch(error => { console.error(error.message); process.exitCode = 1; });
