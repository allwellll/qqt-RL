'use strict';
const assert = require('assert');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard.js');
const metadata = { opponent: 'bun.coop_hunter', difficulty: 'hard', seed: 7, mode: '2v2', map_id: 'training806', client_version: 'dev' };
function memory() { const map = new Map(); return { getItem: k => map.get(k) || null, setItem: (k,v) => map.set(k,v) }; }
(async () => {
  let now = Date.now(), unavailable = false, submitStatus = 200, edgeCommittedThenFailed = false;
  const acceptedMatches = new Set();
  const calls = [], storage = memory();
  const make = (store = storage) => LB.createClient({ config: { url: 'https://db', publishableKey: 'public' },
    storage: store, crypto: webcrypto, now: () => now, fetch: async (url, options) => {
      calls.push({ url, options, payload: JSON.parse(options.body) });
      if (unavailable) return { ok: false, status: 503, json: async () => ({}) };
      if (url.includes('/functions/v1/') && edgeCommittedThenFailed) {
        acceptedMatches.add(JSON.parse(options.body).p_payload.client_match_id);
        return { ok: false, status: 503, json: async () => ({}) };
      }
      if (url.includes('/functions/v1/') && submitStatus !== 200) return { ok: false, status: submitStatus, json: async () => ({}) };
      if (url.endsWith('qqt_submit_result')) acceptedMatches.add(JSON.parse(options.body).p_payload.client_match_id);
      return { ok: true, json: async () => url.endsWith('qqt_leaderboard') ? [] :
        { level: 1, points: acceptedMatches.size * 3, wins: acceptedMatches.size, games: acceptedMatches.size } };
    } });
  const client = make(), id = client.state().player_id;
  assert.match(id, /^[0-9a-f-]{36}$/); assert.equal(make().state().player_id, id);
  assert.notEqual(make(memory()).state().player_id, id);
  for (const nick of ['', 'a'.repeat(25), '<img onerror=x>', 'a\x00b']) assert.throws(() => LB.profile(nick, ''), /昵称/);
  assert.throws(() => LB.profile('ok', 'x'.repeat(81)), /80/);
  assert.throws(() => LB.profile('ok', 'a\x7fb'), /控制/);
  assert.deepEqual(LB.profile('  中文😀  ', '  hello & "world"  '), { nickname: '中文😀', victory_message: 'hello & "world"' });
  assert.equal(LB.profile('😀'.repeat(24), '').nickname, '😀'.repeat(24));
  client.setProfile('安全玩家', '赢了！');
  const match = client.begin(metadata); now += 5000;
  await Promise.all([client.finish(match, { result: 'win', gameDurationMs: 5000 }), client.finish(match, { result: 'win', gameDurationMs: 5000 })]);
  let writes = calls.filter(c => c.url.endsWith('qqt_submit_result'));
  assert.equal(writes.length, 1); assert.equal(Object.keys(writes[0].payload.p_payload).length, 16);
  assert.deepEqual(Object.keys(writes[0].payload), ['p_payload']);
  assert.equal(writes[0].payload.p_payload.nickname, '安全玩家');
  assert.equal(writes[0].payload.p_payload.player_id, id);
  assert.equal(writes[0].payload.p_payload.client_total_ms, 5000); assert.equal(writes[0].options.headers.apikey, 'public');
  assert(!JSON.stringify(writes[0].payload).includes('replay')); assert(!Object.keys(writes[0].payload.p_payload).includes('ip'));
  unavailable = true;
  const next = client.begin(metadata); now += 5000;
  await client.finish(next, { result: 'loss', gameDurationMs: 5000 });
  assert.equal(client.state().pending, 1); assert.match(client.state().status, /可重试/);
  const restored = make(); assert.equal(restored.state().pending, 1);
  const failedPayload = calls.filter(c => c.url.endsWith('qqt_submit_result')).at(-1).payload;
  unavailable = false; await Promise.all([restored.retry(), restored.retry()]);
  writes = calls.filter(c => c.url.endsWith('qqt_submit_result'));
  assert.deepEqual(writes.at(-1).payload, failedPayload); assert.equal(restored.state().pending, 0);
  const gatewayStorage = memory();
  const gateway = LB.createClient({ config: { url: 'https://db', publishableKey: 'public', submitResultUrl: 'https://db/functions/v1/submit-result' },
    storage: gatewayStorage, crypto: webcrypto, now: () => now, fetch: async (url, options) => {
      calls.push({ url, options, payload: JSON.parse(options.body) });
      if (url.includes('/functions/v1/')) {
        const matchId = JSON.parse(options.body).p_payload.client_match_id;
        if (edgeCommittedThenFailed) { acceptedMatches.add(matchId); return { ok: false, status: 503, json: async () => ({}) }; }
        if (submitStatus !== 200) return { ok: false, status: submitStatus, json: async () => ({}) };
      }
      if (url.endsWith('qqt_submit_result')) acceptedMatches.add(JSON.parse(options.body).p_payload.client_match_id);
      return { ok: true, json: async () => url.endsWith('qqt_leaderboard') ? [] :
        { level: 1, points: acceptedMatches.size * 3, wins: acceptedMatches.size, games: acceptedMatches.size } };
    } });
  gateway.setProfile('网关玩家', '');
  submitStatus = 422;
  const rejected = gateway.begin(metadata); now += 5000;
  await gateway.finish(rejected, { result: 'win', gameDurationMs: 5000 });
  assert.equal(calls.filter(c => c.payload.p_payload && c.payload.p_payload.client_match_id === rejected.client_match_id).length, 1,
    '422 business rejection must not fall back to direct RPC');
  assert.equal(gateway.state().pending, 1);
  submitStatus = 200; edgeCommittedThenFailed = true;
  await gateway.retry();
  edgeCommittedThenFailed = false;
  assert.equal(gateway.state().pending, 0, '503 fallback clears an already committed match');
  assert.equal(acceptedMatches.has(rejected.client_match_id), true);
  assert.equal(gateway.state().progress.games, acceptedMatches.size,
    'same client_match_id is idempotent when Edge commits before fallback');
  const bad = restored.begin(metadata); now += 5000;
  const before = calls.length; await restored.finish(bad, { result: 'win', gameDurationMs: 999 });
  assert.equal(calls.length, before);
  const broken = make({ getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } });
  assert.equal(broken.state().persistent, false);
  const target = { items: [], replaceChildren() { this.items = []; }, append(x) { this.items.push(x); } };
  const doc = { createElement() { return { textContent: '', items: [], append(...items) { this.items.push(...items); } }; } };
  const row = { rank: 1, nickname: '<script>boom</script>', victory_message: '<img src=x onerror=alert(1)>', level: 2, progress: 1,
    level_reached_ms: 10000, last_level_up_ms: 5000, best_win_duration_ms: 4200, player_ip: '123.*.*.45', wins: 3, games: 4, win_rate: .75 };
  LB.renderRows(doc, target, [row]);
  assert.equal(target.items[0].items[0].textContent, '1');
  assert.equal(target.items[0].items.length, 5);
  assert.equal(target.items[0].items[4].textContent, row.player_ip);
  assert.equal(target.items[0].items[1].textContent, row.nickname);
  assert.equal(target.items[0].items[2].textContent, '4.2秒');
  assert.equal(target.items[0].items[3].textContent, row.victory_message);
  assert.equal(LB.maskIp('123.45.67.89'), '123.*.*.89');
  assert.equal(LB.maskIp('2001:db8:0:0:0:0:0:42'), '2001:*:*:42');
  assert.equal(LB.maskIp('not-an-ip'), '—');
  const noWin = { ...row, best_win_duration_ms: null };
  LB.renderRows(doc, target, [noWin]);
  assert.equal(target.items[0].items[2].textContent, '—');
  assert.equal(gateway.state().settlement.submitted, true);
  assert.equal(gateway.state().settlement.ranking, null, 'old deployed response never invents ranking');
  const elements = new Map(['settlement','settlement-title','settlement-time','settlement-rank','settlement-status'].map(id => [id, { textContent: '', hidden: true }]));
  const settlementDoc = { getElementById: id => elements.get(id) };
  for (const [outcome, title] of [['win','胜利'],['loss','失败'],['draw','平局']]) {
    LB.renderSettlement(settlementDoc, { status: '<script>offline</script>', settlement: { result: outcome, duration_ms: 5000, submitted: true, ranking: null } });
    assert.equal(elements.get('settlement-title').textContent, title);
    assert.equal(elements.get('settlement-rank').textContent, '排名待数据库升级');
    assert.equal(elements.get('settlement-status').textContent, '<script>offline</script>');
  }
  LB.renderSettlement(settlementDoc, { status: '', settlement: { result: 'win', duration_ms: 5000, ranking: { rank: 2, total: 4, percentile: 33.33 } } });
  assert.match(elements.get('settlement-rank').textContent, /2 名 \/ 4 位玩家.*33.33%/);
  client.clearSettlement(); assert.equal(client.state().settlement, null);
  const fakeMatch = { ...metadata, result: 'win', game_duration_ms: 5000 };
  const ranking = { ...metadata, result: 'win', duration_ms: 5000, rank: 2, total: 4, percentile: 33.33, comparison: 'player-best-v1' };
  assert(LB.validRanking(ranking, fakeMatch));
  assert(!LB.validRanking({ ...ranking, mode: '1v1' }, fakeMatch));
  assert(!LB.validRanking({ ...ranking, rank: '<img>' }, fakeMatch));
  const ambiguousAccepted = new Set(); let edgeBreak = true;
  const ambiguous = LB.createClient({ config: { url: 'https://db', publishableKey: 'public', submitResultUrl: 'https://db/functions/v1/submit-result' },
    storage: memory(), crypto: webcrypto, now: () => now, fetch: async (url, options) => {
      if (url.endsWith('qqt_leaderboard')) return { ok: true, json: async () => [] };
      const payload = JSON.parse(options.body).p_payload;
      ambiguousAccepted.add(payload.client_match_id);
      if (url.includes('/functions/') && edgeBreak) throw new TypeError('connection lost after commit');
      return { ok: true, json: async () => ({ level: 1, points: 3, wins: 1, games: ambiguousAccepted.size,
        match_rank: { ...ranking, ...metadata } }) };
    } });
  const lostConnection = ambiguous.begin(metadata); now += 5000;
  await ambiguous.finish(lostConnection, { result: 'win', gameDurationMs: 5000 });
  assert.equal(ambiguousAccepted.size, 1); assert.equal(ambiguous.state().pending, 0);
  assert.equal(ambiguous.state().settlement.ranking.rank, 2);
  edgeBreak = false;
  await ambiguous.finish(lostConnection, { result: 'win', gameDurationMs: 5000 });
  assert.equal(ambiguousAccepted.size, 1, 'duplicate finish cannot submit again');
  console.log('排行榜身份、校验、幂等、离线重试、payload与纯文本渲染回归通过');
})().catch(error => { console.error(error); process.exitCode = 1; });
