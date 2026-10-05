'use strict';
const assert = require('assert');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard.js');
const metadata = { opponent: 'bun.coop_hunter', difficulty: 'hard', seed: 7, mode: '2v2', map_id: 'training806', client_version: 'dev' };
function memory() { const map = new Map(); return { getItem: k => map.get(k) || null, setItem: (k,v) => map.set(k,v) }; }
(async () => {
  let now = Date.now(), unavailable = false;
  const calls = [], storage = memory();
  const make = (store = storage) => LB.createClient({ config: { url: 'https://db', publishableKey: 'public' },
    storage: store, crypto: webcrypto, now: () => now, fetch: async (url, options) => {
      calls.push({ url, options, payload: JSON.parse(options.body) });
      return unavailable ? { ok: false, status: 503, json: async () => ({}) } : { ok: true,
        json: async () => url.endsWith('qqt_leaderboard') ? [] : { level: 1, points: 3, wins: 1, games: 1 } };
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
  const bad = restored.begin(metadata); now += 5000;
  const before = calls.length; await restored.finish(bad, { result: 'win', gameDurationMs: 999 });
  assert.equal(calls.length, before);
  const broken = make({ getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } });
  assert.equal(broken.state().persistent, false);
  const target = { items: [], replaceChildren() { this.items = []; }, append(x) { this.items.push(x); } };
  const doc = { createElement() { return { textContent: '', append(...items) { this.items = items; } }; } };
  const row = { rank: 1, nickname: '<script>boom</script>', victory_message: '<img src=x onerror=alert(1)>', level: 2, progress: 1,
    level_reached_ms: 10000, last_level_up_ms: 5000, wins: 3, games: 4, win_rate: .75 };
  LB.renderRows(doc, target, [row]);
  assert.equal(target.items[0].items[0].textContent, '1. <script>boom</script> · Lv.2 (1/10)');
  assert.equal(target.items[0].items[2].textContent, row.victory_message); assert.match(target.items[0].items[1].textContent, /75.0%/);
  console.log('排行榜身份、校验、幂等、离线重试、payload与纯文本渲染回归通过');
})().catch(error => { console.error(error); process.exitCode = 1; });
