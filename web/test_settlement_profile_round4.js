'use strict';
const assert = require('assert');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard');
const meta = { opponent: 'bun.coop_hunter', difficulty: 'hard', seed: 7, mode: '1v2', map_id: 'training806', client_version: 'dev' };
function harness() {
  let now = Date.now(), registered = false, failResult = false, failProfile = false, contract = 2;
  let server = { nickname: '服务端昵称', victory_message: '服务端宣言' };
  const store = new Map(), writes = [], profiles = [], matches = new Map(), receipts = new Map();
  const make = () => LB.createClient({ config: { url: 'https://mock', publishableKey: 'public' }, crypto: webcrypto,
    now: () => now, storage: { getItem: k => store.get(k), setItem: (k, v) => store.set(k, v) }, fetch: async (url, options) => {
      const body = JSON.parse(options.body);
      const ok = result => ({ ok: true, json: async () => result });
      if (url.endsWith('qqt_get_profile')) return ok({ profile_contract_version: contract, registered, ...(registered ? server : {}) });
      if (url.endsWith('qqt_leaderboard')) return ok([{ rank: 1, ...server }]);
      if (url.endsWith('qqt_submit_result')) {
        const p = body.p_payload; writes.push(p);
        if (failResult) throw new TypeError('controlled');
        if (!registered) { registered = true; server = { nickname: p.nickname, victory_message: p.victory_message }; }
        if (matches.has(p.client_match_id)) assert.deepEqual(matches.get(p.client_match_id), p);
        matches.set(p.client_match_id, p);
        return ok({ level: 1, points: matches.size, games: matches.size, wins: 0, client_match_id: p.client_match_id, match_upgraded: false });
      }
      profiles.push(body);
      if (failProfile) throw new TypeError('controlled');
      const mid = body.p_client_match_id, match = matches.get(mid); assert(match); assert.equal(body.p_nickname, null);
      const superseded = [...receipts.keys()].some(id => matches.get(id).client_total_ms > match.client_total_ms);
      if (!receipts.has(mid)) {
        if (!superseded) server.victory_message = body.p_victory_message || server.victory_message;
        receipts.set(mid, body);
      } else assert.deepEqual(receipts.get(mid), body);
      return ok({ saved: true, profile_contract_version: contract, client_match_id: mid, ...server, superseded });
    } });
  return { make, store, writes, profiles, set registered(v) { registered = v; }, set failResult(v) { failResult = v; },
    set failProfile(v) { failProfile = v; }, set contract(v) { contract = v; }, get server() { return server; },
    async finish(c) { const m = c.begin(meta); now += 5000; await c.finish(m, { result: 'draw', gameDurationMs: 5000, autoSubmit: false }); return m; } };
}
(async () => {
  // A local name, including old Round 3 caches, cannot establish registration.
  const h = harness(); let c = h.make(); c.setProfile('缓存假昵称', '缓存假宣言');
  assert.equal(c.state().savedNickname, false); assert.equal(c.state().profileReady, false);
  h.registered = true; await c.syncProfile();
  assert.equal(c.state().nickname, '服务端昵称'); assert.equal(c.state().savedNickname, true);
  await h.finish(c); c.setDraft('篡改名字', '正常宣言');
  h.contract = 1; await c.syncProfile(); assert.equal(c.state().profileReady, false);
  assert.equal(await c.submitCard(), false); assert.equal(h.writes.length, 0, 'old backend contract blocks combined write and never claims success');
  h.contract = 2; await c.syncProfile(); h.failProfile = true; await c.submitCard();
  assert.equal(h.writes.length, 1); h.failProfile = false; h.contract = 1;
  await c.submitCard(); assert.equal(h.writes.length, 1); assert.equal(c.state().settlement.complete, undefined, 'unversioned profile response cannot complete card');
  h.contract = 2; await c.submitCard(); assert(c.state().settlement.complete);

  // A queued older registration wins even if a newer first-player card supplied another nickname.
  const old = harness(); c = old.make(); await c.syncProfile(); const first = await old.finish(c);
  c.setDraft('首局昵称', '首局宣言'); old.failResult = true; await c.submitCard();
  c.clearSettlement(); const second = await old.finish(c); c.setDraft('次局昵称', '次局宣言');
  old.failResult = false; await c.submitCard();
  assert.equal(old.server.nickname, '首局昵称'); assert.equal(old.server.victory_message, '次局宣言');
  assert.equal(old.writes.at(-1).client_match_id, second.client_match_id);
  const count = old.writes.length; c = old.make(); await c.syncProfile(); assert(c.openPending());
  assert.equal(c.state().settlement.client_match_id, first.client_match_id); assert(c.state().settlement.submitted);
  await c.submitCard(); assert.equal(old.writes.length, count, 'old result receipt only retries its profile');
  assert.equal(old.server.victory_message, '次局宣言'); assert.match(c.state().settlement.cardStatus, /较新一局/);

  // Queue-only storage from Round 2 recovers an editable card without implicit authorization.
  const legacy = JSON.parse(old.store.get('qqt.leaderboard.v1'));
  const { player_secret, ...payload } = old.writes.at(-1);
  legacy.queue = [{ ...payload, approved: false }]; delete legacy.cards; delete legacy.confirmed_profile;
  legacy.nextProfile = { nickname: '不应恢复改名', victory_message: '旧缓存待提交宣言' }; delete legacy.nextDeclaration;
  old.store.set('qqt.leaderboard.v1', JSON.stringify(legacy)); c = old.make();
  assert.equal(old.writes.length, count); assert(c.openPending()); assert(c.state().settlement.eligible);
  await c.syncProfile(); assert.equal(c.state().nickname, '首局昵称');
  assert.equal(c.state().savedNickname, true); assert.equal(old.writes.length, count);
  // The recovered already-sent payload remains unchanged once approved in persisted storage.
  legacy.queue[0].approved = true; old.store.set('qqt.leaderboard.v1', JSON.stringify(legacy)); c = old.make();
  await c.syncProfile(); await c.submitCard(); assert.deepEqual(old.writes.at(-1), old.writes[count - 1]);
  console.log('Round4 boundaries: server-confirmed identity, old RPC compatibility, earliest registration, old receipt-only retry and legacy queue migration passed');
})().catch(e => { console.error(e); process.exitCode = 1; });
