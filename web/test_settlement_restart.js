'use strict';
const assert = require('assert');
const fs = require('fs');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard');
const meta = { opponent: 'bun.coop_hunter', difficulty: 'hard', seed: 7, mode: '1v2', map_id: 'training806', client_version: 'dev' };
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function setup() {
  const elements = new Map(), store = new Map(), writes = [], profiles = [];
  let now = Date.now(), registered = false, failResult = false, failProfile = false, restarts = 0, holdProfile;
  const el = id => {
    if (!elements.has(id)) elements.set(id, { value: '', textContent: '', hidden: false, listeners: {},
      addEventListener(type, fn) { this.listeners[type] = fn; }, replaceChildren() {}, append() {} });
    return elements.get(id);
  };
  let client;
  client = LB.mount({ getElementById: el, createElement: () => ({ append() {} }) }, {
    config: { url: 'https://mock', publishableKey: 'public' }, crypto: webcrypto, now: () => now,
    storage: { getItem: k => store.get(k), setItem: (k, v) => store.set(k, v) },
    async onRestart() { restarts++; client.clearSettlement(); },
    fetch: async (url, options) => {
      const p = JSON.parse(options.body), ok = x => ({ ok: true, json: async () => x });
      if (url.endsWith('qqt_leaderboard')) return ok([]);
      if (url.endsWith('qqt_get_profile')) return ok({ profile_contract_version: 2, registered,
        ...(registered ? { nickname: '首次昵称', victory_message: '已保存感言' } : {}) });
      if (url.endsWith('qqt_submit_result')) {
        writes.push(p.p_payload); if (failResult) throw new TypeError('controlled result failure'); registered = true;
        return ok({ level: 1, points: 1, wins: 0, games: 1, client_match_id: p.p_payload.client_match_id });
      }
      profiles.push(p); if (failProfile) throw new TypeError('controlled profile failure');
      if (holdProfile) await holdProfile;
      return ok({ saved: true, profile_contract_version: 2, client_match_id: p.p_client_match_id,
        nickname: '首次昵称', victory_message: p.p_victory_message });
    }
  });
  return { client, el, writes, profiles, get restarts() { return restarts; }, set failResult(v) { failResult = v; },
    set failProfile(v) { failProfile = v; }, set holdProfile(v) { holdProfile = v; },
    async finish(result) {
      await client.syncProfile(); const m = client.begin(meta); now += 5000;
      await client.finish(m, { result, gameDurationMs: 5000, autoSubmit: false });
      client.setDraft('首次昵称', '已保存感言');
    },
    submit() { return el('settlement-form').listeners.submit({ preventDefault() {} }); },
    close() { return el('settlement-close').listeners.click({}); }
  };
}
(async () => {
  const mode = process.argv[2] || 'all';
  if (mode === 'copy' || mode === 'all') {
    const html = fs.readFileSync('web/index.html', 'utf8');
    assert.match(html, /<label>胜利感言<input id="player-message"/);
    assert.match(html, /class="message">胜利感言<\/th>/);
    for (const file of ['web/index.html', 'web/leaderboard.js'])
      assert(!fs.readFileSync(file, 'utf8').includes('宣言'), `${file} must use the new player-facing wording`);
  }
  if (mode === 'submit' || mode === 'all') {
    for (const result of ['win', 'draw']) {
      const h = setup(); await h.finish(result);
      h.failResult = true; await h.submit(); await tick();
      assert.equal(h.restarts, 0, 'result failure must not restart');
      h.failResult = false; h.failProfile = true; await h.submit(); await tick();
      assert.equal(h.restarts, 0, 'profile failure must not restart');
      assert.equal(h.client.state().settlement.result, result);
      const count = h.writes.length;
      h.failProfile = false; let release; h.holdProfile = new Promise(resolve => { release = resolve; });
      const first = h.submit(), duplicate = h.submit(); await tick();
      assert.equal(h.restarts, 0, 'pending success must not restart early');
      const closeWhilePending = h.close(); assert.equal(h.restarts, 0);
      release(); await Promise.all([first, duplicate, closeWhilePending]); await tick();
      assert.equal(h.restarts, 1, 'profile success must restart exactly once');
      assert.equal(h.writes.length, count, 'profile retry must not repeat the result');
      assert.equal(h.client.state().settlement, null);
      await h.submit(); await h.close(); await tick(); assert.equal(h.restarts, 1, 'stale actions cannot restart the replacement match');
    }
  }
  if (mode === 'close' || mode === 'all') {
    for (const result of ['win', 'draw', 'loss']) {
      const h = setup(); await h.finish(result);
      await Promise.all([h.close(), h.close(), h.close()]); await tick();
      assert.equal(h.restarts, 1, 'X must restart exactly once');
      assert.equal(h.writes.length, 0); assert.equal(h.profiles.length, 0);
      assert.equal(h.client.state().settlement, null);
    }
  }
  console.log(`Settlement ${mode}: wording, success-only restart, X restart, failure/pending protection and duplicate/stale actions passed`);
})().catch(e => { console.error(e); process.exitCode = 1; });
