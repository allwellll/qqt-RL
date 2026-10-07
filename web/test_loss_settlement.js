'use strict';
const assert = require('assert');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard');
const meta = { opponent: 'bun.coop_hunter', difficulty: 'hard', seed: 7, mode: '1v2', map_id: 'training806', client_version: 'dev' };
const text = '小伙子，再沉淀沉淀吧';
(async () => {
  for (const registered of [false, true]) {
    let now = Date.now(); const store = new Map(), writes = [], profiles = [];
    const make = () => LB.createClient({ config: { url: 'https://mock', publishableKey: 'public' }, crypto: webcrypto,
      now: () => now, storage: { getItem: k => store.get(k), setItem: (k, v) => store.set(k, v) },
      fetch: async (url, options) => {
        const p = JSON.parse(options.body), ok = result => ({ ok: true, json: async () => result });
        if (url.endsWith('qqt_get_profile')) return ok({ profile_contract_version: 2, registered,
          ...(registered ? { nickname: '已确认昵称', victory_message: '旧宣言' } : {}) });
        if (url.endsWith('qqt_leaderboard')) return ok([]);
        if (url.endsWith('qqt_submit_result')) { writes.push(p.p_payload); return ok({ level: 1, points: 1,
          wins: 0, games: 1, client_match_id: p.p_payload.client_match_id, match_upgraded: false }); }
        profiles.push(p); return ok({ saved: true, profile_contract_version: 2, client_match_id: p.p_client_match_id,
          nickname: '已确认昵称', victory_message: p.p_victory_message });
      } });
    let c = make(); await c.syncProfile(); const match = c.begin(meta); now += 5000;
    await c.finish(match, { result: 'loss', gameDurationMs: 5000, autoSubmit: false });
    assert.equal(LB.settlementText(c.state().settlement).percentile, '', 'loss must not advertise an unavailable submission');
    const elements = new Map(['settlement', 'settlement-title', 'settlement-time', 'settlement-rank', 'settlement-status',
      'settlement-form', 'settlement-submit', 'settlement-close', 'nickname-field', 'player-nickname', 'player-message', 'current-nickname']
      .map(id => [id, { hidden: false, disabled: false, required: true, textContent: '' }]));
    const render = () => LB.renderSettlement({ getElementById: id => elements.get(id) }, c.state());
    render();
    assert.equal(elements.get('settlement-form').hidden, true, 'loss must hide the entire profile form');
    assert.equal(elements.get('settlement-status').textContent, text);
    for (const id of ['player-nickname', 'player-message', 'settlement-submit'])
      assert.equal(elements.get(id).disabled, true, `loss must disable ${id}`);
    assert.equal(elements.get('current-nickname').hidden, true);
    assert.equal(elements.get('player-nickname').required, false);
    c.setDraft('不应提交昵称', '不应提交宣言');
    assert.equal(await c.submitCard(), false, 'programmatic form submission must not authorize a loss');
    assert.equal(writes.length, 0, 'hiding the form must not introduce automatic result writes');
    assert.equal(profiles.length, 0);
    // Existing authorized result-only flows remain idempotent, but cannot write profiles.
    await c.submitSettlement(); assert.equal(writes.length, 1);
    await assert.rejects(() => c.submitProfile('不应提交昵称', '不应提交宣言'), /失败局/);
    assert.equal(c.profileActive(), false);
    c = make(); await c.syncProfile();
    assert.equal(c.state().settlement.result, 'loss'); render();
    assert.equal(await c.submitCard(), false);
    await assert.rejects(() => c.submitProfile('不应提交昵称', '不应提交宣言'), /失败局/);
    await c.submitSettlement(); assert.equal(writes.length, 1, 'reload retains result receipt');
    assert.equal(profiles.length, 0, 'loss never emits qqt_update_profile, including restored cards');
    assert.equal(elements.get('settlement-status').textContent, text);
    // The same DOM is reset for win/draw; their profile controls stay available.
    for (const result of ['win', 'draw']) {
      const state = { ...c.state(), submitting: false, settlement: { client_match_id: match.client_match_id, result, eligible: true, duration_ms: 5000 } };
      LB.renderSettlement({ getElementById: id => elements.get(id) }, state);
      assert.equal(elements.get('settlement-form').hidden, false);
      assert.equal(elements.get('player-message').disabled, false);
      assert.equal(elements.get('settlement-submit').disabled, false);
      assert.equal(elements.get('nickname-field').hidden, registered);
    }
  }
  console.log('Loss settlement: exact message, hidden/disabled profile, zero profile RPCs, reload/result idempotency and win/draw DOM reset passed');
})().catch(e => { console.error(e); process.exitCode = 1; });
