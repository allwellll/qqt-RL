'use strict';
const assert = require('assert');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard');
const ids = ['settlement', 'settlement-title', 'settlement-time', 'settlement-rank', 'settlement-status',
  'settlement-form', 'settlement-submit', 'settlement-close', 'nickname-field', 'player-nickname', 'player-message', 'current-nickname'];
const elements = new Map(ids.map(id => [id, { hidden: false, disabled: false, textContent: '' }]));
const doc = { getElementById: id => elements.get(id) };
const loss = { client_match_id: webcrypto.randomUUID(), result: 'loss', duration_ms: 5000, eligible: true };
const base = { settlement: loss, profileReady: true, savedNickname: true, nickname: '玩家' };
function empty(state) {
  LB.renderSettlement(doc, state);
  assert.equal(elements.get('settlement').hidden, true, 'inactive settlement must be hidden');
  for (const id of ['settlement-title', 'settlement-time', 'settlement-rank', 'settlement-status', 'current-nickname'])
    assert.equal(elements.get(id).textContent, '', `${id} must be emptied, not just covered by CSS`);
  assert(elements.get('settlement-form').hidden);
  for (const id of ['player-nickname', 'player-message', 'settlement-submit']) assert(elements.get(id).disabled);
}
const which = process.env.CASE;
if (!which || which === 'loading') {
  empty({ ...base, settlementReady: false });
  // First ready frame restores the draft; loading a different map removes it again.
  LB.renderSettlement(doc, { ...base, settlementReady: true });
  assert.equal(elements.get('settlement-status').textContent, '小伙子，再沉淀沉淀吧');
  empty({ ...base, settlementReady: false });
}
if (!which || which === 'clear') {
  LB.renderSettlement(doc, base);
  empty({ ...base, settlement: null });
  LB.renderSettlement(doc, base);
  empty({ ...base, settlement: { ...loss, closed: true } });
}
if (!which || which === 'invalid') {
  for (const value of [{}, { result: 'loss' }, { ...loss, result: 'unknown' },
    { ...loss, client_match_id: 'bad' }, { ...loss, duration_ms: NaN }, { ...loss, duration_ms: -1 }])
    empty({ ...base, settlement: value });
}
if (!which) {
  for (const result of ['win', 'draw']) {
    LB.renderSettlement(doc, { ...base, settlement: { ...loss, result } });
    assert(!elements.get('settlement').hidden);
    assert(!elements.get('settlement-form').hidden);
    assert(!elements.get('player-message').disabled);
    assert(!elements.get('settlement-status').textContent.includes('小伙子'));
  }
  let now = 10000, store = new Map();
  const options = { config: {}, storage: { getItem: k => store.get(k), setItem: (k, v) => store.set(k, v) },
    crypto: webcrypto, now: () => now, fetch: async () => { throw new Error('Unexpected network request'); } };
  const c = LB.createClient(options), match = c.begin({}); now += 5000;
  c.finish(match, { result: 'loss', gameDurationMs: 5000, autoSubmit: false });
  let changes = [];
  const recovered = LB.createClient({ ...options, settlementReady: false, onChange: state => changes.push(state) });
  empty(recovered.state());
  recovered.setSettlementReady(true);
  assert.equal(changes.at(-1).settlementReady, true);
  LB.renderSettlement(doc, recovered.state());
  assert.equal(elements.get('settlement-status').textContent, '小伙子，再沉淀沉淀吧');
  recovered.clearSettlement();
  assert.equal(changes.at(-1).settlement, null, 'clear must emit immediately, including replay cleanup');
  empty(recovered.state());
}
console.log('Settlement lifecycle: loading, cleanup, closed/invalid, restored loss and win/draw passed');
