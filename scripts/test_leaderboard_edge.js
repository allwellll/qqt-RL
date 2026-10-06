'use strict';
const assert = require('assert');
(async () => {
  const { createSettlementHandler } = await import('../supabase/functions/submit-result/handler.mjs');
  const calls = []; let fail = false, rejected = false;
  const progress = { level: 1, points: 3, wins: 1, games: 1 };
  const handler = createSettlementHandler({
    env: key => ({ SUPABASE_URL: 'https://db', SUPABASE_SERVICE_ROLE_KEY: 'server-only' })[key],
    createClient(url, key, options) {
      assert.equal(key, 'server-only'); assert.equal(options.auth.persistSession, false);
      if (fail) throw new Error('sensitive exception');
      return { async rpc(name, payload) { calls.push({ name, payload });
        return rejected ? { error: { code: '22023', message: 'sensitive detail' } } : { data: progress }; } };
    },
  });
  function request(origin = 'https://allwellll.github.io', extra = {}) {
    return new Request('https://edge', { method: 'POST', headers: { origin, ...extra }, body: JSON.stringify({ p_payload: { nickname: 'ok' } }) });
  }
  const good = await handler(request(undefined, { 'cf-connecting-ip': '::::', 'x-real-ip': '1.2.3.4', 'x-forwarded-for': '1::2::3' }));
  assert.equal(good.status, 200);
  assert.deepEqual(await good.json(), { ...progress, metadata_recorded: false, network_metadata_recorded: false });
  assert.deepEqual(calls, [{ name: 'qqt_submit_result', payload: { p_payload: { nickname: 'ok' } } }]);
  assert.equal(good.headers.get('vary'), 'Origin'); assert.equal(good.headers.get('access-control-allow-origin'), 'https://allwellll.github.io');
  const denied = await handler(request('https://evil.test')); assert.equal(denied.status, 403); assert.equal(calls.length, 1);
  assert.equal(denied.headers.get('access-control-allow-origin'), null);
  rejected = true; assert.equal((await handler(request())).status, 422); rejected = false;
  fail = true; const failure = await handler(request()); assert.equal(failure.status, 500);
  assert.equal(failure.headers.get('vary'), 'Origin'); assert(!JSON.stringify(await failure.json()).includes('sensitive'));
  console.log('Edge origin restriction, spoofed-header non-recording, progress and top-level exception tests passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
