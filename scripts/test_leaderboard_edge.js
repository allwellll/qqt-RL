'use strict';
const assert = require('assert');
(async () => {
  const { createSettlementHandler, parseIp, requestIp } = await import('../supabase/functions/submit-result/handler.mjs');
  const calls = []; let fail = false, rejected = false, ipFailed = false;
  const progress = { level: 1, points: 3, wins: 1, games: 1 };
  const handler = createSettlementHandler({
    env: key => ({ SUPABASE_URL: 'https://db', SUPABASE_SERVICE_ROLE_KEY: 'server-only' })[key],
    createClient(url, key, options) {
      assert.equal(key, 'server-only'); assert.equal(options.auth.persistSession, false);
      if (fail) throw new Error('sensitive exception');
      return { async rpc(name, payload) { calls.push({ name, payload });
        if (name === 'qqt_record_player_ip' || name === 'qqt_record_match_ip') return ipFailed ? { error: { code: 'PGRST202' } } : { data: { recorded: true, ip_display: '1.*.*.4' } };
        return rejected ? { error: { code: '22023', message: 'sensitive detail' } } : { data: progress }; } };
    },
  });
  function request(origin = 'https://allwellll.github.io', extra = {}) {
    return new Request('https://edge', { method: 'POST', headers: { origin, ...extra }, body: JSON.stringify({ p_payload: { player_id: 'test-id', client_match_id: 'test-match', nickname: 'ok' } }) });
  }
  const good = await handler(request(undefined, { 'cf-connecting-ip': '::::', 'x-real-ip': '1.2.3.4', 'x-forwarded-for': '1::2::3' }));
  assert.equal(good.status, 200);
  assert.deepEqual(await good.json(), { ...progress, metadata_recorded: true, network_metadata_recorded: true, ip_display: '1.*.*.4' });
  assert.deepEqual(calls, [{ name: 'qqt_submit_result', payload: { p_payload: { player_id: 'test-id', client_match_id: 'test-match', nickname: 'ok' } } },
    { name: 'qqt_record_match_ip', payload: { p_player_id: 'test-id', p_client_match_id: 'test-match', p_raw_ip: '1.2.3.4' } }]);
  assert.equal(good.headers.get('vary'), 'Origin'); assert.equal(good.headers.get('access-control-allow-origin'), 'https://allwellll.github.io');
  const denied = await handler(request('https://evil.test')); assert.equal(denied.status, 403); assert.equal(calls.length, 2);
  assert.equal(denied.headers.get('access-control-allow-origin'), null);
  for (const ip of ['1.2.3.999','01.2.3.4','1.2.3.4/24','::::','1::2::3','1.2.3.4:80','[::1]','fe80::1%eth0']) assert.equal(parseIp(ip),null);
  for (const ip of ['1.2.3.4','2001:db8::42','::1','::ffff:192.0.2.1']) assert(parseIp(ip));
  assert.equal(requestIp(new Headers({'x-forwarded-for':'123.45.67.89, 10.0.0.1'})), '123.45.67.89', 'spoofed but valid source is accepted');
  const none = await (await handler(request())).json(); assert.equal(none.network_metadata_recorded,false);
  ipFailed = true;
  const ipFailure = await handler(request(undefined, {'x-forwarded-for':'1.2.3.4'}));
  assert.equal(ipFailure.status,200); assert.equal((await ipFailure.json()).network_metadata_recorded,false); ipFailed = false;
  rejected = true; assert.equal((await handler(request())).status, 422); rejected = false;
  fail = true; const failure = await handler(request()); assert.equal(failure.status, 500);
  assert.equal(failure.headers.get('vary'), 'Origin'); assert(!JSON.stringify(await failure.json()).includes('sensitive'));
  console.log('Edge origin restriction, accepted source IP, server masking, invalid IP and metadata failure, progress and top-level exception tests passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
