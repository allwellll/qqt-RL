'use strict';
const assert = require('assert');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { PGlite } = require('@electric-sql/pglite');
(async () => {
  const db = new PGlite();
  try {
    await db.exec('create role anon; create role authenticated;');
    await db.exec(fs.readFileSync('supabase/migrations/20261005140000_leaderboard.sql','utf8'));
    const pid = randomUUID(), other = randomUUID();
    const payload = (overrides = {}) => ({ player_id: pid, player_secret: 'a'.repeat(64), client_match_id: randomUUID(),
      nickname: '测试玩家', victory_message: '胜利！', result: 'win', game_duration_ms: 5000,
      wall_duration_ms: 5100, client_total_ms: 5000, opponent: 'bun.coop_hunter', difficulty: 'hard',
      seed: 17, mode: '2v2', map_id: 'training806', completed_at: new Date().toISOString(), client_version: 'dev', ...overrides });
    const submit = async p => (await db.query('select public.qqt_submit_result($1::jsonb) result', [JSON.stringify(p)])).rows[0].result;
    const asAnon = async callback => { await db.exec('set role anon'); try { return await callback(); } finally { await db.exec('reset role'); } };
    const first = payload(), result = await asAnon(() => submit(first));
    assert.equal(result.points, 3); assert.equal(result.games, 1);
    assert.deepEqual(await asAnon(() => submit(first)), result, 'duplicate is idempotent before rate check');
    await assert.rejects(asAnon(() => submit({ ...first, result: 'draw' })), /match id already used/);
    await assert.rejects(asAnon(() => submit(payload({ player_secret: 'b'.repeat(64) }))), /credential mismatch/);
    await assert.rejects(asAnon(() => submit(payload())), /rate limit/);
    for (const patch of [{ nickname: '' }, { nickname: '<script>x</script>' }, { victory_message: 'x'.repeat(81) },
      { game_duration_ms: 1 }, { result: 'other' }, { mode: 'watch' }, { seed: -1 }, { client_version: 'secret' },
      { completed_at: 'infinity' }, { completed_at: new Date(Date.now()-8*86400000).toISOString() },
      { ip: '1.2.3.4' }, { player_id: null }, { nickname: {} }, { game_duration_ms: '5000' }]) {
      await assert.rejects(asAnon(() => submit(payload(patch))), /invalid|range|constraint|syntax/);
    }
    for (const table of ['players','match_results','player_progress','level_events']) {
      await assert.rejects(asAnon(() => db.query(`select * from qqt_private.${table}`)), /permission denied/);
      await assert.rejects(asAnon(() => db.query(`delete from qqt_private.${table}`)), /permission denied/);
      await assert.rejects(asAnon(() => db.query(`update qqt_private.${table} set player_id=player_id`)), /permission denied/);
    }
    await assert.rejects(asAnon(() => db.query("insert into qqt_private.players values ($1,$2,'x','',clock_timestamp())",[randomUUID(),new Uint8Array(32)])), /permission denied/);
    const policy = await db.query("select relname,relrowsecurity from pg_class where relnamespace='qqt_private'::regnamespace and relkind='r'");
    assert(policy.rows.every(r => r.relrowsecurity)); assert.equal(policy.rows.length, 4);
    for (let i = 0; i < 3; i++) {
      await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[pid]);
      await asAnon(() => submit(payload({ client_total_ms: 10000+i*5000 })));
    }
    const ranks = (await asAnon(() => db.query('select * from public.qqt_leaderboard()'))).rows;
    assert.equal(ranks[0].level, 2); assert.equal(ranks[0].progress, 2);
    assert.equal(ranks[0].level_reached_ms, 20000); assert.equal(ranks[0].last_level_up_ms, 20000);
    const events = (await db.query('select * from qqt_private.level_events')).rows;
    assert.equal(events.length, 1); assert.equal(events[0].level, 2);
    await asAnon(() => submit(payload({ player_id: other, nickname: '另一玩家' })));
    const sorted = (await asAnon(() => db.query('select * from public.qqt_leaderboard()'))).rows;
    assert.equal(sorted[0].nickname, '测试玩家'); assert.equal(sorted[1].nickname, '另一玩家');
    assert(!Object.keys(sorted[0]).some(x => /player_id|hash|ip/.test(x)));
    // Equal level: shorter level-reached time wins, then wins, then win rate.
    await db.query('update qqt_private.player_progress set points=12,level=2,total_game_ms=20000,level_reached_ms=15000 where player_id=$1',[other]);
    assert.equal((await asAnon(() => db.query('select * from public.qqt_leaderboard()'))).rows[0].nickname,'另一玩家');
    await db.query('update qqt_private.player_progress set level_reached_ms=20000 where player_id=$1',[other]);
    assert.equal((await asAnon(() => db.query('select * from public.qqt_leaderboard()'))).rows[0].nickname,'测试玩家');
    await db.query('update qqt_private.player_progress set wins=4,games=4 where player_id=$1',[other]);
    await db.query('update qqt_private.player_progress set games=5 where player_id=$1',[pid]);
    assert.equal((await asAnon(() => db.query('select * from public.qqt_leaderboard()'))).rows[0].nickname,'另一玩家');
    console.log('本地PGlite SQL迁移、权限/RLS、校验、幂等、限流、升级与排序通过；不是Supabase远端E2E');
  } finally { await db.close(); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
