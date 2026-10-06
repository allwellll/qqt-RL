'use strict';
const assert = require('assert');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { PGlite } = require('@electric-sql/pglite');
(async () => {
  const db = new PGlite();
  try {
    await db.exec('create role anon; create role authenticated; create role service_role;');
    await db.exec(fs.readFileSync('supabase/migrations/20261005140000_leaderboard.sql', 'utf8'));
    const payload = (result = 'win', ms = 5000, patch = {}) => ({ player_id: randomUUID(), player_secret: 'a'.repeat(64), client_match_id: randomUUID(),
      nickname: '排名玩家', victory_message: '', result, game_duration_ms: ms, wall_duration_ms: ms,
      client_total_ms: ms, opponent: 'bun.coop_hunter', difficulty: 'hard', mode: '2v2', map_id: 'training806',
      seed: 7, client_version: 'dev', completed_at: new Date().toISOString(), ...patch });
    const submit = async p => {
      await db.exec('set role anon');
      try { return (await db.query('select public.qqt_submit_result($1::jsonb) result', [JSON.stringify(p)])).rows[0].result; }
      finally { await db.exec('reset role'); }
    };
    const historical = payload(); await submit(historical);
    // Ranking migration works directly on the old schema, with real historical records.
    const migration = fs.readFileSync('supabase/migrations/20261006090000_match_ranking.sql', 'utf8');
    await db.exec(migration); await db.exec(migration);
    const current = payload('win', 6000), response = await submit(current);
    assert.equal(response.match_rank.rank, 2); assert.equal(response.match_rank.total, 2); assert.equal(response.match_rank.percentile, 0);
    assert.deepEqual(await submit(current), response, 'retry is idempotent without new samples');
    const fast = payload('win', 4000), fastest = await submit(fast);
    assert.equal(fastest.match_rank.rank, 1); assert.equal(fastest.match_rank.percentile, 100);
    const tie = await submit(payload('win', 5000));
    assert.equal(tie.match_rank.rank, 2); assert.equal(tie.match_rank.total, 4); assert.equal(tie.match_rank.percentile, 33.33);
    const isolated = await submit(payload('win', 3000, { mode: '1v1' }));
    assert.equal(isolated.match_rank.total, 1); assert.equal(isolated.match_rank.percentile, 0);
    for (const patch of [{ map_id: 'arena' }, { difficulty: 'easy' }, { opponent: 'bun.hunter' }]) {
      assert.equal((await submit(payload('win', 3000, patch))).match_rank.total, 1, 'map/difficulty/opponent isolate cohorts');
    }
    for (const result of ['loss', 'draw']) {
      await submit(payload(result, 4000));
      const survived = await submit(payload(result, 8000));
      assert.equal(survived.match_rank.rank, 1); assert.equal(survived.match_rank.percentile, 100); assert.equal(survived.match_rank.order, 'desc');
    }
    await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1", [fast.player_id]);
    const slowerOwn = await submit(payload('win', 7000, { player_id: fast.player_id }));
    assert.equal(slowerOwn.match_rank.rank, 4); assert.equal(slowerOwn.match_rank.total, 4, 'one sample per player, own current match not historical best');
    await db.exec('set role anon');
    await assert.rejects(() => db.query('select qqt_private.qqt_submit_result_core($1::jsonb)', [JSON.stringify(current)]), /permission denied/);
    await db.exec('reset role');
    await assert.rejects(() => submit({ ...current, player_secret: 'b'.repeat(64) }), /credential mismatch/);
    console.log('Incremental old-schema rank RPC: cohorts, outcomes, ties, percentile, peer deduplication, retry and private-core ACL passed');
  } finally { await db.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
