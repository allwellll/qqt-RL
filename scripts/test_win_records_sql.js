'use strict';
const assert = require('assert');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { PGlite } = require('@electric-sql/pglite');
const migration = 'supabase/migrations/20261008090000_win_records.sql';
(async () => {
  const db = new PGlite();
  try {
    await db.exec('create role anon; create role authenticated; create role service_role;');
    for (const file of ['20261005140000_leaderboard.sql','20261005173000_leaderboard_ip_and_best_win.sql'])
      await db.exec(fs.readFileSync(`supabase/migrations/${file}`, 'utf8'));
    await db.exec(fs.readFileSync('supabase/manual_upgrade_20261006.sql', 'utf8'));
    await db.exec(fs.readFileSync('supabase/manual_profile_update_20261007.sql', 'utf8'));
    const asRole = async (role, fn) => { await db.exec(`set role ${role}`); try { return await fn(); } finally { await db.exec('reset role'); } };
    const call = (name,args=[]) => asRole('anon',async () => (await db.query(`select public.${name}(${args.map((_,i)=>`$${i+1}`).join(',')}) result`,args)).rows[0].result);
    const payload = (extra={}) => ({player_id:randomUUID(),player_secret:'a'.repeat(64),client_match_id:randomUUID(),nickname:'同名玩家',
      victory_message:'赛果感言',result:'win',game_duration_ms:5000,wall_duration_ms:5000,client_total_ms:5000,
      opponent:'bun.coop_hunter',difficulty:'hard',seed:7,mode:'1v2',map_id:'training806',completed_at:new Date().toISOString(),client_version:'dev',...extra});
    const submit = p => call('qqt_submit_result',[JSON.stringify(p)]);
    const profile = (p,message,mid=p.client_match_id,secret=p.player_secret,nickname=null) =>
      call('qqt_update_profile',[p.player_id,secret,mid,nickname,message]);
    const allowNext = p => db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[p.player_id]);
    const legacy=payload({nickname:'历史玩家',game_duration_ms:9000,wall_duration_ms:9000,client_total_ms:9000});
    await submit(legacy);await profile(legacy,'不可当成逐局历史');
    await asRole('service_role',()=>db.query('select public.qqt_record_player_ip($1,$2)',[legacy.player_id,'12.34.56.78']));
    if (fs.existsSync(migration)) await db.exec(fs.readFileSync(migration,'utf8'));
    // RED on the deployed baseline: no per-win contract exists.
    const before=await call('qqt_win_leaderboard');assert.equal(before.leaderboard_contract_version,1);
    assert.equal(before.rows.length,1);assert.equal(before.rows[0].victory_message,null);assert.equal(before.rows[0].player_ip,null);
    const first=payload();await submit(first);await profile(first,'第一局');
    await allowNext(first);const second={...first,client_match_id:randomUUID(),game_duration_ms:4000,client_total_ms:9000};
    await submit(second);await profile(second,'第二局');
    await allowNext(first);const third={...first,client_match_id:randomUUID(),game_duration_ms:3000,client_total_ms:12000,victory_message:''};
    await submit(third);await profile(third,'');
    let mine=await call('qqt_my_win_leaderboard',[first.player_id,first.player_secret]);
    assert.deepEqual(mine.rows.filter(r=>r.is_mine).map(r=>r.victory_message),['第二局','第二局','第一局']);
    assert.equal(mine.rows.filter(r=>r.is_latest).length,1);assert.equal(mine.rows.find(r=>r.is_latest).game_duration_ms,3000);
    const ids=mine.rows.filter(r=>r.is_mine).map(r=>r.record_id);assert.equal(new Set(ids).size,3);
    await profile(first,'第一局');assert.deepEqual((await call('qqt_my_win_leaderboard',[first.player_id,first.player_secret])).rows,mine.rows);
    await assert.rejects(()=>profile(first,'改写旧局'),/already accepted/);
    const peer=payload({game_duration_ms:3500});await submit(peer);await profile(peer,'同名但非本人');
    mine=await call('qqt_my_win_leaderboard',[first.player_id,first.player_secret]);
    assert.equal(mine.rows.filter(r=>r.nickname==='同名玩家'&&!r.is_mine).length,1,'nickname cannot authorize highlighting');
    const late=payload({nickname:'乱序玩家',game_duration_ms:6000,client_total_ms:6000});await submit(late);
    await allowNext(late);const newer={...late,client_match_id:randomUUID(),game_duration_ms:6500,client_total_ms:12500};
    await submit(newer);await profile(newer,'新局先提交');
    assert.equal((await profile(late,'旧局迟到')).superseded,true);
    const lateRows=(await call('qqt_my_win_leaderboard',[late.player_id,late.player_secret])).rows.filter(r=>r.is_mine);
    assert.deepEqual(lateRows.map(r=>r.victory_message),['旧局迟到','新局先提交']);
    assert.equal((await call('qqt_get_profile',[late.player_id,late.player_secret])).victory_message,'新局先提交');
    await assert.rejects(()=>call('qqt_my_win_leaderboard',[first.player_id,'b'.repeat(64)]),/credential mismatch/);
    await assert.rejects(()=>profile(first,'伪造',peer.client_match_id),/own completed match/);
    await assert.rejects(()=>profile(first,'改名',first.client_match_id,first.player_secret,'新名字'),/immutable/);
    const ip=(p,raw,mid=p.client_match_id)=>asRole('service_role',async()=>
      (await db.query('select public.qqt_record_match_ip($1,$2,$3) result',[p.player_id,mid,raw])).rows[0].result);
    assert.equal((await ip(first,'123.45.67.89')).ip_display,'123.*.*.89');
    assert.equal((await ip(first,'8.8.8.8')).ip_display,'123.*.*.89','retry cannot replace the first match IP');
    assert.equal((await ip(second,'2001:db8::42')).ip_display,'2001:*:*:42');
    await assert.rejects(()=>ip(first,'1.2.3.4',peer.client_match_id),/own completed match/);
    assert.equal((await ip(legacy,'1.2.3.4')).recorded,false,'historical match IP cannot be invented on retry');
    for (const result of ['loss','draw']) {
      await allowNext(first);const p={...first,client_match_id:randomUUID(),result,client_total_ms:result==='loss'?17000:22000};
      await submit(p);await profile(p,'非胜利资料');
    }
    assert.equal((await call('qqt_my_win_leaderboard',[first.player_id,first.player_secret])).rows.filter(r=>r.is_mine).length,3);
    for (let i=0;i<25;i++) await submit(payload({nickname:`对手${i}`,game_duration_ms:1000+i*10}));
    const top=await call('qqt_win_leaderboard');assert.equal(top.rows.length,20);assert(top.rows.every(r=>!r.is_mine&&!r.is_latest));
    mine=await call('qqt_my_win_leaderboard',[first.player_id,first.player_secret]);assert.equal(mine.rows.length,21);
    assert.equal(mine.rows.at(-1).rank,26);assert(mine.rows.at(-1).is_mine&&mine.rows.at(-1).is_latest);
    assert.deepEqual(await call('qqt_my_win_leaderboard',[randomUUID(),'c'.repeat(64)]),top);
    await allowNext(first);const fast={...first,client_match_id:randomUUID(),game_duration_ms:1050,client_total_ms:23050};
    await submit(fast);await profile(fast,'新的胜局');
    mine=await call('qqt_my_win_leaderboard',[first.player_id,first.player_secret]);assert.equal(mine.rows.length,20);
    assert.equal(mine.rows.filter(r=>r.is_latest).length,1);assert(mine.rows.find(r=>r.is_latest).rank<=20);
    const tie=payload({game_duration_ms:1050});await submit(tie);
    assert.deepEqual((await call('qqt_win_leaderboard')).rows,(await call('qqt_win_leaderboard')).rows,'tie order is stable');
    const empty=await call('qqt_my_win_leaderboard',[peer.player_id,peer.player_secret]);assert(empty.rows.length<=21);
    for(const row of (await call('qqt_win_leaderboard')).rows)
      assert.deepEqual(Object.keys(row).sort(),['rank','record_id','nickname','game_duration_ms','victory_message','player_ip','is_mine','is_latest'].sort());
    for(const role of ['anon','authenticated','service_role']) {
      await assert.rejects(()=>asRole(role,()=>db.query('select * from qqt_private.win_records')),/permission denied/);
      await assert.rejects(()=>asRole(role,()=>db.query('select qqt_private.win_leaderboard_data(null)')),/permission denied/);
    }
    for(const role of ['anon','authenticated'])await assert.rejects(()=>asRole(role,()=>db.query(
      'select public.qqt_record_match_ip($1,$2,$3)',[first.player_id,first.client_match_id,'1.2.3.4'])),/permission denied/);
    await assert.rejects(()=>profile(fast,'<img>'),/invalid profile/);
    await assert.rejects(()=>submit(payload({nickname:'<script>'})),/invalid result fields/);
    const snapshot=await db.query('select * from qqt_private.win_records order by record_id');
    await db.exec(fs.readFileSync(migration,'utf8'));
    assert.deepEqual((await db.query('select * from qqt_private.win_records order by record_id')).rows,snapshot.rows,'reentrant migration preserves snapshots/handles');
    assert.equal(fs.readFileSync(migration,'utf8'),fs.readFileSync('supabase/manual_win_records_20261008.sql','utf8'));
    assert((await db.query("select relrowsecurity from pg_class where oid='qqt_private.win_records'::regclass")).rows[0].relrowsecurity);
    console.log('Win-record SQL: per-match snapshots, legacy nulls, empty retention, identity/ownership, global rank/top20/extra, ties, idempotency, reentry, XSS, RLS and IP privacy passed');
  } finally { await db.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
