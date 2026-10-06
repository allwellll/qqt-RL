'use strict';
const assert = require('assert');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { PGlite } = require('@electric-sql/pglite');
(async () => {
  const db = new PGlite();
  try {
    await db.exec('create role anon; create role authenticated; create role service_role;');
    for (const file of ['20261005140000_leaderboard.sql','20261005173000_leaderboard_ip_and_best_win.sql']) {
      await db.exec(fs.readFileSync(`supabase/migrations/${file}`,'utf8'));
    }
    // Execute exactly the supplied SQL Editor bundle, then verify new migration reapplication.
    await db.exec(fs.readFileSync('supabase/manual_upgrade_20261006.sql','utf8'));
    await db.exec(fs.readFileSync('supabase/migrations/20261006220000_upgrade_profile_and_raw_ip.sql','utf8'));
    const asRole = async (role, fn) => { await db.exec(`set role ${role}`); try { return await fn(); } finally { await db.exec('reset role'); } };
    const call = async (name, args) => (await db.query(`select public.${name}(${args.map((_,i)=>`$${i+1}`).join(',')}) result`,args)).rows[0].result;
    const submit = p => asRole('anon', () => call('qqt_submit_result',[JSON.stringify(p)]));
    const profile = (p, nick, message, secret=p.player_secret, mid=p.client_match_id) => asRole('anon',()=>call('qqt_update_profile',[p.player_id,secret,mid,nick,message]));
    const outcomes = [];
    for (const result of ['win','loss','draw']) {
      const pid = randomUUID();
      const p = { player_id:pid, player_secret:'a'.repeat(64), client_match_id:randomUUID(),nickname:`test-${result}`,
        victory_message:'旧宣言',result,game_duration_ms:5000,wall_duration_ms:5000,client_total_ms:5000,
        opponent:'bun.coop_hunter',difficulty:'hard',seed:7,mode:'1v2',map_id:'training806',completed_at:new Date().toISOString(),client_version:'dev' };
      const first = await submit(p); assert.equal(first.match_upgraded,false);
      await assert.rejects(()=>profile(p,'新名字','新宣言'),/requires completed level upgrade/);
      await db.query('update qqt_private.player_progress set points=9,level=1 where player_id=$1',[pid]);
      await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[pid]);
      const upgrade = {...p, client_match_id:randomUUID(),client_total_ms:10000};
      const upgraded = await submit(upgrade); assert.equal(upgraded.match_upgraded,true); assert.equal(upgraded.level,2);
      assert.deepEqual(await submit(upgrade),upgraded,'idempotent upgrade returns same proof');
      await assert.rejects(()=>profile(upgrade,'ok','x','b'.repeat(64)),/credential mismatch/);
      await assert.rejects(()=>profile(upgrade,'ok','x',undefined,randomUUID()),/requires completed level upgrade/);
      for (const [nick,msg] of [['',''],['a'.repeat(25),''],['<img>',''],['ok','a\x01b'],['ok','x'.repeat(81)]]) {
        await assert.rejects(()=>profile(upgrade,nick,msg),/invalid profile/);
      }
      assert.deepEqual(await profile(upgrade,'安全😀','首次宣言'),{saved:true,nickname:'安全😀',victory_message:'首次宣言'});
      await profile(upgrade,'安全😀','第二宣言'); await profile(upgrade,'安全😀','');
      let row = (await db.query('select nickname,victory_message from qqt_private.players where player_id=$1',[pid])).rows[0];
      assert.deepEqual(row,{nickname:'安全😀',victory_message:'第二宣言'});
      await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[pid]);
      await submit({...p,client_match_id:randomUUID(),client_total_ms:15000});
      row = (await db.query('select nickname,victory_message from qqt_private.players where player_id=$1',[pid])).rows[0];
      assert.equal(row.victory_message,'第二宣言','old queued payload cannot overwrite profile RPC');
      for (const [ip,masked] of [['123.45.67.89','123.*.*.89'],['2001:db8::42','2001:*:*:42'],['::1','0:*:*:1'],['::ffff:192.0.2.1','0:*:*:201']]) {
        const stored = await asRole('service_role',()=>call('qqt_record_player_ip',[pid,ip]));
        assert.deepEqual(stored,{recorded:true,ip_display:masked});
        const privateRow = (await db.query('select host(raw_ip) ip,ip_display from qqt_private.players where player_id=$1',[pid])).rows[0];
        assert(privateRow.ip); assert.equal(privateRow.ip_display,masked);
        const top = await asRole('anon',()=>db.query('select * from public.qqt_leaderboard()'));
        assert.equal(top.rows.find(x=>x.nickname==='安全😀' && x.player_ip===masked)?.player_ip,masked);
        assert(!JSON.stringify(top.rows).includes(ip));
        assert(!Object.keys(top.rows[0]).some(k=>/raw|secret|player_id|hash/.test(k)));
      }
      for (const invalid of ['1.2.3.999','1.2.3.4/24','::::','1.2.3.4;drop table x','']) {
        await assert.rejects(()=>asRole('service_role',()=>call('qqt_record_player_ip',[pid,invalid])),/invalid network metadata/);
      }
      for (const role of ['anon','authenticated']) {
        await assert.rejects(()=>asRole(role,()=>call('qqt_record_player_ip',[pid,'8.8.8.8'])),/permission denied/);
        await assert.rejects(()=>asRole(role,()=>db.query('select raw_ip from qqt_private.players')),/permission denied/);
      }
      await assert.rejects(()=>asRole('service_role',()=>call('qqt_record_player_ip',[pid,'8.*.*.8','a'.repeat(64)])),/permission denied/);
      outcomes.push({result,level:upgraded.level});
    }
    console.log('Incremental SQL: upgrade win/loss/draw, profile auth/XSS/empty preservation, stale queue, raw inet and public masking/ACL passed',outcomes);
  } finally { await db.close(); }
})().catch(e=>{console.error(e.message);process.exitCode=1;});
