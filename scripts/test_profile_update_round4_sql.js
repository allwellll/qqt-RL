'use strict';
const assert = require('assert');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { PGlite } = require('@electric-sql/pglite');
(async () => {
  const db = new PGlite();
  try {
    assert.equal(fs.readFileSync('supabase/manual_profile_update_20261007.sql','utf8'),
      fs.readFileSync('supabase/migrations/20261007120000_match_profile_updates.sql','utf8'),'manual bundle matches incremental migration');
    await db.exec('create role anon; create role authenticated; create role service_role;');
    for (const file of ['20261005140000_leaderboard.sql','20261005173000_leaderboard_ip_and_best_win.sql'])
      await db.exec(fs.readFileSync(`supabase/migrations/${file}`, 'utf8'));
    await db.exec(fs.readFileSync('supabase/manual_upgrade_20261006.sql', 'utf8'));
    const html=fs.readFileSync('docs/qqt_round4_sql_upgrade_20261007.html','utf8');
    const block=id=>html.match(new RegExp(`<pre id="${id}">([\\s\\S]*?)</pre>`))[1]
      .replace(/&#x27;/g,"'").replace(/&quot;/g,'"').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
    assert.equal(block('upgrade'),fs.readFileSync('supabase/manual_profile_update_20261007.sql','utf8'),'offline HTML embeds exact executable SQL');
    await db.exec(block('preflight'));
    await db.exec(fs.readFileSync('supabase/manual_profile_update_20261007.sql', 'utf8'));
    await db.exec(fs.readFileSync('supabase/migrations/20261007120000_match_profile_updates.sql', 'utf8'));
    await db.exec(block('verify'));
    const asRole = async (role, fn) => { await db.exec(`set role ${role}`); try { return await fn(); } finally { await db.exec('reset role'); } };
    const call = async (name, args) => (await db.query(`select public.${name}(${args.map((_,i)=>`$${i+1}`).join(',')}) result`, args)).rows[0].result;
    const submit = p => asRole('anon', () => call('qqt_submit_result', [JSON.stringify(p)]));
    const update = (p, message, {secret=p.player_secret, mid=p.client_match_id, nickname=null}={}) =>
      asRole('anon', () => call('qqt_update_profile', [p.player_id, secret, mid, nickname, message]));
    const get = (p, secret=p.player_secret) => asRole('anon', () => call('qqt_get_profile', [p.player_id, secret]));
    const payload = (overrides={}) => ({player_id:randomUUID(),player_secret:'a'.repeat(64),client_match_id:randomUUID(),
      nickname:'首次昵称',victory_message:'初始宣言',result:'loss',game_duration_ms:5000,wall_duration_ms:5000,
      client_total_ms:5000,opponent:'bun.coop_hunter',difficulty:'hard',seed:7,mode:'1v2',map_id:'training806',
      completed_at:new Date().toISOString(),client_version:'dev',...overrides});

    const unknown=payload(); assert.deepEqual(await get(unknown), {profile_contract_version:2,registered:false});
    const first=payload(); const firstResult=await submit(first); assert.equal(firstResult.match_upgraded,false);
    assert.deepEqual(await get(first), {profile_contract_version:2,registered:true,nickname:'首次昵称',victory_message:'初始宣言'});
    const saved=await update(first,'普通局新宣言');
    assert.deepEqual(saved,{saved:true,profile_contract_version:2,client_match_id:first.client_match_id,nickname:'首次昵称',victory_message:'普通局新宣言',superseded:false});
    assert.deepEqual(await update(first,'普通局新宣言'),saved,'same match and same intent is idempotent');
    await assert.rejects(()=>update(first,'同局不同宣言'),/already accepted/);
    await assert.rejects(()=>update(first,'x',{secret:'b'.repeat(64)}),/credential mismatch/);
    await assert.rejects(()=>update(first,'x',{mid:randomUUID()}),/own completed match/);
    await assert.rejects(()=>update(first,'x',{nickname:'篡改昵称'}),/nickname is immutable/);
    for (const message of ['x'.repeat(81),'<img>','a\x01b']) await assert.rejects(()=>update(first,message,{mid:randomUUID()}),/invalid profile/);

    await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[first.player_id]);
    const second={...first,client_match_id:randomUUID(),result:'win',client_total_ms:10000,completed_at:new Date().toISOString(),nickname:'恶意改名',victory_message:'旧队列覆盖'};
    await submit(second);
    let profile=await get(first); assert.equal(profile.nickname,'首次昵称');assert.equal(profile.victory_message,'普通局新宣言','later result cannot overwrite profile');
    await update(second,'每局可更新'); profile=await get(first);assert.equal(profile.nickname,'首次昵称');assert.equal(profile.victory_message,'每局可更新');

    await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[first.player_id]);
    const third={...first,client_match_id:randomUUID(),result:'win',client_total_ms:15000,completed_at:new Date().toISOString()};
    await submit(third); await update(third,''); profile=await get(first); assert.equal(profile.victory_message,'每局可更新','empty declaration retains old value');

    const other=payload({player_secret:'c'.repeat(64),nickname:'另一玩家'});await submit(other);
    await assert.rejects(()=>update(first,'伪造',{mid:other.client_match_id}),/own completed match/);
    await assert.rejects(()=>get(first,'d'.repeat(64)),/credential mismatch/);
    await assert.rejects(()=>submit({...first,player_secret:'d'.repeat(64)}),/credential mismatch/);
    await assert.rejects(()=>submit({...other,client_match_id:first.client_match_id}),/match id already used/);
    for(const field of ['nickname','victory_message'])
      for(const value of ['<script>','a\x01b'])await assert.rejects(()=>submit(payload({[field]:value})),/invalid result fields/);

    // Real upgrade uses exactly the same immutable-name/profile contract.
    await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[first.player_id]);
    const fourth={...first,client_match_id:randomUUID(),result:'win',client_total_ms:20000};
    assert.equal((await submit(fourth)).match_upgraded,true);
    await update(fourth,'升级局宣言',{nickname:'首次昵称'});
    await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[first.player_id]);
    const late={...first,client_match_id:randomUUID(),client_total_ms:5000};await submit(late);
    assert.equal((await update(late,'离线旧局')).superseded,true,'late-arriving old match cannot replace newer profile');
    assert.equal((await update(late,'离线旧局')).victory_message,'升级局宣言');
    assert.equal((await update(first,'普通局新宣言')).superseded,true,'old receipt replay returns newest server value');
    assert.equal((await get(first)).victory_message,'升级局宣言');

    // Deployed pre-profile_saved identities keep their names on the first new result.
    assert.equal((await db.query('select profile_saved from qqt_private.players where player_id=$1',[other.player_id])).rows[0].profile_saved,false);
    await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1",[other.player_id]);
    await submit({...other,client_match_id:randomUUID(),client_total_ms:10000,nickname:'恶意重命名',victory_message:'结果绕过资料'});
    assert.equal((await get(other)).nickname,'另一玩家');assert.equal((await get(other)).victory_message,'初始宣言');
    const ip=await asRole('service_role',()=>call('qqt_record_player_ip',[first.player_id,'123.45.67.89']));
    assert.equal(ip.ip_display,'123.*.*.89');
    await update(fourth,'升级局宣言');
    assert.equal((await db.query('select host(raw_ip) ip from qqt_private.players where player_id=$1',[first.player_id])).rows[0].ip,'123.45.67.89');
    for (const role of ['anon','authenticated','service_role'])
      await assert.rejects(()=>asRole(role,()=>db.query('select * from qqt_private.profile_receipts')),/permission denied/);
    const top=await asRole('anon',()=>db.query('select * from public.qqt_leaderboard()'));
    const row=top.rows.find(x=>x.nickname==='首次昵称');assert.equal(row.victory_message,'升级局宣言');assert.equal(row.player_ip,'123.*.*.89');
    assert(!Object.keys(row).some(k=>/player_id|secret|raw|hash/.test(k)));
    const policies=await db.query("select relname,relrowsecurity from pg_class join pg_namespace n on n.oid=relnamespace where n.nspname='qqt_private' and relkind='r'");
    assert.equal(policies.rows.length,5);assert(policies.rows.every(r=>r.relrowsecurity));
    for(const role of ['anon','authenticated']){
      await assert.rejects(()=>asRole(role,()=>db.query('select raw_ip from qqt_private.players')),/permission denied/);
      await assert.rejects(()=>asRole(role,()=>call('qqt_record_player_ip',[first.player_id,'1.2.3.4'])),/permission denied/);
    }
    console.log('Round4 SQL: executable/reentrant bundle, upgrade/non-upgrade, identity/match forgery, immutable legacy nickname, empty retention, old/late receipts, idempotency, XSS, RLS/ACL and raw IP privacy passed');
  } finally { await db.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
