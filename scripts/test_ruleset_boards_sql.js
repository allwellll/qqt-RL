'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),{randomUUID}=require('node:crypto'),{PGlite}=require('@electric-sql/pglite');
(async()=>{
 const db=new PGlite();
 try{
  await db.exec('create role anon;create role authenticated;create role service_role;');
  for(const f of ['supabase/migrations/20261005140000_leaderboard.sql','supabase/migrations/20261005173000_leaderboard_ip_and_best_win.sql','supabase/manual_upgrade_20261006.sql','supabase/manual_profile_update_20261007.sql','supabase/migrations/20261008090000_win_records.sql'])await db.exec(fs.readFileSync(f,'utf8'));
  const call=async(name,args=[])=>{await db.exec('set role anon');try{return(await db.query(`select public.${name}(${args.map((_,i)=>'$'+(i+1)).join(',')}) result`,args)).rows[0].result;}finally{await db.exec('reset role');}};
  const payload=(version)=>({player_id:randomUUID(),player_secret:'a'.repeat(64),client_match_id:randomUUID(),nickname:'规则隔离测试',victory_message:'',result:'win',game_duration_ms:5000,wall_duration_ms:5000,client_total_ms:5000,opponent:'bun.hunter',difficulty:'hard',seed:7,mode:'1v1',map_id:'training806',completed_at:new Date().toISOString(),client_version:'dev',...(version?{ruleset_version:version}:{})});
  const old=payload();await call('qqt_submit_result',[old]);
  const sql=fs.readFileSync('supabase/migrations/20261031074000_ruleset_boards.sql','utf8');await db.exec(sql);
  const v2=payload('bun-native-v2');let receipt=await call('qqt_submit_result',[v2]);assert.equal(receipt.ruleset_version,'bun-native-v2');assert.equal(receipt.match_rank.total,1);
  assert.equal((await call('qqt_submit_result',[v2])).games,1);
  await assert.rejects(()=>call('qqt_submit_result',[{...v2,ruleset_version:'stock-window-v1'}]),/ruleset mismatch/);
  await assert.rejects(()=>call('qqt_submit_result',[{...payload(),ruleset_version:'bad'}]),/ruleset/);
  const v1=await call('qqt_my_win_leaderboard_ruleset',[old.player_id,old.player_secret,'stock-window-v1']);
  const native=await call('qqt_my_win_leaderboard_ruleset',[v2.player_id,v2.player_secret,'bun-native-v2']);
  assert.equal(v1.rows.length,1);assert.equal(native.rows.length,1);assert.notEqual(v1.rows[0].record_id,native.rows[0].record_id);assert.equal(native.ruleset_contract_version,2);
  assert.equal((await call('qqt_win_leaderboard')).rows.length,1);
  assert.equal((await call('qqt_my_win_leaderboard',[old.player_id,old.player_secret])).rows.length,1);
  await assert.rejects(()=>call('qqt_my_win_leaderboard_ruleset',[v2.player_id,'b'.repeat(64),'bun-native-v2']),/credential mismatch/);
  assert.equal((await db.query('select count(*) n from qqt_private.match_results')).rows[0].n,2);
  assert.equal((await db.query('select ruleset_version from qqt_private.match_results where client_match_id=$1',[old.client_match_id])).rows[0].ruleset_version,'stock-window-v1');
  await db.exec(sql);assert.equal((await call('qqt_win_leaderboard')).rows.length,1);
  assert.equal((await call('qqt_ruleset_contract')).ruleset_contract_version,2);
  await db.exec('set role anon');await assert.rejects(()=>db.query('select * from qqt_private.match_results'),/permission denied/);await db.exec('reset role');
  console.log('ruleset migration PASS: preservation, isolation, retry, ACL, repeat migration');
 }finally{await db.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
