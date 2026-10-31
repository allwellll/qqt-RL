'use strict';
const assert=require('node:assert/strict'),{webcrypto}=require('node:crypto'),LB=require('./leaderboard');
const metadata={opponent:'bun.hunter',difficulty:'hard',seed:7,mode:'1v1',map_id:'training806',client_version:'test'};
(async()=>{
 const values=new Map(),storage={getItem:k=>values.get(k)||null,setItem:(k,v)=>values.set(k,v)};
 let now=Date.now(),offline=false,wrong=false;const calls=[];
 const make=version=>LB.createClient({config:{url:'https://fixture.invalid',publishableKey:'public',leaderboardRpc:'qqt_my_win_leaderboard',...(version?{rulesetVersion:version}:{})},storage,crypto:webcrypto,now:()=>now,fetch:async(url,o)=>{
  const args=JSON.parse(o.body);calls.push({url,args});if(offline)throw Error('offline');
  if(url.endsWith('qqt_my_win_leaderboard_ruleset'))return {ok:true,json:async()=>({leaderboard_contract_version:1,ruleset_contract_version:2,ruleset_version:wrong?'stock-window-v1':args.p_ruleset_version,rows:[]})};
  if(url.endsWith('qqt_my_win_leaderboard'))return {ok:true,json:async()=>({leaderboard_contract_version:1,rows:[]})};
  return {ok:true,json:async()=>({level:1,points:3,wins:1,games:1})};
 }});
 const old=make();old.setProfile('分榜测试','');offline=true;const pending=old.begin(metadata);now+=5000;await old.finish(pending,{result:'win',gameDurationMs:5000});assert.equal(old.state().pending,1);
 const v2=make('bun-native-v2');offline=false;await v2.retry();const legacy=calls.filter(x=>x.url.endsWith('qqt_submit_result')).at(-1).args.p_payload;assert(!('ruleset_version'in legacy),'legacy retry keeps old unversioned payload');
 const match=v2.begin(metadata);assert.equal(match.ruleset_version,'bun-native-v2');now+=5000;await v2.finish(match,{result:'win',gameDurationMs:5000});assert.equal(calls.filter(x=>x.url.endsWith('qqt_submit_result')).at(-1).args.p_payload.ruleset_version,'bun-native-v2');
 await v2.setBoardRuleset('stock-window-v1');assert.equal(calls.at(-1).args.p_ruleset_version,'stock-window-v1');
 wrong=true;assert.equal(await v2.setBoardRuleset('bun-native-v2'),false);assert.deepEqual(v2.state().rows,[]);assert.throws(()=>v2.setBoardRuleset('invalid'));
 const receipt={...metadata,result:'win',duration_ms:5000,rank:1,total:1,percentile:0,comparison:'player-best-v1'};
 assert(!LB.validRanking({...receipt,ruleset_version:'stock-window-v1'},{...match,result:'win',game_duration_ms:5000}));
 assert(LB.validRanking({...receipt,ruleset_version:'bun-native-v2'},{...match,result:'win',game_duration_ms:5000}));
 console.log('ruleset client PASS: versioned submission, immutable legacy queue, board contract, mismatched ranking rejection');
})().catch(e=>{console.error(e);process.exitCode=1});
