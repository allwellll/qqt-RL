'use strict';
const assert=require('assert'),{webcrypto}=require('crypto'),LB=require('./leaderboard');
const meta={opponent:'bun.coop_hunter',difficulty:'hard',seed:20261007,mode:'1v2',map_id:'training806',client_version:'dev'};
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {resolve,promise};};
(async()=>{
 let now=10000, failResult=false,failProfile=false,failRead=false,upgrade=true,hold=null,profileSaved=false;
 let server={nickname:'旧昵称',victory_message:'旧宣言',rank:5,best_win_duration_ms:9000};
 const writes=[],profiles=[],reads=[],store=new Map();
 const make=()=>LB.createClient({config:{url:'https://mock',publishableKey:'public'},crypto:webcrypto,now:()=>now,
 storage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v)},fetch:async(url,opts)=>{
  if(url.endsWith('qqt_leaderboard')){const old={...server};reads.push(old);if(hold){const d=hold;hold=null;await d.promise;}return {ok:!failRead,status:503,json:async()=>[old]};}
  if(url.endsWith('qqt_submit_result')){const p=JSON.parse(opts.body).p_payload;writes.push(p);if(failResult)throw new TypeError('offline');server={...server,rank:1,...(!profileSaved?{nickname:p.nickname,victory_message:p.victory_message}:{})};return {ok:true,json:async()=>({level:2,points:3,wins:1,games:1,client_match_id:p.client_match_id,match_upgraded:upgrade})};}
  const p=JSON.parse(opts.body);profiles.push(p);if(failProfile)throw new TypeError('offline');server={...server,nickname:p.p_nickname,victory_message:p.p_victory_message||server.victory_message};profileSaved=true;return {ok:true,json:async()=>({saved:true,nickname:server.nickname,victory_message:server.victory_message})};
 }});
 let c=make(),match=c.begin(meta);now+=5000;await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
 assert.equal(c.state().settlement.draft.nickname,'');assert.equal(await c.submitCard(),false);assert.equal(writes.length,0,'first nickname required');
 c.setDraft('新昵称','胜利宣言');assert(c.closeSettlement());c.clearSettlement();assert(c.openPending());assert.equal(c.state().settlement.draft.nickname,'新昵称');
 c=make();assert.equal(c.state().settlement.draft.victory_message,'胜利宣言','reload restores input');
 hold=deferred();const slow=hold,oldRead=c.refresh();
 failResult=true;await Promise.all([c.submitCard(),c.submitCard(),c.submitCard()]);assert.equal(writes.length,1);
 assert.equal(c.state().settlement.cardStatus,'提交失败，请重试');failResult=false;failProfile=true;
 await c.submitCard();assert.equal(writes.length,2);assert.deepEqual(writes[0],writes[1]);assert(c.state().settlement.submitted);
 assert.equal(c.state().rows[0].rank,1,'confirmed result immediately refreshes list before profile finishes');
 slow.resolve();await oldRead;assert.equal(c.state().rows[0].rank,1,'old list cannot overwrite new result');
 c=make();assert(c.state().settlement.submitted,'profile failure retains server receipt on reload');
 failProfile=false;await Promise.all([c.submitCard(),c.submitCard()]);assert.equal(writes.length,2,'profile retry does not resubmit result');
 assert.equal(profiles.length,2);assert.equal(c.state().rows[0].victory_message,'胜利宣言');assert(c.state().settlement.complete);
 assert.equal(await c.submitCard(),false);assert.equal(writes.length,2);
 c.clearSettlement();match=c.begin(meta);now+=5000;await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});
 assert.equal(c.state().settlement.draft.nickname,'新昵称','returning nickname reused');c.setDraft('新昵称','');
 failRead=true;await c.submitCard();assert(c.state().settlement.complete);assert.match(c.state().settlement.cardStatus,/提交成功.*刷新失败/);
 const total=writes.length;failRead=false;await c.refresh();assert.equal(writes.length,total);assert.equal(c.state().settlement.cardStatus,'提交成功！','refresh recovery clears failure message');assert.equal(c.state().rows[0].victory_message,'胜利宣言','empty declaration preserves server value');
 // Edit during an in-flight combined action: accepted input saves, newer draft stays editable.
 c.clearSettlement();match=c.begin(meta);now+=5000;await c.finish(match,{result:'loss',gameDurationMs:5000,autoSubmit:false});c.setDraft('新昵称','已确认');
 hold=deferred();const gate=hold,request=c.submitCard();await Promise.resolve();await Promise.resolve();
 c.setDraft('更新昵称','新的修改');assert.equal(c.closeSettlement(),false,'close blocked during combined request');gate.resolve();await request;
 assert.equal(c.state().settlement.complete,false);assert.equal(c.state().settlement.draft.victory_message,'新的修改');
 await c.submitCard();assert.equal(c.state().rows[0].nickname,'更新昵称');assert.equal(c.state().rows[0].victory_message,'新的修改');
 upgrade=false; c.clearSettlement();match=c.begin(meta);now+=5000;
 await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
 const saved={nickname:c.state().nickname,message:c.state().victory_message};
 c.setDraft(saved.nickname,'下次升级宣言'); await c.submitCard();
 assert.equal(c.state().nickname,saved.nickname);assert.equal(c.state().victory_message,saved.message,'non-upgrade preserves actual saved profile');
 assert.match(c.state().settlement.cardStatus,/升级时更新/);
 assert.equal(c.state().rows[0].victory_message,saved.message,'result RPC does not change an existing profile without upgrade RPC');
 c.clearSettlement();match=c.begin(meta);now+=5000;upgrade=true;
 await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
 assert.equal(c.state().settlement.draft.victory_message,'下次升级宣言','deferred message remains available next match');
 await c.submitCard();assert.equal(c.state().rows[0].victory_message,'下次升级宣言');
 // Nickname-only changes on a non-upgrade must survive into the next upgrade.
 upgrade=false;c.clearSettlement();match=c.begin(meta);now+=5000;
 await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});
 c.setDraft('下次升级昵称','');await c.submitCard();
 assert.match(c.state().settlement.cardStatus,/新资料.*升级/);
 assert.equal(c.state().nickname,'更新昵称');
 c.clearSettlement();match=c.begin(meta);now+=5000;upgrade=true;
 await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
 assert.equal(c.state().settlement.draft.nickname,'下次升级昵称');await c.submitCard();
 assert.equal(c.state().rows[0].nickname,'下次升级昵称');
 // Published Round 2 queue-only state migrates without implicit writes.
 const key='qqt.leaderboard.v1',legacy=JSON.parse(store.get(key));
 const {player_secret,...payload}=writes[0];legacy.queue=[{...payload,approved:false}];delete legacy.cards;
 store.set(key,JSON.stringify(legacy));const beforeLegacy=writes.length;c=make();
 assert.equal(writes.length,beforeLegacy);assert(c.openPending());assert(c.state().settlement.eligible);
 assert.equal(c.state().settlement.client_match_id,payload.client_match_id);
 assert.equal(c.state().settlement.draft.nickname,payload.nickname);
 c.closeSettlement();c=make();assert(c.openPending());
 await c.submitCard();assert.equal(writes.length,beforeLegacy+1);
 // First non-upgrade submission accepts A while B is edited: preserve B for upgrade.
 store.clear();profileSaved=false;upgrade=false;c=make();match=c.begin(meta);now+=5000;
 await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});c.setDraft('首次A','宣言A');
 hold=deferred();const firstGate=hold,firstRequest=c.submitCard();await Promise.resolve();await Promise.resolve();
 c.setDraft('首次B','宣言B');firstGate.resolve();await firstRequest;
 assert.equal(c.state().nickname,'首次A');assert.equal(c.state().settlement.complete,false);
 const beforeSecond=writes.length;await c.submitCard();assert.equal(writes.length,beforeSecond);
 assert.match(c.state().settlement.cardStatus,/新资料.*升级/);
 c=make();c.clearSettlement();match=c.begin(meta);now+=5000;upgrade=true;
 await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
 assert.deepEqual(c.state().settlement.draft,{nickname:'首次B',victory_message:'宣言B'});
 await c.submitCard();assert.equal(c.state().rows[0].nickname,'首次B');
 assert.equal(c.state().rows[0].victory_message,'宣言B');
 // An approved failed older match is drained with a new one, and keeps its receipt.
 c.clearSettlement();match=c.begin(meta);const older=match.client_match_id;now+=5000;
 await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});c.setDraft('旧局昵称','旧局宣言');
 failResult=true;await c.submitCard();c.clearSettlement();match=c.begin(meta);now+=5000;
 await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});c.setDraft('新局昵称','新局宣言');failResult=false;
 hold=deferred();const oldGate=hold,newRequest=c.submitCard();await Promise.resolve();await Promise.resolve();
 assert.equal(c.openPending(),false,'cannot switch cards during a combined write');oldGate.resolve();await newRequest;
 const beforeRecover=writes.length;c=make();assert(c.openPending());
 assert.equal(c.state().settlement.client_match_id,older);assert(c.state().settlement.submitted);assert(c.state().settlement.upgraded);
 assert.equal(c.state().pending,0);await c.submitCard();assert.equal(writes.length,beforeRecover,'recovered older receipt only retries profile');
 assert(c.state().settlement.complete);assert.equal(c.state().rows[0].victory_message,'旧局宣言');
 // The earliest accepted registration is authoritative even when a later card is current.
 store.clear();profileSaved=false;upgrade=false;c=make();match=c.begin(meta);now+=5000;
 await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});c.setDraft('首局昵称','首局宣言');failResult=true;await c.submitCard();
 c.clearSettlement();match=c.begin(meta);now+=5000;
 await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});c.setDraft('次局昵称','次局宣言');failResult=false;await c.submitCard();
 assert.equal(c.state().nickname,'首局昵称');assert.equal(c.state().victory_message,'首局宣言');
 assert.equal(writes.at(-1).nickname,'首局昵称');assert.equal(writes.at(-1).victory_message,'首局宣言');
 assert.equal(c.state().rows[0].nickname,'首局昵称');assert.match(c.state().settlement.cardStatus,/新资料.*升级/);
 assert(!JSON.stringify(c.state()).includes('战绩已暂存'));
 console.log('Combined card: first/returning, required input, close/R/reload, frozen retry, upgrade/profile retry, stale read, refresh failure and concurrent edits passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
