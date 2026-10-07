'use strict';
const assert=require('assert'),{webcrypto}=require('crypto'),LB=require('./leaderboard');
const meta={opponent:'bun.coop_hunter',difficulty:'hard',seed:20261007,mode:'1v2',map_id:'training806',client_version:'dev'};
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {resolve,promise};};
(async()=>{
 let now=10000,failResult=false,failProfile=false,failRead=false,holdRead=null,holdProfile=null;
 let holdIdentity=null,registered=false,server={nickname:null,victory_message:'',rank:5,best_win_duration_ms:9000};
 const writes=[],profiles=[],reads=[],store=new Map();
 const make=()=>LB.createClient({config:{url:'https://mock',publishableKey:'public'},crypto:webcrypto,now:()=>now,
  storage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v)},fetch:async(url,opts)=>{
   const body=JSON.parse(opts.body);
   if(url.endsWith('qqt_get_profile')) { const result=registered
    ?{profile_contract_version:2,registered:true,nickname:server.nickname,victory_message:server.victory_message}
    :{profile_contract_version:2,registered:false};if(holdIdentity){const d=holdIdentity;holdIdentity=null;await d.promise;}return {ok:true,json:async()=>result}; }
   if(url.endsWith('qqt_leaderboard')){const old={...server};reads.push(old);if(holdRead){const d=holdRead;holdRead=null;await d.promise;}return {ok:!failRead,status:503,json:async()=>[old]};}
   if(url.endsWith('qqt_submit_result')){const p=body.p_payload;writes.push(p);if(failResult)throw new TypeError('offline');
    if(!registered){registered=true;server={...server,nickname:p.nickname,victory_message:p.victory_message};}
    server={...server,rank:1};return {ok:true,json:async()=>({level:1,points:3,wins:1,games:writes.length,client_match_id:p.client_match_id,match_upgraded:false})};}
   profiles.push(body);if(failProfile)throw new TypeError('offline');if(holdProfile)await holdProfile.promise;
   server={...server,victory_message:body.p_victory_message||server.victory_message};
   return {ok:true,json:async()=>({saved:true,profile_contract_version:2,client_match_id:body.p_client_match_id,nickname:server.nickname,victory_message:server.victory_message,superseded:false})};
  }});
 let c=make();await c.syncProfile();let match=c.begin(meta);now+=5000;await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
 assert.equal(c.state().savedNickname,false);assert.equal(c.state().settlement.draft.nickname,'');
 assert.equal(await c.submitCard(),false);assert.equal(writes.length,0,'first nickname required');
 c.setDraft('首次昵称','首次宣言');assert(c.closeSettlement());c.clearSettlement();assert(c.openPending());
 c=make();await c.syncProfile();assert.deepEqual(c.state().settlement.draft,{nickname:'首次昵称',victory_message:'首次宣言'});
 failResult=true;await Promise.all([c.submitCard(),c.submitCard(),c.submitCard()]);assert.equal(writes.length,1);assert.match(c.state().settlement.cardStatus,/提交失败/);
 failResult=false;failProfile=true;await c.submitCard();assert.equal(writes.length,2);assert(c.state().settlement.submitted);assert.match(c.state().settlement.cardStatus,/感言提交失败/);
 c=make();await c.syncProfile();assert(c.state().settlement.submitted,'reload retains result receipt');assert.equal(c.state().savedNickname,true);
 failProfile=false;holdRead=deferred();const stale=holdRead,oldRead=c.refresh();await Promise.resolve();
 await Promise.all([c.submitCard(),c.submitCard()]);assert.equal(writes.length,2,'profile retry never resubmits result');assert.equal(profiles.length,2);
 stale.resolve();await oldRead;assert.equal(c.state().rows[0].rank,1,'stale read cannot overwrite post-profile refresh');
 assert.equal(c.state().rows[0].nickname,'首次昵称');assert.equal(c.state().rows[0].victory_message,'首次宣言');assert(c.state().settlement.complete);

 c.clearSettlement();match=c.begin(meta);now+=5000;await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});
 assert.equal(c.state().savedNickname,true);assert.equal(c.state().settlement.draft.nickname,'首次昵称');c.setDraft('篡改名字','每局新宣言');
 holdIdentity=deferred();const oldIdentity=holdIdentity,identityRead=c.syncProfile();
 holdProfile=deferred();const request=c.submitCard();await new Promise(r=>setTimeout(r,0));
 assert.equal(profiles.at(-1).p_nickname,null,'returning client never sends mutable nickname');assert.equal(writes.at(-1).nickname,'首次昵称');
 c.setDraft('仍然篡改','下一局宣言');holdProfile.resolve();holdProfile=null;await request;
 oldIdentity.resolve();await identityRead;assert.equal(c.state().victory_message,'每局新宣言','late identity read cannot roll profile back');
 assert.equal(server.nickname,'首次昵称');assert.equal(server.victory_message,'每局新宣言');assert.match(c.state().settlement.cardStatus,/下一局/);
 c.clearSettlement();match=c.begin(meta);now+=5000;await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});
 assert.equal(c.state().settlement.draft.victory_message,'下一局宣言');c.setDraft('x','');failRead=true;await c.submitCard();
 assert.equal(server.victory_message,'每局新宣言','empty declaration retains server value');assert.match(c.state().settlement.cardStatus,/刷新失败/);
 failRead=false;await c.refresh();assert.equal(c.state().rows[0].victory_message,'每局新宣言');assert.equal(c.state().settlement.cardStatus,'提交成功！');
 assert(!JSON.stringify(c.state()).includes('升级时更新'));
 // A pre-registration identity read must not restore the nickname field after success.
 store.clear();registered=false;c=make();await c.syncProfile();match=c.begin(meta);now+=5000;
 await c.finish(match,{result:'draw',gameDurationMs:5000,autoSubmit:false});c.setDraft('新身份昵称','新身份宣言');
 holdIdentity=deferred();const preRegistration=holdIdentity,preRegistrationRead=c.syncProfile();
 await c.submitCard();assert.equal(c.state().savedNickname,true);preRegistration.resolve();await preRegistrationRead;
 assert.equal(c.state().savedNickname,true,'late registered=false response cannot restore nickname editor');
 assert.equal(c.state().nickname,'新身份昵称');assert.equal(c.state().victory_message,'新身份宣言');
 console.log('Round4 card: first registration, returning declaration-only, result/profile retry, immutable intent, next-match edits, empty retention and stale refresh passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
