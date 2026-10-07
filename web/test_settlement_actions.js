'use strict';
const assert = require('assert');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard');
const meta = {opponent:'bun.coop_hunter',difficulty:'hard',seed:20261007,mode:'1v2',map_id:'training806',client_version:'dev'};
(async () => {
  let now = 10000, fail = true, release, upgraded = false, profileRelease;
  const storageMap = new Map(), writes = [];
  const storage = {getItem:k=>storageMap.get(k)||null, setItem:(k,v)=>storageMap.set(k,v)};
  const make = () => LB.createClient({config:{url:'https://mock',publishableKey:'public'}, storage, crypto:webcrypto, now:()=>now,
    fetch:async (url, options) => {
      if (url.endsWith('qqt_leaderboard')) return {ok:true,json:async()=>[]};
      if (url.endsWith('qqt_update_profile')) {
        if (profileRelease) await new Promise(resolve=>{profileRelease.resolve=resolve;});
        return {ok:true,json:async()=>({saved:true,nickname:'玩家',victory_message:'练习'})};
      }
      const payload = JSON.parse(options.body).p_payload; writes.push(payload);
      if (release) await new Promise(resolve=>{release.resolve=resolve;});
      if (fail) throw new TypeError('offline');
      return {ok:true,json:async()=>({level:2,points:10,wins:1,games:1,client_match_id:payload.client_match_id,match_upgraded:upgraded})};
    }});
  const c = make(), match = c.begin(meta); now += 5000;
  await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
  await c.finish(match,{result:'win',gameDurationMs:5000,autoSubmit:false});
  assert.equal(writes.length,0); assert.equal(c.state().pending,1);
  assert(c.closeSettlement()); assert(c.state().settlement.closed);
  c.reopenSettlement(); assert(!c.state().settlement.closed);
  await c.retry(); assert.equal(writes.length,0,'refresh/retry cannot authorize a draft');
  const restored = make(); await restored.retry(); assert.equal(writes.length,0,'reload cannot authorize a draft');
  await Promise.all([c.submitSettlement(),c.submitSettlement(),c.submitSettlement()]);
  assert.equal(writes.length,1); assert.match(c.state().status,/重试/);
  assert(c.closeSettlement()); c.reopenSettlement();
  fail = false; upgraded = true; release = {};
  const a = c.submitSettlement(), b = c.submitSettlement();
  assert(c.state().submitting); assert.equal(writes.length,2);
  release.resolve(); await Promise.all([a,b]); release = null;
  assert.deepEqual(writes[1],writes[0], 'retry preserves ID and full fingerprint payload');
  assert(!Object.hasOwn(writes[0],'approved'),'local approval is never sent to server');
  assert.equal(c.state().pending,0); assert(c.state().settlement.submitted);
  assert(c.profileActive()); assert(c.closeSettlement(), 'idle form closes without losing persistent input'); c.reopenSettlement();
  await c.submitProfile('玩家','练习'); assert(!c.profileActive(), 'saved profile allows close/R');
  assert(c.closeSettlement()); c.reopenSettlement(); c.markProfileDirty();
  assert(c.profileActive(), 'editing again protects unsaved form'); assert(c.closeSettlement()); c.reopenSettlement();
  profileRelease = {}; const saving = c.submitProfile('玩家','练习'); c.markProfileDirty();
  profileRelease.resolve(); await saving; profileRelease = null;
  assert(c.profileActive(), 'edits during an in-flight save must remain protected');
  c.skipProfile(); assert(c.closeSettlement());
  await c.submitSettlement(); assert.equal(writes.length,2,'success disables repeated submission');
  c.clearSettlement();
  const draft = c.begin(meta); now+=5000;
  await c.finish(draft,{result:'loss',gameDurationMs:5000,autoSubmit:false});
  c.clearSettlement(); assert.equal(c.state().pending,1,'R keeps unapproved match');
  const reload = make(); await reload.submitPending(); assert.equal(reload.state().pending,0,'explicit pending submit recovers refresh/restart drafts');
  console.log('Settlement actions: explicit approval, close/reopen, concurrency, retry fingerprint, upgrade guard and persisted drafts passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
