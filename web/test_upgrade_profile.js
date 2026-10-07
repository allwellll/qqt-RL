'use strict';
const assert = require('assert');
const { webcrypto } = require('crypto');
const LB = require('./leaderboard');
const QQT = require('./sim');
const Sound = require('./sound');
(async () => {
  let serverMessage='旧宣言', now=Date.now(), hold, writes=0;
  const map=new Map(), storage={getItem:k=>map.get(k),setItem:(k,v)=>map.set(k,v)};
  const config={url:'https://db',publishableKey:'public'};
  const make=()=>LB.createClient({config,storage,crypto:webcrypto,now:()=>now,fetch:async(url,options)=>{
    const p=JSON.parse(options.body);
    if(url.endsWith('qqt_leaderboard')) return {ok:true,json:async()=>[]};
    if(url.endsWith('qqt_get_profile')) return {ok:true,json:async()=>({profile_contract_version:2,registered:true,nickname:'缓存昵称',victory_message:serverMessage})};
    if(url.endsWith('qqt_update_profile')) {writes++;if(hold)await hold;serverMessage=p.p_victory_message||serverMessage;
      return {ok:true,json:async()=>({saved:true,profile_contract_version:2,client_match_id:p.p_client_match_id,nickname:'缓存昵称',victory_message:serverMessage,superseded:false})};}
    return {ok:true,json:async()=>({level:1,points:3,games:1,wins:1,match_upgraded:false,client_match_id:p.p_payload.client_match_id})};
  }});
  const client=make();await client.syncProfile();
  const meta={opponent:'bun.coop_hunter',difficulty:'hard',seed:7,mode:'1v2',map_id:'training806',client_version:'dev'};
  const match=client.begin(meta);now+=5000;await client.finish(match,{result:'win',gameDurationMs:5000});
  assert.equal(client.state().settlement.upgraded,false);
  let release;hold=new Promise(r=>release=r);
  const first=client.submitProfile('篡改昵称','普通局宣言'),duplicate=client.submitProfile('另一个昵称','重复');
  assert.equal(first,duplicate);assert.equal(writes,1);assert.equal(client.state().profileSubmitting,true);release();await first;hold=null;
  assert.equal(client.state().savedNickname,true);assert.equal(make().state().nickname,'缓存昵称');assert.equal(make().state().victory_message,'普通局宣言');
  assert.equal(client.state().settlement.profileReceiptVersion,2);

  // Actual deterministic Sim release outcome, including slot-count >1 and failed placement.
  const sim=new QQT.Sim(7); sim.reset('open',{nativeItems:true,nativeTrap:true});
  sim.wall.fill(0); sim.brick.fill(0); sim.crate.fill(0); sim.fuse.fill(0);
  sim.pos[0]=5.5;sim.pos[1]=5.5;sim.pos[2]=1.5;sim.pos[3]=1.5;
  sim._addItemToSlots(0,1,2);
  const idle=[QQT.MOVE_IDLE,0,0,1];
  const before=Sound.snapshot(sim); const info=sim.step([[QQT.MOVE_IDLE,0,1,1],idle]);
  assert.equal(info.itemReleased[0],true); assert.equal(sim.itemSlots[0][0].count,1);
  assert.equal(Sound.detectEvents(before,sim,info,0).filter(x=>x==='pickup').length,1);
  assert(!Sound.detectEvents(before,sim,info,1).includes('pickup'));
  assert(!Sound.detectEvents(before,sim,sim.step([[QQT.MOVE_IDLE,0,1,1],idle]),0).includes('pickup'),'blocked second release is silent');
  assert(!Sound.detectEvents(before,sim,info,-1).includes('pickup'));
  let started=0,gain;
  class Context {constructor(){this.state='running';this.destination={};} async decodeAudioData(){return {};} createBufferSource(){return {connect(){return this;},start(){started++;}};} createGain(){gain={gain:{},connect(){return this;}};return gain;} }
  const audio=Sound.createPlayer({AudioContext:Context,fetch:async()=>({arrayBuffer:async()=>new ArrayBuffer(1)})});
  await audio.load();audio.setEnabled(false);audio.play('pickup');assert.equal(started,0);
  audio.setEnabled(true);audio.play('pickup');assert.equal(started,1);assert.equal(gain.gain.value,.6);
  console.log('Round4 every-match profile receipt plus successful item-release reuse/mute passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
