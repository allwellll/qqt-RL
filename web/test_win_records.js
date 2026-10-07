'use strict';
const assert=require('assert'),{webcrypto}=require('crypto'),LB=require('./leaderboard');
const meta={opponent:'bun.coop_hunter',difficulty:'hard',seed:7,mode:'1v2',map_id:'training806',client_version:'dev'};
const memory=()=>{const map=new Map();return{getItem:k=>map.get(k)||null,setItem:(k,v)=>map.set(k,v)}};
const mode=process.argv[2]||'all';
(async()=>{
  if(mode==='all'||mode==='nickname'){
    const storage=memory();let randomCalls=0,now=Date.now();
    const crypto={randomUUID:()=>webcrypto.randomUUID(),getRandomValues:array=>{randomCalls++;array.fill(2);return array;}};
    const make=()=>LB.createClient({storage,crypto,config:{},now:()=>now,fetch:async()=>({ok:true,json:async()=>({profile_contract_version:2,registered:false})})});
    let c=make();assert.match(c.state().nickname,/^QQT玩家[A-HJKMNP-Z2-9]{3}$/,'RED: safe editable three-character default');
    const initial=c.state().nickname,count=randomCalls;c=make();assert.equal(c.state().nickname,initial);assert.equal(randomCalls,count);
    await c.syncProfile();let m=c.begin(meta);now+=5000;await c.finish(m,{result:'win',gameDurationMs:5000,autoSubmit:false});
    assert.equal(c.state().settlement.draft.nickname,initial);
    c.setDraft('QQT玩家','感言');c=make();assert.equal(c.state().settlement.draft.nickname,'QQT玩家');
    c.setDraft('','');c=make();assert.equal(c.state().settlement.draft.nickname,'','deletion survives reload');
    await c.syncProfile();assert.equal(await c.submitCard(),false,'blank input is not silently replaced by the default');
    c.clearSettlement();m=c.begin(meta);now+=5000;await c.finish(m,{result:'draw',gameDurationMs:5000,autoSubmit:false});
    assert.equal(c.state().settlement.draft.nickname,'','new match cannot restore suffix after editing');
  }
  if(mode==='all'||mode==='rows'){
    const calls=[],storage=memory();let hold,release,version=0;
    const row=(rank,isMine=false,isLatest=false)=>({rank,record_id:webcrypto.randomUUID(),nickname:'同名',game_duration_ms:1000+rank,
      victory_message:rank===1?'<img src=x>':`第${rank}局`,player_ip:'12.*.*.34',is_mine:isMine,is_latest:isLatest});
    let rows=[row(1,true),row(2,false),row(3,true,true)];
    assert.throws(()=>LB.winRows({leaderboard_contract_version:1,rows:[row(37,true,true)]}),/格式/,'榜外行须跟在完整Top20之后');
    for(const bad of [{player_ip:'1.2.3.4'},{player_id:webcrypto.randomUUID()},{is_latest:true,is_mine:false}])
      assert.throws(()=>LB.winRows({leaderboard_contract_version:1,rows:[{...row(1),...bad}]}),/格式/);
    const c=LB.createClient({storage,crypto:webcrypto,config:{url:'https://mock',leaderboardRpc:'qqt_my_win_leaderboard'},fetch:async(url,opts)=>{
      const body=JSON.parse(opts.body);calls.push({url,body});const response={leaderboard_contract_version:1,rows:rows.map(r=>({...r}))};
      if(hold){hold=false;await new Promise(r=>release=r);}return{ok:true,json:async()=>response};}});
    await c.refresh();assert.equal(c.state().rows.length,3,'RED: consume authenticated per-win contract');
    assert(calls.every(x=>x.url.endsWith('qqt_my_win_leaderboard')&&x.body.p_player_secret));
    const node=()=>({items:[],attributes:{},textContent:'',className:'',append(x){this.items.push(x)},setAttribute(k,v){this.attributes[k]=v}});
    const doc={createElement:node},target={items:[],replaceChildren(){this.items=[]},append(x){this.items.push(x)}};
    LB.renderRows(doc,target,c.state().rows);assert(target.items[0].className.includes('is-mine'));
    assert(!target.items[1].className.includes('is-mine'),'same nickname never highlights peer');
    assert(target.items[2].className.includes('is-latest'));assert.equal(target.items[0].items[3].textContent,'<img src=x>');
    assert(target.items[2].items[1].items.some(x=>x.textContent==='最新'));
    rows=[...Array.from({length:20},(_,i)=>row(i+1)),row(37,true,true)];await c.refresh();LB.renderRows(doc,target,c.state().rows);
    assert.equal(target.items.length,22);assert.equal(target.items[20].className,'leaderboard-ellipsis');
    assert.equal(target.items[20].items.length,1);assert.equal(target.items[20].items[0].attributes.colspan,'5');
    assert.equal(target.items[21].items[0].textContent,'37');
    hold=true;const stale=c.refresh();await Promise.resolve();rows=[row(1,true,true)];await c.refresh();release();await stale;
    assert.equal(c.state().rows.length,1,'late old read cannot roll back newest highlight');
    const json=JSON.parse(storage.getItem('qqt.leaderboard.v1'));assert(!JSON.stringify(json).includes('12.*.*.34'),'public rows are not identity storage');
    version++;
  }
  console.log(`Win-record ${mode}: editable defaults, private markers, global rank/ellipsis, text rendering and stale reads passed`);
})().catch(e=>{console.error(e);process.exitCode=1;});
