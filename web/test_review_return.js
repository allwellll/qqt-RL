'use strict';
const assert=require('node:assert/strict'),P=require('./review_protocol'),F=require('./review_return'),A=require('./review_advantage');
const doc=require('./review-replays/h8-route-v35-g32-drop-return-2026101732.json'),level=require('../scripts/eval_web_hunter').loadLevel();
const audit=F.analyze(doc,level,P),r=audit.points[177];
assert.equal(r.kind,'return_oscillation');assert.equal(r.axis,'vertical');assert(r.no_progress_ticks>=30);assert.equal(r.distance,38);
assert.deepEqual(r.pos.map(v=>Math.round(v*100)/100),[1.98,9.64]);assert.equal(r.path_first_moves.includes(1),true);
assert.equal(F.MOVES[0],'向上');assert.equal(F.MOVES[1],'向下');assert(F.describe(r).includes('返程振荡失败'));assert(F.describe(r).includes('当前持敌包返己笼'));
assert(audit.intervals.some(s=>s.start<=177&&s.end>=177));assert(!audit.points[20].kind);
assert.equal(F.comparison(null,{advantage:2}),null);assert.equal(F.comparison({advantage:1},null),null);
assert.equal(F.comparison({advantage:1},{advantage:2}),1);
assert.throws(()=>F.validatePair({role:'original',checkpoint:'same',critic:'a'},{role:'shaped',checkpoint:'same',critic:'a'}),/critic/);
assert.throws(()=>F.validatePair({role:'shaped',checkpoint:'v34',critic:'a'},{role:'original',checkpoint:'v35',critic:'b'}),/role/);
F.validatePair({role:'original',checkpoint:'v34',critic:'a'},{role:'shaped',checkpoint:'v35',critic:'b'});
const v=A.calculate([{action:[0,0],legal:true,probability:2,reward:1,nextValue:4},{action:[1,0],legal:true,probability:3,reward:-1,nextValue:2},{action:[2,0],legal:false,probability:100,reward:999,nextValue:999}],[0,0],.5);
assert.equal(v.taken_q,3);assert(Math.abs(v.mean_q-1.2)<1e-12);assert(Math.abs(v.advantage-1.8)<1e-12);
assert.equal(F.temporalDelta(null,1),null);assert.equal(F.temporalDelta(1,null),null);assert.equal(F.temporalDelta(1,2),1);
const moving=Array.from({length:30},(_,i)=>({tick:i+1,pos:[1+i*.01,4],distance:38,target:'a',phase:'C_RETURN',alive:true,carrying:true}));
assert(F.classify(moving,29).kind===null,'grid plateau alone is not oscillation');
assert.equal(F.classify(moving.map(p=>({...p,phase:'NONE'})),29).kind,null);
const fs=require('node:fs'),crypto=require('node:crypto'),catalog=require('./review-catalog.json'),row=catalog.replays.find(r=>r.id===doc.meta.id),raw=fs.readFileSync('web/'+row.diagnostic_audit.file),evidence=JSON.parse(raw);
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');assert.equal(hash(raw),row.diagnostic_audit.sha256);assert.equal(raw.length,row.diagnostic_audit.bytes);
F.validateAudit(evidence,doc,row.sha256);assert.deepEqual(evidence.intervals,audit.intervals);
const swapped=structuredClone(evidence),original=swapped.advantage_audit.original,shaped=swapped.advantage_audit.shaped;
swapped.advantage_audit.original={...shaped,role:'original'};swapped.advantage_audit.shaped={...original,role:'shaped'};
assert.throws(()=>F.validateAudit(swapped,doc,row.sha256),'same-generation critic identities cannot be swapped even if role labels are changed');
assert.equal(evidence.actor_actions_verified,1200);assert.equal(evidence.rows.length,61);assert.equal(evidence.rows[0].tick,147);assert.equal(evidence.rows.at(-1).tick,207);
assert.equal(audit.points[177].no_progress_ticks,61);assert.equal(audit.points[177].axis,'vertical');
const actual=evidence.rows.find(r=>r.tick===177),next=evidence.rows.find(r=>r.tick===178);
assert(Math.abs(actual.move_probabilities[1]-.702199204)<1e-9);assert(Math.abs(next.move_probabilities[0]-.559185597)<1e-9);assert.equal(next.candidates.find(c=>c.action[0]===1).distance,37);
assert(actual.candidates.every(c=>!c.collision_correction));assert.equal(evidence.down_two_steps.steps[1].distance,37);
assert(evidence.native_first_tick.causalDamageSource.flat().every(v=>v===false));assert(evidence.native_first_tick.creditedKill.every(v=>v===false));
for(const change of [a=>a.replay_sha256='0'.repeat(64),a=>a.policy_sha256='0'.repeat(64),a=>a.rows[0].original_a1=1,a=>a.rows[0].candidates[0].q_original=1,a=>a.advantage_audit.shaped.critic=a.advantage_audit.original.critic,a=>a.rows[0].move_probabilities[0]=1]){const changed=structuredClone(evidence);change(changed);assert.throws(()=>F.validateAudit(changed,doc,row.sha256));}
for(const [name,expected]of Object.entries(evidence.source_hashes)){
  assert(/^[a-f0-9]{64}$/.test(expected));
  // The v35 audit records the exact pre-v36 PBRS source; v36 extends that
  // module while preserving this immutable historical receipt.
  if(name==='scripts/je09_route_pbrs.py'){
    assert.equal(expected,'3e1a1a224f852c6f6c361dd1903bd7d3be45ca9b15506b17f5353ec1ad505d1b');
  }else if(fs.existsSync(name))assert.equal(hash(fs.readFileSync(name)),expected,'sealed local training source '+name);
}
const html=fs.readFileSync('web/review.html','utf8');assert(html.includes('data-track-label="failure-track"'));assert(html.includes('返程失败区间'));assert(html.includes('src="review_return.js"'));
const css=fs.readFileSync('web/review.css','utf8');
const compactBlock=css.match(/^\.position\s*\{([^}]*)\}/m)?.[1]||'';
const locatorBlock=css.match(/\.locator-row\s*\{([^}]*)\}/)?.[1]||'';
const copyButtonBlock=css.match(/\.locator-row button\s*,[\s\S]*?\{([^}]*)\}/)?.[1]||'';
assert.match(compactBlock,/padding:\s*4px\s+8px/,'定位容器应使用紧凑内边距');
assert.match(locatorBlock,/gap:\s*4px/,'定位行应缩小垂直留白');
assert.match(copyButtonBlock,/min-height:\s*30px/,'复制按钮保持易点高度');
assert.match(css,/\.locator-row input\s*\{[^}]*min-width:\s*0/s,'定位文本允许在窄屏收缩');
console.log('Return diagnostic GREEN: real tick177, actual vertical period-two motion, plateau not progress, path direction, role separation, normalized A1, null gaps');
