'use strict';
const assert=require('node:assert/strict'),fs=require('fs');
const catalog=require('./review-catalog.json');
// Run against the current public manifest before implementation: incomplete
// real recordings must fail, regardless of whether their UI can render nulls.
const failures=[];
for(const row of [...catalog.replays,...catalog.history_replays||[]]){
  const d=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));
  if(!d.meta.diagnostic_admission||d.meta.diagnostic_admission.status!=='qualified')failures.push(row.id);
}
assert.deepEqual(failures,[],'RED: public diagnostic manifest includes recordings without per-decision reward and original/shaped A1 evidence');
const G=require('./review_admission');
assert.equal(G.SCHEMA,'qqt.review-admission/v1');
assert.equal(G.catalogGate(catalog),catalog.replays.length);
assert(catalog.replays.length===7,'replacement publication must retain multiple complete fixed-seed episodes');
assert.deepEqual(catalog.replays.map(r=>r.seed).sort(),[2026101800,2026101810,2026101811,2026101812,2026101813,2026101814,2026101815]);
assert.equal(catalog.metrics_admission.original_trajectory_recomputed,1);assert.equal(catalog.metrics_admission.best_model_replacement,6);
const historical=require('../reports/review_catalog_before_gate_20261018.json');
for(const row of [...historical.replays,...historical.history_replays||[]])assert.throws(()=>G.validate(JSON.parse(fs.readFileSync('web/review-replays/'+row.file))),/diagnostic admission/);
assert.throws(()=>G.catalogGate({replays:[],history_replays:[]}),/empty/);
for(const row of catalog.replays){const d=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));assert.equal(d.metrics.raw_a1_increment[0],null);for(const k of require('./review_metrics').FIELDS)assert.equal(d.metrics[k].length,d.actions.length);assert(!Object.hasOwn(d,'diagnostic_ticks'),'no full branch tables in public data');}
console.log('Review admission GREEN: complete real Critic/reward, all seeds, old incomplete recordings rejected');
// Metrics must be bound to replay identity, even when a forgery preserves the
// local Q/A equations and changes every duplicate column consistently.
(async()=>{const P=require('./review_protocol'),row=catalog.replays[0],d=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));assert.equal(await P.hash(P.identity(d)),d.meta.replay_identity);d.metrics.raw_q1_taken[0]++;assert.notEqual(await P.hash(P.identity(d)),d.meta.replay_identity);console.log('Metrics identity tamper RED-GREEN');})().catch(e=>{console.error(e);process.exitCode=1;});
