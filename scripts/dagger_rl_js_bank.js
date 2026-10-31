'use strict';
// Diagnostic-only full native state/RNG clones. No production or training hooks.
const Q=require('../web/sim');
const assert=require('node:assert/strict');
const SCHEMA='qqt.production-js-state-bank/v1';
function statefulRng(seed){
  let word=seed>>>0,draws=0;
  const rng=()=>{
    word=(word+0x6D2B79F5)|0;draws++;
    let t=Math.imul(word^(word>>>15),1|word);
    t=(t+Math.imul(t^(t>>>7),61|t))^t;
    return ((t^(t>>>14))>>>0)/4294967296;
  };
  rng.snapshot=()=>({word:word>>>0,draws});
  rng.restore=s=>{
    assert.ok(Number.isInteger(s.word)&&s.word>=0&&s.word<=0xffffffff,'RNG word');
    assert.ok(Number.isSafeInteger(s.draws)&&s.draws>=0,'RNG draws');
    word=s.word>>>0;draws=s.draws;
  };
  return rng;
}

class StatefulSim extends Q.Sim {
  // The parent constructor assignment reaches this setter before its reset.
  // Consequently every production RNG draw, including construction, is captured.
  set rng(next){
    if(!this._bankRng){this._bankRng=statefulRng(this.seed);return;}
    // Native Review Player wraps the existing stream to count draws. Preserve
    // that stream and its state accessors instead of reseeding at assignment.
    const previous=this._bankRng;
    assert.equal(typeof next,'function','RNG observer must be callable');
    next.snapshot=()=>previous.snapshot();next.restore=state=>previous.restore(state);
    this._bankRng=next;
  }
  get rng(){return this._bankRng;}
}

function capture(sim,metrics=null){
  assert.ok(sim instanceof StatefulSim,'stateful simulator required');
  if(metrics){
    const {ItemMetrics}=require('./dagger_rl_metrics_v2');
    assert.ok(metrics instanceof ItemMetrics&&metrics.s===sim,'known passive item observer required');
  }
  const fields={},excluded=[];
  for(const [key,value]of Object.entries(sim)){
    if(key==='_bankRng')continue;
    if(typeof value==='function'){
      // The route evaluator wraps this method solely to observe its pre-state.
      assert.ok(key==='_bunUpdate'||(metrics&&['_placeHeldItem','_contactFieldItem'].includes(key)),
        'unrecognized function in full state bank');
      excluded.push(key);continue;
    }
    fields[key]=value;
  }
  return {schema:SCHEMA,fields:structuredClone(fields),rng:sim.rng.snapshot(),
    excluded_instrumentation:excluded};
}

function captureInstrumented(sim,metrics){
  const record=capture(sim,metrics),aux={};
  for(const [key,value]of Object.entries(metrics)){
    if(key==='s')continue;
    assert.notEqual(typeof value,'function','unknown observer callback');
    aux[key]=value;
  }
  record.item_metrics=structuredClone(aux);
  return record;
}

function restoreInstrumented(record){
  assert.ok(record.item_metrics,'external item ownership/slide state required');
  const sim=restore(record);
  const {ItemMetrics}=require('./dagger_rl_metrics_v2');
  const metrics=new ItemMetrics(sim,record.item_metrics.p);
  Object.assign(metrics,structuredClone(record.item_metrics));
  return {sim,metrics};
}

function restore(record){
  assert.equal(record.schema,SCHEMA,'bank schema');
  assert.ok(record.fields&&record.rng,'complete state/RNG required');
  const sim=Object.create(StatefulSim.prototype);
  Object.assign(sim,structuredClone(record.fields));
  sim._bankRng=statefulRng(sim.seed);sim.rng.restore(record.rng);
  return sim;
}

module.exports={statefulRng,StatefulSim,capture,restore,captureInstrumented,restoreInstrumented,SCHEMA};
