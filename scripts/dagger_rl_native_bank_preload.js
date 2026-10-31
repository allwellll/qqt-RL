'use strict';
// Diagnostic-only natural replay recorder. No action, reward or physical edits.
const fs=require('fs'),path=require('path'),v8=require('v8'),assert=require('node:assert/strict');
const Q=require('../web/sim'),bank=require('./dagger_rl_js_bank');
const {ItemMetrics}=require('./dagger_rl_metrics_v2');
const contexts=new WeakMap();let externalRng=null;
const out=process.env.DAGGER_NATIVE_BANK_OUT;
assert.ok(out,'diagnostic bank output must be explicit');
const wanted=new Set((process.env.DAGGER_NATIVE_BANK_CATEGORIES||'').split(',').filter(Boolean));
assert.ok([...wanted].every(x=>['inventory','release'].includes(x)));
const saved=new Set();

class SourceSim extends bank.StatefulSim{
  _bunUpdate(...args){
    const c=contexts.get(this);
    if(c)c.native={alive:Array.from(this.alive),trapped:Array.from(this.trapped),
      carry:Array.from(this.bunCarried),stock:this.bunStored.map(x=>Array.from(x))};
    return super._bunUpdate(...args);
  }
  step(joint){
    const c=contexts.get(this)||{};contexts.set(this,c);
    const before={tick:this.t,carry:Array.from(this.bunCarried),alive:Array.from(this.alive),
      trapped:Array.from(this.trapped),stock:this.bunStored.map(x=>Array.from(x)),pos:Array.from(this.pos)};
    c.joint=structuredClone(joint);c.native=null;
    const result=super.step(joint);
    for(let p=0;p<2;p++){
      const enemy=1-p;
      if(before.carry[p]<0&&this.bunCarried[p]===enemy){
        c.proof=c.proof||{};
        const native=c.native||before;
        const stock=native.stock[enemy][enemy]>this.bunStored[enemy][enemy];
        c.proof[p]={actual_acquisition:true,player:p,origin:enemy,tick:this.t,
          source:stock?(!native.alive[enemy]?'dead_stock':native.trapped[enemy]>0?'native_sugar_stock':'free_alive_stock'):
            before.carry[enemy]===enemy?'contact_transfer':'native_nonstock_pickup',
          pre_step:before,native_bun_stage:native,after_position:Array.from(this.pos),
          original_source_replay:true,manual_carry_edit:false};
      }
      if(this.bunCarried[p]!==enemy&&c.proof)delete c.proof[p];
    }
    return result;
  }
}
Q.Sim=SourceSim;
Q.mulberry32=seed=>{externalRng=bank.statefulRng(seed);return externalRng;};

class Recorder extends ItemMetrics{
  before(action){
    super.before(action);this._pendingBank=null;
    const c=contexts.get(this.s),p=this.p;
    const proof=c?.proof?.[p];
    if(this.phase0!=='carry'||!proof?.actual_acquisition||!this.inventory.some(x=>x?.item===1&&x.count>0))return;
    const categories=[];
    if(wanted.has('inventory')&&!saved.has('inventory'))categories.push('inventory');
    if(wanted.has('release')&&!saved.has('release')&&action[2]&&this.inventory[0]?.item===1)categories.push('release');
    if(!categories.length)return;
    // The recorder itself is outside the saved metric aux; exclude its pending slot.
    delete this._pendingBank;
    const snapshot=bank.captureInstrumented(this.s,this);
    this._pendingBank={categories,snapshot,proof:structuredClone(proof),
      before_releases:this.row('carry',1).legal_releases,action:structuredClone(action),
      policy_rng:externalRng.snapshot()};
  }
  after(){
    const pending=this._pendingBank;delete this._pendingBank;
    super.after();
    if(!pending)return;
    const realRelease=this.row('carry',1).legal_releases>pending.before_releases;
    const categories=pending.categories.filter(x=>x!=='release'||realRelease);
    if(!categories.length)return;
    const c=contexts.get(this.s),{sim:clone,metrics:copy}=bank.restoreInstrumented(pending.snapshot);
    clone.step(c.joint);copy.after();
    const original=bank.captureInstrumented(this.s,this),recovered=bank.captureInstrumented(clone,copy);
    assert.deepEqual(recovered.fields,original.fields,'real carrier complete-state next-step parity');
    assert.deepEqual(recovered.rng,original.rng,'real carrier RNG next-step parity');
    assert.deepEqual(recovered.item_metrics,original.item_metrics,'real carrier item-aux next-step parity');
    fs.mkdirSync(out,{recursive:true});
    for(const category of categories){
      const file=path.join(out,`${category}.bank`),record={schema:'dagger.native-causal-carrier/v1',
        category,seed:this.s.seed,player:this.p,tick:pending.snapshot.fields.t,
        snapshot:pending.snapshot,causal_proof:pending.proof,
        observed_student_action:pending.action,observed_joint_action:c.joint,
        policy_rng_after_decisions:pending.policy_rng,next_step_parity:true,
        next:original,source:'actual production natural replay; no synthetic carrier',
        policy_protocol:{greedy:true,inferEvery:1,recurrent:false}};
      fs.writeFileSync(file+'.tmp',v8.serialize(record));fs.renameSync(file+'.tmp',file);
      saved.add(category);
    }
  }
}
// Keep recorder-private properties out of the external metric state contract.
// They are removed before every capture; ordinary ItemMetrics remains unchanged.
const target=require.resolve('./dagger_rl_metrics');require(target);
require.cache[target].exports={ItemMetrics:Recorder};
