'use strict';
const assert=require('node:assert/strict');
const {build}=require('../scripts/expert_validation_js_eval');
for(const mode of ['categorical','greedy']){
  const evaluator=build(mode);
  assert.equal(typeof evaluator.run,'function');
  assert.equal(typeof evaluator.objective,'function');
}
console.log('expert validation JS adapters compile for categorical and greedy modes');
