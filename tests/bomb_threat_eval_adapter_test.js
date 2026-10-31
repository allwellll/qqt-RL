'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const {withoutDiagnosticReview}=require('../scripts/bomb_threat_eval_adapter');
const source=fs.readFileSync('scripts/dagger_rl_route.js','utf8');
const result=withoutDiagnosticReview(source);
assert(result.includes('        if(false){\n          const level=authority.loadLevel()'));
assert.equal(result.replace('        if(false){\n          const level=authority.loadLevel()',
 '        {\n          const level=authority.loadLevel()'),source);
assert(result.includes('P.validate(replay);await new P.Player(replay,level).verify();'));
assert.throws(()=>withoutDiagnosticReview('missing anchor'),/anchor changed/);
console.log('evaluator-only diagnostic export bypass; physics/actions/official verification unchanged GREEN');
