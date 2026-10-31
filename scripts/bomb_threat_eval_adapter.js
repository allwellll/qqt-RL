'use strict';
// STEER_20261029_0120 authorizes disabling only the evaluator's game0
// diagnostic Review export. No Player validation or production rule changes.
function withoutDiagnosticReview(source){
 const anchor='        {\n          const level=authority.loadLevel(),modelDoc=JSON.parse(fs.readFileSync(modelFile));';
 if(source.split(anchor).length!==2)throw Error('diagnostic Review anchor changed');
 return source.replace(anchor,anchor.replace('        {','        if(false){'));
}
module.exports={withoutDiagnosticReview};
