'use strict';
// The passive preload imports the original CLI as a module. Invoke its exact
// exported run explicitly, rather than relying on require.main after caching.
const Q = require('../web/sim.js');
const originalStep = Q.Sim.prototype.step;
Q.Sim.prototype.step = function (actions) {
  this._hLearnerSeat = 0; // Frozen authority always evaluates learner seat0.
  return originalStep.call(this, actions);
};
require('./je05_web_hunter.js').run(require('./eval_web_hunter.js').parseArgs(process.argv.slice(2)))
  .catch(error => { console.error(error); process.exitCode = 1; });
