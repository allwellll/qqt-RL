'use strict';
// Apply the existing passive R1 hooks to the frozen production fresh adapter.
const Module = require('node:module');
const load = Module._load;
Module._load = function(request, parent, isMain) {
  if (request === './bomb_threat_eval' && parent.filename.endsWith('/bomb_threat_r1_progress_eval.js')) {
    Module._load = load;
    return require('./bomb_threat_fresh256_eval');
  }
  return load.call(this, request, parent, isMain);
};
require('./bomb_threat_r1_progress_eval');
