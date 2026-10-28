'use strict';
// Evaluation-only passive adapter; no action/model/physics or RNG change.
const target=require.resolve('./dagger_rl_metrics');
require(target);
require.cache[target].exports=require('./dagger_rl_metrics_v2');
