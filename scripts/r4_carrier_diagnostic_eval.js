'use strict';
const Q=require('../web/sim-bun-native-v2');
require('./r4_combat_metrics').install(Q);
require('./r4_carrier_metrics').install(Q);
require('./r3_items_candidate_eval');
