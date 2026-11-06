'use strict';
// Independent full-action wrapper; the complete-v4 policy and P8 witness stay composed.
const P8 = require('./interceptor_complete_v7_p8');
const VERSION = 'bun-interceptor-complete-v8-p8-full-action';
const ACTION_SCHEMA = Object.freeze({
  version: VERSION,
  request: ['move', 'bomb'],
  itemSlotRequest: 'none',
  itemTarget: 'cell-or-minus-one',
  memory: ['noProgress'],
  nativeItemEffects: 'observed-from-simulator-state',
  unsupportedState: 'baseline-action-and-memory-preserved'
});

class BunInterceptorBot extends P8.BunInterceptorBot {
  analyze(state, pid) {
    const decision = super.analyze(state, pid);
    return {...decision, action_schema: VERSION, item_slot: 0,
      item_target: this.itemTarget};
  }
}

module.exports = {...P8, VERSION, ACTION_SCHEMA, BunInterceptorBot};
