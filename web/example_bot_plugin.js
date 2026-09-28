(function (root) {
  'use strict';

  class IdleBot {
    reset(context) { this.episodeId = context.episode_id; }
    act(observation, playerId, rng) {
      return root.QQTBots.validateAction(
        { move: 4, ability: 0 }, observation.legal_moves, observation.legal_abilities);
    }
    close() {}
  }

  root.registerExampleIdleBot = function registerExampleIdleBot(registry) {
    registry.register({
      id: 'example.idle', version: '1.0.0', display_name: '示例原地等待 Bot',
      runtime: ['browser'], capabilities: {
        deterministic: true, batched: 'none', jittable: false,
        async: false, transition_observer: false, frozen: true,
      },
      config_schema: { type: 'object', properties: {}, additionalProperties: false },
      defaults: {},
      identity_hash: '720e9ee111bd7461b416f15931c218c81a58705492825975dd8017fa148438dd',
      implementation_hash: 'ceea9d17ffb16ea402ba2b21d6c3562181fcaca3b748731b6d99e76596ef2cba',
      fixture_hash: '0'.repeat(64),
      provenance_hash: '8488658105f42a4a88ef265ccba9065ee48f123113ae47dca5dbbd622e8c2478',
    }, () => new IdleBot());
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
