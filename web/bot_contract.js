(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QQTBots = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildBotContract() {
  'use strict';

  const TACTICAL_IDENTITY_HASH = 'b77820adcfefa02bd4ab4d0efddad405aae7a6c77b3c235088f9af2f80cb4f83';

  function validateAction(action, legalMoves = [0, 1, 2, 3, 4], legalAbilities = [0, 1, 2]) {
    if (!action || !Number.isInteger(action.move) || !Number.isInteger(action.ability)) {
      throw new Error('BotAction requires integer move and ability');
    }
    if (action.move < 0 || action.move > 4 || action.ability < 0 || action.ability > 2) {
      throw new Error(`invalid Bun action: ${action.move},${action.ability}`);
    }
    if (!legalMoves.includes(action.move) || !legalAbilities.includes(action.ability)) {
      throw new Error(`illegal Bun action: ${action.move},${action.ability}`);
    }
    return action;
  }

  class BotRegistry {
    constructor(runtime) { this.runtime = runtime; this.entries = new Map(); }
    register(spec, factory) {
      if (this.entries.has(spec.id)) throw new Error(`duplicate bot id: ${spec.id}`);
      this.entries.set(spec.id, { spec, factory });
    }
    validatedConfig(spec, config = {}) {
      const properties = spec.config_schema.properties || {};
      const unknown = Object.keys(config).filter((key) => !(key in properties));
      if (unknown.length && spec.config_schema.additionalProperties === false) {
        throw new Error(`unknown config keys for ${spec.id}: ${unknown.join(',')}`);
      }
      const merged = Object.assign({}, spec.defaults, config);
      for (const [key, rule] of Object.entries(properties)) {
        if (!(key in merged) || !rule.type) continue;
        const value = merged[key];
        const valid = rule.type === 'integer' ? Number.isInteger(value)
          : rule.type === 'number' ? typeof value === 'number' && Number.isFinite(value)
            : rule.type === 'boolean' ? typeof value === 'boolean'
              : rule.type === 'string' ? typeof value === 'string' : true;
        if (!valid) throw new Error(`invalid config type for ${spec.id}.${key}`);
        if (rule.enum && !rule.enum.includes(value)) {
          throw new Error(`invalid config value for ${spec.id}.${key}`);
        }
      }
      return merged;
    }
    describe(id, config = {}) {
      const entry = this.entries.get(id);
      if (!entry) throw new Error(`unknown bot id: ${id}`);
      return { spec: entry.spec, config: this.validatedConfig(entry.spec, config),
        implementation_hash: entry.spec.implementation_hash,
        fixture_hash: entry.spec.fixture_hash };
    }
    validate(id, config = {}, requirements = {}) {
      const entry = this.entries.get(id);
      if (!entry) throw new Error(`unknown bot id: ${id}`);
      if (requirements.version && requirements.version !== entry.spec.version) {
        throw new Error(`incompatible bot version: ${requirements.version}`);
      }
      for (const [name, expected] of Object.entries(requirements.capabilities || {})) {
        if (entry.spec.capabilities[name] !== expected) throw new Error(`capability mismatch: ${name}`);
      }
      return this.validatedConfig(entry.spec, config);
    }
    create(id, config = {}, requirements = {}) {
      const entry = this.entries.get(id);
      if (!entry) throw new Error(`unknown bot id: ${id}`);
      if (!entry.spec.runtime.includes(this.runtime) || !entry.factory) {
        throw new Error(`bot ${id} does not support runtime ${this.runtime}`);
      }
      return entry.factory(this.validate(id, config, requirements));
    }
    list() { return Array.from(this.entries.values()).map((entry) => entry.spec); }
  }

  class TacticalAdapter {
    constructor(BunRuleTacticalBot, config) { this.bot = new BunRuleTacticalBot({ horizonSteps: config.horizon }); }
    reset(context) { this.bot.reset(); }
    act(observation, playerId, rng) {
      const decision = this.bot.analyze(observation.state, playerId);
      return validateAction({ move: Number(decision.action[0]), ability: Number(decision.action[1]) },
        observation.legal_moves, observation.legal_abilities);
    }
    observe_transition(event, nextObservation, playerId) {
      this.bot.observeTransition(event, nextObservation.state, playerId);
    }
    close() {}
  }

  class RandomRoamBot {
    constructor(config) { this.allowIdle = config.allow_idle; }
    reset(context) {}
    act(observation, playerId, rng) {
      let moves = observation.legal_moves.filter((move) => this.allowIdle || move !== 4);
      if (!moves.length) moves = observation.legal_moves.slice();
      const value = typeof rng === 'function' ? rng() : ((Number(rng) >>> 0) / 4294967296);
      return validateAction({ move: moves[Math.floor(value * moves.length) % moves.length], ability: 0 },
        observation.legal_moves, observation.legal_abilities);
    }
    close() {}
  }

  // 猎手 Bot 需要连续坐标/道具等完整局面，由宿主经 observation.metadata.sim 提供。
  class HunterAdapter {
    constructor(hunterApi, config) {
      this.bot = new hunterApi.BunHunterBot({ difficulty: config.difficulty });
      this.stateFromSim = hunterApi.hunterStateFromSim;
    }
    reset(context) { this.bot.reset(context && context.seed); }
    act(observation, playerId, rng) {
      const sim = observation.metadata && observation.metadata.sim;
      if (!sim) throw new Error('bun.hunter requires observation.metadata.sim');
      const decision = this.bot.analyzeSim ? this.bot.analyzeSim(sim, playerId)
        : this.bot.analyze(this.stateFromSim(sim), playerId);
      return validateAction({ move: decision.action[0], ability: decision.action[1] },
        observation.legal_moves, observation.legal_abilities);
    }
    close() {}
  }

  function createDefaultRegistry(dependencies = {}) {
    const registry = new BotRegistry('browser');
    const commonFixtureHash = '72812b6f7a0009c2df7f68e294d6d33132fa4d2e240954ec34cdc6d47585abe9';
    registry.register({
      id: 'bun.tactical_v2', version: '2.0.0', display_name: 'Bun Tactical v2',
      runtime: ['python', 'browser'], capabilities: { deterministic: true, batched: 'native_host',
        jittable: false, async: false, transition_observer: true, frozen: true },
      config_schema: { type: 'object', additionalProperties: false, properties: {
        name: { type: 'string' }, horizon: { type: 'integer' }, aggression: { type: 'number' },
        escape_margin: { type: 'integer' }, tie_break: { type: 'integer' },
        bomb_threshold: { type: 'integer' }, positioning_preference: { type: 'string' },
        structural_mode: { type: 'string' } } },
      defaults: { name: 'full_v2', horizon: 40, aggression: 1.0, escape_margin: 0,
        tie_break: 0, bomb_threshold: 6, positioning_preference: 'chase', structural_mode: 'v2' },
      identity_hash: TACTICAL_IDENTITY_HASH,
      implementation_hash: '822213708f3f6cdebfa84adab703c3fae4ad2f52aca82146045d359e1a0c4a2d',
      fixture_hash: commonFixtureHash,
      provenance_hash: '05fa7e025511e0a39f7bb2178d58208003145c8d86ba5c79193ddb223c881cdf',
    }, dependencies.BunRuleTacticalBot
      ? (config) => new TacticalAdapter(dependencies.BunRuleTacticalBot, config) : null);
    registry.register({
      id: 'bun.random_roam', version: '1.0.0', display_name: '随机漫游 Bot',
      runtime: ['python', 'browser'], capabilities: { deterministic: false, batched: 'none',
        jittable: false, async: false, transition_observer: false, frozen: true },
      config_schema: { type: 'object', additionalProperties: false,
        properties: { allow_idle: { type: 'boolean' } } }, defaults: { allow_idle: true },
      identity_hash: 'fd14534fbef9c73830902306532853683d4a9e4ba40987a4171520016e0ff486',
      implementation_hash: '84b51f314dafe1065eb1ce8ef29ac8636962a8c01ed64b25c67138a68812fab4',
      fixture_hash: commonFixtureHash,
      provenance_hash: 'f5e0883238166c13ee6a71b6393438c8e07097a68559825673e42daaa5f1496a',
    }, (config) => new RandomRoamBot(config));
    registry.register({
      id: 'bun.hunter', version: '1.0.0', display_name: '猎手 Bot（躲泡/进攻/偷包）',
      runtime: ['browser'], capabilities: { deterministic: true, batched: 'none',
        jittable: false, async: false, transition_observer: false, frozen: false },
      config_schema: { type: 'object', additionalProperties: false, properties: {
        difficulty: { type: 'string', enum: ['easy', 'normal', 'hard'] } } },
      defaults: { difficulty: 'normal' },
    }, dependencies.hunter ? (config) => new HunterAdapter(dependencies.hunter, config) : null);
    registry.register({
      id: 'bun.coop_hunter', version: '1.0.0', display_name: '协作猎手',
      runtime: ['browser'], capabilities: { deterministic: true, batched: 'none',
        jittable: false, async: false, transition_observer: false, frozen: false },
      config_schema: { type: 'object', additionalProperties: false, properties: {
        difficulty: { type: 'string', enum: ['easy', 'normal', 'hard'] } } },
      defaults: { difficulty: 'hard' },
    }, dependencies.coopHunter ? (config) => new HunterAdapter({
      BunHunterBot: dependencies.coopHunter.BunCoopHunterBot,
      hunterStateFromSim: dependencies.coopHunter.hunterStateFromSim,
    }, config) : null);
    registry.register({
      id: 'bun.browser_model', version: '1.0.0', display_name: '浏览器模型 Bot',
      runtime: ['browser'], capabilities: { deterministic: true, batched: 'none',
        jittable: false, async: true, transition_observer: false, frozen: true },
      config_schema: { type: 'object', additionalProperties: false, properties: {} }, defaults: {},
      identity_hash: '986d63a2e8571565141f741969811713454b551358ffde79e0aac35e4b2cb516',
      implementation_hash: '097f24118b0f4902580d99a4aae8817df628efde7a401b321becd0d0d0837a74',
      fixture_hash: commonFixtureHash,
      provenance_hash: '19f8803a55a9ece5ef47220ee561fd2d58c9d3271296a04d76d0fbccd4a3f62d',
    }, dependencies.modelFactory || null);
    return registry;
  }

  return { BotRegistry, createDefaultRegistry, validateAction, TACTICAL_IDENTITY_HASH };
});
