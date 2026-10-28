(function (root) {
  'use strict';
  const names = ['COMBAT_CONTROL', 'STOCK_WINDOW', 'DROP_RECOVERY', 'CARRY_RETURN', 'CAGE_DELIVERY'];
  function features(obs, state) {
    if (obs.length !== 24 * 195 || state.length !== 24) throw Error('phase5 requires native Bun24 inputs');
    let nearMass = 0, drop = false;
    for (let r = 0; r < 13; r++) for (let c = 0; c < 15; c++) {
      const i = r * 15 + c;
      const home = (rr, cc) => rr >= 0 && rr < 13 && cc >= 0 && cc < 15 && obs[14 * 195 + rr * 15 + cc] > 0;
      if (home(r, c) || home(r - 1, c) || home(r + 1, c) || home(r, c - 1) || home(r, c + 1)) nearMass += obs[i];
      drop ||= obs[17 * 195 + i] > 0;
    }
    const cargo = state[9] > 0 && state[14] > 0;
    const freeEmpty = state[9] === 1 && state[13] === 0;
    const stock = freeEmpty && (state[10] <= 0 || state[10] > 1) && state[20] > 0;
    const phase = cargo ? (nearMass >= .5 ? 4 : 3) : freeEmpty && drop ? 2 : stock ? 1 : 0;
    return names.map((_, i) => Number(i === phase));
  }
  function install(QQT) {
    const proto = QQT.TransformerModel.prototype;
    if (proto._phase5Installed) return;
    const original = proto._run;
    proto._run = function (n, observations, states) {
      const size = this.T('state_w').length;
      if (size === 24 * this.meta.embed) return original.call(this, n, observations, states);
      if (size !== 29 * this.meta.embed) throw Error('unsupported phase5 state projection');
      const encoded = states.map((state, i) => Float64Array.from([...state, ...features(observations[i], state)]));
      return original.call(this, n, observations, encoded);
    };
    proto._phase5Installed = true;
  }
  const api = {features, install, names, encoding: 'phase5-visible-v1'};
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
    install(require('./sim.js'));
  } else {
    root.QQTPhaseConditioning = api;
    if (root.QQT) install(root.QQT);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
