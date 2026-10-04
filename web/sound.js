'use strict';

(function initSound(root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.QQTSound = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function soundFactory() {
  const FILES = {
    place: 'assets/snd/放炮.wav',
    boom: 'assets/snd/爆炸.wav',
    pickup: 'assets/snd/吃道具音效.wav',
    pop: 'assets/native/syrup_pop.wav',
  };
  const VOLUME = { place: 0.6, boom: 0.5, pickup: 0.6, pop: 0.7 };

  function snapshot(sim) {
    return {
      crate: Uint8Array.from(sim.crate),
      bunCarried: sim.bunCarried ? sim.bunCarried.slice() : [-1, -1],
      trapped: sim.trapped ? sim.trapped.slice() : [],
    };
  }

  // 拾取只对监听者(listenerPid)播放；放泡和爆炸全场可闻。
  function detectEvents(before, sim, info, listenerPid = 0) {
    const events = [];
    if (info && info.placed && info.placed.some(Boolean)) events.push('place');
    if (info && info.covered && info.covered.some((value) => value > 0)) events.push('boom');
    if (before && before.trapped && before.trapped.some((ticks, pid) => ticks > 0 &&
      ((info && info.died && info.died[pid]) || !sim.alive[pid]))) events.push('pop');
    const p = listenerPid;
    if (before && p >= 0 && sim.alive[p]) {
      const [row, column] = sim.centerCell(p);
      const cell = row * 15 + column;
      const blasted = info && info.covered && info.covered[cell] > 0;
      const tookCrate = before.crate[cell] && !sim.crate[cell] && !blasted;
      const tookBun = before.bunCarried[p] < 0 && sim.bunCarried && sim.bunCarried[p] >= 0;
      if (tookCrate || tookBun) events.push('pickup');
    }
    return events;
  }

  function createPlayer(options = {}) {
    const Ctor = options.AudioContext || (typeof AudioContext !== 'undefined' ? AudioContext : null);
    const fetchFn = options.fetch || (typeof fetch !== 'undefined' ? fetch : null);
    let context = null;
    const buffers = {};
    let enabled = true;

    async function load() {
      if (!Ctor || !fetchFn) return;
      try {
        context = new Ctor();
        await Promise.all(Object.entries(FILES).map(async ([name, file]) => {
          const data = await (await fetchFn(file)).arrayBuffer();
          buffers[name] = await context.decodeAudioData(data);
        }));
      } catch (_error) {
        // 音效加载失败不影响对局，静默降级。
      }
    }

    // 浏览器自动播放策略：首次用户交互后才能出声。
    function unlock() {
      if (context && context.state === 'suspended') context.resume();
    }

    function play(name) {
      if (!enabled || !context || !buffers[name]) return;
      const source = context.createBufferSource();
      source.buffer = buffers[name];
      const gain = context.createGain();
      gain.gain.value = VOLUME[name] == null ? 0.6 : VOLUME[name];
      source.connect(gain).connect(context.destination);
      source.start();
    }

    return {
      load, unlock, play,
      setEnabled(value) { enabled = !!value; },
      get enabled() { return enabled; },
    };
  }

  return { FILES, snapshot, detectEvents, createPlayer };
});
