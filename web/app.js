'use strict';

(async function startBunArena() {
  const canvas = document.getElementById('game');
  const context = canvas.getContext('2d');
  const status = document.getElementById('status');
  const restart = document.getElementById('restart');
  const opponentSelect = document.getElementById('opponent');
  const modelFile = document.getElementById('model-file');
  const modelStatus = document.getElementById('model-status');
  const levelList = await fetch('assets/maps/levels.json').then((response) => response.json());
  const level = levelList.find((item) => item.qqt_id === 806);
  if (!level) throw new Error('Bun06 level missing');

  const bot = new BunRuleTacticalBot();
  const held = new Set();
  const modelRng = QQT.mulberry32(0x515154);
  let bombQueued = false;
  let loadedModel = null;
  let sim;

  function reset() {
    sim = new QQT.Sim(Date.now() >>> 0);
    sim.reset(level);
    bot.reset();
    bombQueued = false;
  }

  function humanAction() {
    let move = QQT.MOVE_IDLE;
    if (held.has('KeyW')) move = QQT.MOVE_UP;
    else if (held.has('KeyS')) move = QQT.MOVE_DOWN;
    else if (held.has('KeyA')) move = QQT.MOVE_LEFT;
    else if (held.has('KeyD')) move = QQT.MOVE_RIGHT;
    const action = [move, bombQueued ? 1 : 0, 0, 0];
    bombQueued = false;
    return action;
  }

  function drawCell(row, column, color, inset = 0) {
    const width = canvas.width / QQT.W;
    const height = canvas.height / QQT.H;
    context.fillStyle = color;
    context.fillRect(column * width + inset, row * height + inset,
      width - inset * 2, height - inset * 2);
  }

  function render() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (let row = 0; row < QQT.H; row++) {
      for (let column = 0; column < QQT.W; column++) {
        const index = row * QQT.W + column;
        drawCell(row, column, (row + column) % 2 ? '#18343c' : '#1b3a43');
        if (sim.wall[index]) drawCell(row, column, '#546a70', 3);
        else if (sim.brick[index]) drawCell(row, column, '#a55a37', 5);
        if (sim.bunLoose[index * 2] || sim.bunLoose[index * 2 + 1]) {
          drawCell(row, column, '#ffd06a', 17);
        }
        if (sim.fuse[index] > 0) {
          context.fillStyle = '#101010';
          context.beginPath();
          context.arc((column + .5) * canvas.width / QQT.W,
            (row + .5) * canvas.height / QQT.H, 14, 0, Math.PI * 2);
          context.fill();
        }
        if (sim.blastLinger[index] > 0) drawCell(row, column, '#ffb347aa', 2);
      }
    }
    for (let player = 0; player < 2; player++) {
      if (!sim.alive[player]) continue;
      const y = sim.pos[player * 2] * canvas.height / QQT.H;
      const x = sim.pos[player * 2 + 1] * canvas.width / QQT.W;
      context.fillStyle = player === 0 ? '#54a8ff' : '#ff5f6d';
      context.beginPath();
      context.arc(x, y, 18, 0, Math.PI * 2);
      context.fill();
      if (sim.bunCarried[player] >= 0) {
        context.fillStyle = '#ffd06a';
        context.fillRect(x - 8, y - 30, 16, 12);
      }
    }
    status.textContent = JSON.stringify({
      tick: sim.t,
      human_alive: sim.alive[0],
      bot_alive: sim.alive[1],
      bun_score: sim.bunScore,
      carrying: sim.bunCarried,
      winner: sim.done ? sim.winner : null,
    }, null, 2);
  }

  function tick() {
    if (!sim.done) {
      const human = humanAction();
      const useModel = opponentSelect.value === 'model' && loadedModel;
      const opponent = useModel ? loadedModel.act(sim, 1, modelRng) : bot.act(sim, 1);
      const info = sim.step([human, opponent]);
      if (!useModel) bot.observeTransition(info, QQTBunRuleBot.stateFromSim(sim), 1);
    }
    render();
  }

  window.addEventListener('keydown', (event) => {
    if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'].includes(event.code)) event.preventDefault();
    held.add(event.code);
    if (event.code === 'Space') bombQueued = true;
    if (event.code === 'KeyR') reset();
  });
  window.addEventListener('keyup', (event) => held.delete(event.code));
  restart.addEventListener('click', reset);
  modelFile.addEventListener('change', async () => {
    const file = modelFile.files && modelFile.files[0];
    if (!file) return;
    try {
      const document = JSON.parse(await file.text());
      const arch = document.meta && document.meta.arch;
      loadedModel = arch === 'transformer' ? new QQT.TransformerModel(document)
        : arch === 'cnn' ? new QQT.CNNModel(document) : new QQT.MLPModel(document);
      opponentSelect.value = 'model';
      modelStatus.textContent = `已加载：${document.meta.display_name || document.meta.name || file.name}`;
      reset();
    } catch (error) {
      loadedModel = null;
      opponentSelect.value = 'rule';
      modelStatus.textContent = `加载失败：${error.message}`;
    }
  });
  reset();
  render();
  setInterval(tick, 100);
})();
