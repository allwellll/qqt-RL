'use strict';
(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.QQTLeaderboard = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  const KEY = 'qqt.leaderboard.v1';
  const length = value => Array.from(value).length;
  function profile(nickname, victoryMessage) {
    const nick = String(nickname).trim(), message = String(victoryMessage).trim();
    if (!length(nick) || length(nick) > 24 || length(message) > 80 || /[\x00-\x1f\x7f<>]/.test(nick + message)) {
      throw new Error('昵称需 1–24 字，宣言最多 80 字，不能包含控制字符或尖括号');
    }
    return { nickname: nick, victory_message: message };
  }
  function uuid(crypto) { return crypto.randomUUID(); }
  function secret(crypto) {
    return Array.from(crypto.getRandomValues(new Uint8Array(32)), b => b.toString(16).padStart(2, '0')).join('');
  }
  function createClient({ config, storage, crypto, fetch, onChange = () => {}, now = Date.now }) {
    let data, persistent = true, pendingRequest = null, status = '', rows = [], progress = null;
    try { data = JSON.parse(storage.getItem(KEY)); } catch (_) { persistent = false; }
    if (!data || !/^[0-9a-f-]{36}$/.test(data.player_id || '') || !/^[0-9a-f]{64}$/.test(data.secret || '')) {
      data = { player_id: uuid(crypto), secret: secret(crypto), nickname: 'QQT玩家', victory_message: '', total_ms: 0, queue: [] };
    }
    data.queue = Array.isArray(data.queue) ? data.queue.slice(-20) : [];
    data.total_ms = Number.isSafeInteger(data.total_ms) ? data.total_ms : 0;
    progress = data.progress || null;
    try { Object.assign(data, profile(data.nickname, data.victory_message)); }
    catch (_) { Object.assign(data, { nickname: 'QQT玩家', victory_message: '' }); }
    const completed = new Set(data.queue.map(x => x.client_match_id));
    function state() { return { nickname: data.nickname, victory_message: data.victory_message,
      player_id: data.player_id, persistent, status, rows, progress, pending: data.queue.length }; }
    function emit(text) { if (text !== undefined) status = text; onChange(state()); }
    function save() {
      try { storage.setItem(KEY, JSON.stringify(data)); } catch (_) { persistent = false; }
    }
    save();
    async function rpc(name, body) {
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 8000);
      try {
        const response = await fetch(`${config.url}/rest/v1/rpc/${name}`, { method: 'POST',
          headers: { apikey: config.publishableKey, 'Content-Type': 'application/json' },
          body: JSON.stringify(body), signal: controller.signal });
        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          if (['PGRST202', 'PGRST205'].includes(error.code)) throw new Error('排行榜数据库尚未初始化，本地游戏可继续');
          throw new Error(`排行榜请求失败 (${response.status})`);
        }
        return await response.json();
      } finally { clearTimeout(timer); }
    }
    async function refresh() {
      try {
        const responseRows = await rpc('qqt_leaderboard', {});
        if (!Array.isArray(responseRows)) throw new Error('排行榜响应格式错误');
        rows = responseRows;
        emit(data.queue.length ? `有 ${data.queue.length} 局待提交` : '排行榜已更新 · 客户端赛果');
        return true;
      } catch (error) { emit(`${error.message}；可重试`); return false; }
    }
    function drain() {
      if (pendingRequest) return pendingRequest;
      pendingRequest = (async () => {
        try {
          data.queue = data.queue.filter(match => Number.isFinite(Date.parse(match.completed_at)) &&
            now() - Date.parse(match.completed_at) <= 7 * 86400000);
          save();
          while (data.queue.length) {
            const match = data.queue[0];
            progress = await rpc('qqt_submit_result', { p_payload: { ...match, player_secret: data.secret } });
            data.progress = progress;
            data.queue.shift(); save(); emit('结算已提交');
          }
          await refresh();
        } catch (error) { emit(`${error.message}；${data.queue.length} 局待提交，可重试`); }
      })().finally(() => { pendingRequest = null; });
      return pendingRequest;
    }
    function begin(metadata) { return { ...metadata, client_match_id: uuid(crypto), started: now() }; }
    function finish(match, { result, gameDurationMs }) {
      if (!match || completed.has(match.client_match_id)) return Promise.resolve(false);
      completed.add(match.client_match_id);
      const wall = Math.round(now() - match.started);
      if (!['win','loss','draw'].includes(result) || !Number.isInteger(gameDurationMs) ||
          gameDurationMs < 1000 || gameDurationMs > 240000 || wall < Math.max(1000, gameDurationMs * .75) || wall > 3600000) {
        emit('本局计时超出排行榜范围，未提交；本地游戏可继续'); return Promise.resolve(false);
      }
      if (data.queue.length >= 20) { emit('待提交结算已达 20 局，请先重试'); return Promise.resolve(false); }
      data.total_ms += gameDurationMs;
      const { client_match_id, opponent, difficulty, seed, mode, map_id, client_version } = match;
      data.queue.push({ player_id: data.player_id, client_match_id, nickname: data.nickname,
        victory_message: data.victory_message, result, game_duration_ms: gameDurationMs,
        wall_duration_ms: wall, client_total_ms: data.total_ms, opponent, difficulty,
        seed, mode, map_id, client_version, completed_at: new Date(now()).toISOString() });
      save(); emit('正在提交结算…'); return drain();
    }
    return { state, begin, finish, refresh, retry: drain,
      setProfile(nick, message) { Object.assign(data, profile(nick, message)); save(); emit('昵称和宣言已保存；下次结算同步'); } };
  }
  function time(ms) { return ms ? `${(Number(ms) / 1000).toFixed(1)}秒` : '—'; }
  function renderRows(document, target, rows) {
    target.replaceChildren();
    for (const row of rows) {
      const item = document.createElement('li'); item.className = 'leaderboard-entry';
      const title = document.createElement('strong');
      title.textContent = `${row.rank}. ${row.nickname} · Lv.${row.level} (${row.progress}/10)`;
      const stats = document.createElement('span');
      stats.textContent = `升至本级 ${time(row.level_reached_ms)} · 本次升级 ${time(row.last_level_up_ms)} · ${row.wins}/${row.games} 胜 · ${(Number(row.win_rate) * 100).toFixed(1)}%`;
      const message = document.createElement('span'); message.textContent = row.victory_message;
      item.append(title, stats, message); target.append(item);
    }
  }
  function mount(document, options) {
    const el = id => document.getElementById(id);
    const client = createClient({ ...options, onChange(state) {
      el('leaderboard-status').textContent = state.status;
      el('leaderboard-identity').textContent = state.persistent ? '匿名身份保存在此浏览器' : '本地存储不可用：刷新后匿名身份会变化';
      renderRows(document, el('leaderboard-list'), state.rows);
      el('leaderboard-empty').hidden = state.rows.length > 0;
      if (state.progress) el('leaderboard-progress').textContent = `你的等级 Lv.${state.progress.level} · ${state.progress.points % 10}/10 · ${state.progress.wins}/${state.progress.games} 胜`;
    } });
    el('player-nickname').value = client.state().nickname;
    el('player-message').value = client.state().victory_message;
    el('leaderboard-profile').addEventListener('submit', event => {
      event.preventDefault();
      try { client.setProfile(el('player-nickname').value, el('player-message').value); }
      catch (error) { el('leaderboard-status').textContent = error.message; }
    });
    el('leaderboard-retry').addEventListener('click', () => client.retry());
    client.refresh();
    return client;
  }
  return { createClient, profile, renderRows, mount, time };
});
