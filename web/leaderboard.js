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
  // A short UUID fragment is explicitly labelled local; it is never an IP address.
  function defaultNickname(playerId, trustedIpDisplay = null) {
    const ip = trustedMaskedIp(trustedIpDisplay);
    const id = playerId.replace(/-/g, '');
    return `QQT玩家·${ip || `本地${id.slice(0, 3)}…${id.slice(-3)}`}`;
  }
  function trustedMaskedIp(value) {
    if (typeof value !== 'string') return null;
    if (/^\d{1,3}\.\*\.\*\.\d{1,3}$/.test(value) &&
        [value.split('.')[0], value.split('.')[3]].every(part => Number(part) <= 255)) return value;
    return /^[0-9a-f]{1,4}:\*:\*:[0-9a-f]{1,4}$/i.test(value) ? value : null;
  }
  function createClient({ config, storage, crypto, fetch, onChange = () => {}, now = Date.now }) {
    let data, persistent = true, pendingRequest = null, status = '', rows = [], progress = null, settlement = null, profileRequest = null;
    try { data = JSON.parse(storage.getItem(KEY)); } catch (_) { persistent = false; }
    if (!data || !/^[0-9a-f-]{36}$/.test(data.player_id || '') || !/^[0-9a-f]{64}$/.test(data.secret || '')) {
      const playerId = uuid(crypto);
      data = { player_id: playerId, secret: secret(crypto), nickname: defaultNickname(playerId),
        nickname_auto: true, victory_message: '', total_ms: 0, queue: [] };
    }
    data.queue = Array.isArray(data.queue) ? data.queue.slice(-20) : [];
    data.total_ms = Number.isSafeInteger(data.total_ms) ? data.total_ms : 0;
    progress = data.progress || null;
    try { Object.assign(data, profile(data.nickname, data.victory_message)); }
    catch (_) { Object.assign(data, { nickname: defaultNickname(data.player_id), nickname_auto: true, victory_message: '' }); }
    const completed = new Set(data.queue.map(x => x.client_match_id));
    function state() { return { nickname: data.nickname, victory_message: data.victory_message,
      player_id: data.player_id, persistent, status, rows, progress, settlement, pending: data.queue.length,
      savedNickname: data.nickname_auto !== true, profileSubmitting: !!profileRequest }; }
    function emit(text) { if (text !== undefined) status = text; onChange(state()); }
    function save() {
      try { storage.setItem(KEY, JSON.stringify(data)); } catch (_) { persistent = false; }
    }
    save();
    async function requestRpc(endpoint, body) {
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 8000);
      try {
        const response = await fetch(endpoint, { method: 'POST',
          headers: { apikey: config.publishableKey, 'Content-Type': 'application/json' },
          body: JSON.stringify(body), signal: controller.signal });
        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          if (['PGRST202', 'PGRST205'].includes(error.code)) throw new Error(endpoint.endsWith('qqt_update_profile')
            ? '资料提交待数据库升级' : '排行榜数据库尚未初始化，本地游戏可继续');
          throw new Error(`排行榜请求失败 (${response.status})`);
        }
        return await response.json();
      } finally { clearTimeout(timer); }
    }
    async function rpc(name, body) {
      if (name === 'qqt_submit_result' && config.submitResultUrl) {
        try {
          return await requestRpc(config.submitResultUrl, body);
        } catch (error) {
          // An undeployed Edge Function must not take the local game offline. The direct
          // RPC has no trusted IP and is deliberately only a compatibility fallback.
          if (!(error instanceof TypeError) && error.name !== 'AbortError' &&
              !/排行榜请求失败 \((404|405|501|502|503)\)/.test(error.message || '')) throw error;
        }
      }
      return requestRpc(`${config.url}/rest/v1/rpc/${name}`, body);
    }
    async function refresh() {
      try {
        const responseRows = await rpc('qqt_leaderboard', {});
        if (!Array.isArray(responseRows)) throw new Error('排行榜响应格式错误');
        rows = responseRows;
        emit(data.queue.length ? `有 ${data.queue.length} 局待提交` : '排行榜已更新');
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
          do {
            while (data.queue.length) {
              const match = data.queue[0];
              const result = await rpc('qqt_submit_result', { p_payload: { ...match, player_secret: data.secret } });
              if (!result || !['level', 'points', 'wins', 'games'].every(key => Number.isInteger(result[key])) ||
                  result.level < 1 || result.points < 0 || result.wins < 0 || result.games < result.wins) {
                throw new Error('结算响应格式错误');
              }
              progress = result;
              // Only the authenticated settlement response may confirm metadata for this
              // player. Never infer an IP from browser inputs or someone else's Top row.
              if (data.nickname_auto === true && result.network_metadata_recorded === true &&
                  trustedMaskedIp(result.ip_display)) {
                data.nickname = defaultNickname(data.player_id, result.ip_display);
              }
              if (settlement && settlement.client_match_id === match.client_match_id) {
                // The queued payload carries the terminal result; the original
                // match metadata passed to begin() intentionally does not.
                settlement = { ...settlement, submitted: true,
                  ranking: validRanking(result.match_rank, match) ? result.match_rank : null,
                  upgraded: result.client_match_id === match.client_match_id && result.match_upgraded === true,
                  level: result.level };
              }
              data.progress = progress;
              data.queue.shift(); save(); emit('结算已提交');
            }
            await refresh();
            // A fast restart may finish another match while the prior leaderboard read
            // is pending. Drain that newly queued result before releasing this request.
          } while (data.queue.length);
        } catch (error) { emit(`${error.message}；${data.queue.length} 局待提交，可重试`); }
      })().finally(() => { pendingRequest = null; });
      return pendingRequest;
    }
    function begin(metadata) { return { ...metadata, client_match_id: uuid(crypto), started: now() }; }
    function finish(match, { result, gameDurationMs }) {
      if (!match || completed.has(match.client_match_id)) return Promise.resolve(false);
      completed.add(match.client_match_id);
      settlement = { client_match_id: match.client_match_id, result, duration_ms: gameDurationMs, submitted: false, ranking: null };
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
    function submitProfile(nick, message) {
      if (profileRequest) return profileRequest;
      if (!settlement || !settlement.submitted || !settlement.upgraded || settlement.profileSkipped) {
        return Promise.reject(new Error('仅本局排行榜升级后可提交资料'));
      }
      let values;
      try { values = profile(nick, message); }
      catch (error) { return Promise.reject(error); }
      const matchId = settlement.client_match_id;
      profileRequest = (async () => {
        const result = await rpc('qqt_update_profile', { p_player_id: data.player_id, p_player_secret: data.secret,
          p_client_match_id: matchId, p_nickname: values.nickname, p_victory_message: values.victory_message });
        if (!result || result.saved !== true || typeof result.nickname !== 'string' ||
            typeof result.victory_message !== 'string') throw new Error('资料提交响应格式错误');
        const saved = profile(result.nickname, result.victory_message);
        Object.assign(data, saved, { nickname_auto: false }); save();
        emit('昵称和宣言已提交'); await refresh();
        return true;
      })().finally(() => { profileRequest = null; emit(); });
      emit(); return profileRequest;
    }
    return { state, begin, finish, refresh, retry: drain, submitProfile,
      skipProfile() { if (settlement) settlement = { ...settlement, profileSkipped: true }; emit(); }, clearSettlement() { settlement = null; },
      setProfile(nick, message) { Object.assign(data, profile(nick, message), { nickname_auto: false }); save(); emit('昵称和宣言已保存'); } };
  }
  function time(ms) { return ms ? `${(Number(ms) / 1000).toFixed(1)}秒` : '—'; }
  function validRanking(rank, match) {
    return !!rank && Number.isSafeInteger(rank.rank) && Number.isSafeInteger(rank.total) &&
      rank.rank >= 1 && rank.total >= rank.rank && Number.isFinite(rank.percentile) &&
      rank.percentile >= 0 && rank.percentile <= 100 && rank.comparison === 'player-best-v1' &&
      ['result', 'mode', 'map_id', 'difficulty', 'opponent'].every(key => rank[key] === match[key]) &&
      rank.duration_ms === match.game_duration_ms;
  }
  function settlementText(value) {
    if (!value) return null;
    return { time: `本局耗时 ${time(value.duration_ms)}`,
      rank: value.ranking ? `第 ${value.ranking.rank} 名 / ${value.ranking.total} 位玩家` : '',
      percentile: value.ranking ? `超过 ${value.ranking.percentile.toFixed(2)}% 玩家`
        : value.submitted ? '排名待数据库升级' : '排名等待结算提交' };
  }
  // Screen-reader mirror only. All visible terminal text is drawn in the original Canvas.
  function renderSettlement(document, state) {
    const el = id => document.getElementById(id), value = state.settlement;
    el('settlement').hidden = !value;
    if (!value) return;
    const lines = settlementText(value);
    el('settlement-title').textContent = { win: '胜利', loss: '失败', draw: '平局' }[value.result] || '本局结束';
    el('settlement-time').textContent = lines.time;
    el('settlement-rank').textContent = [lines.rank, lines.percentile].filter(Boolean).join(' · ');
    el('settlement-status').textContent = state.status;
  }
  function maskIp(value) {
    const ip = String(value || '').trim();
    if (/^\d{1,3}\.\*\.\*\.\d{1,3}$/.test(ip) || /^[0-9a-f]{1,4}:\*:\*:[0-9a-f]{1,4}$/i.test(ip)) return ip;
    const v4 = ip.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
    if (v4 && v4.slice(1).every(part => Number(part) <= 255)) return `${v4[1]}.*.*.${v4[4]}`;
    const v6 = ip.split(':').filter(Boolean);
    if (v6.length >= 2 && v6.every(part => /^[0-9a-f]{1,4}$/i.test(part))) return `${v6[0].slice(0, 4)}:*:*:${v6[v6.length - 1].slice(-4)}`;
    return '—';
  }
  function renderRows(document, target, rows) {
    target.replaceChildren();
    for (const row of rows) {
      const item = document.createElement('tr'); item.className = 'leaderboard-entry';
      const cells = [
        [row.rank, 'rank'], [row.nickname, 'player'], [time(row.best_win_duration_ms), 'best-win'],
        [row.victory_message || '—', 'message'], [maskIp(row.player_ip || row.ip_display), 'ip'],
      ];
      for (const [value, className] of cells) {
        const cell = document.createElement('td'); cell.className = className; cell.textContent = String(value ?? '—'); item.append(cell);
      }
      target.append(item);
    }
  }
  function renderUpgradeProfile(document, state) {
    const form = document.getElementById('leaderboard-profile'), value = state.settlement;
    form.hidden = !(value && value.submitted && value.upgraded && !value.profileSkipped);
    document.getElementById('profile-submit').disabled = state.profileSubmitting;
    if (!form.hidden) document.getElementById('profile-title').textContent = `排行榜升至 ${value.level} 级！`;
  }
  function mount(document, options) {
    const el = id => document.getElementById(id);
    let renderedMatch;
    const client = createClient({ ...options, onChange(state) {
      el('leaderboard-status').textContent = state.persistent ? state.status : `本地存储不可用 · ${state.status}`;
      const value = state.settlement;
      if (value && value.upgraded && renderedMatch !== value.client_match_id) {
        renderedMatch = value.client_match_id;
        el('player-nickname').value = state.savedNickname ? state.nickname : '';
        el('player-message').value = state.victory_message;
        el('profile-status').textContent = '';
      }
      renderUpgradeProfile(document, state);
      renderRows(document, el('leaderboard-list'), state.rows);
      el('leaderboard-empty').hidden = state.rows.length > 0;
    } });
    el('leaderboard-profile').addEventListener('submit', async event => {
      event.preventDefault();
      const matchId = client.state().settlement?.client_match_id;
      try {
        await client.submitProfile(el('player-nickname').value, el('player-message').value);
        if (client.state().settlement?.client_match_id === matchId) el('profile-status').textContent = '资料已提交，可继续修改宣言';
      } catch (error) {
        if (client.state().settlement?.client_match_id === matchId) el('profile-status').textContent = error.message;
      }
    });
    el('profile-skip').addEventListener('click', () => client.skipProfile());
    el('leaderboard-retry').addEventListener('click', () => client.retry());
    client.refresh();
    return client;
  }
  return { createClient, profile, renderRows, renderSettlement, settlementText, renderUpgradeProfile, validRanking, mount, time, maskIp, defaultNickname };
});
