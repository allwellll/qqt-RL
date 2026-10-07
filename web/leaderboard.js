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
      throw new Error('昵称需 1–24 字，感言最多 80 字，不能包含控制字符或尖括号');
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
  function createClient({ config, storage, crypto, fetch, onChange = () => {}, now = Date.now, settlementReady = true }) {
    let data, persistent = true, pendingRequest = null, status = '', rows = [], progress = null, settlement = null, profileRequest = null, refreshRevision = 0, combinedRequest = null, identityRequest = null, identityRevision = 0;
    let profileKnown = false;
    try { data = JSON.parse(storage.getItem(KEY)); } catch (_) { persistent = false; }
    if (!data || !/^[0-9a-f-]{36}$/.test(data.player_id || '') || !/^[0-9a-f]{64}$/.test(data.secret || '')) {
      const playerId = uuid(crypto);
      data = { player_id: playerId, secret: secret(crypto), nickname: defaultNickname(playerId),
        nickname_auto: true, victory_message: '', total_ms: 0, queue: [] };
    }
    data.queue = Array.isArray(data.queue) ? data.queue.slice(-20) : [];
    data.total_ms = Number.isSafeInteger(data.total_ms) ? data.total_ms : 0;
    data.nextDeclaration ??= data.nextProfile?.victory_message || '';
    delete data.nextProfile;
    progress = data.progress || null;
    try { Object.assign(data, profile(data.nickname, data.victory_message)); }
    catch (_) { Object.assign(data, { nickname: defaultNickname(data.player_id), nickname_auto: true, victory_message: '' }); }
    data.cards = Array.isArray(data.cards) ? data.cards.slice(-20) : [];
    // Round 2 stored unfinished results only in queue. Recover them into editable
    // cards without authorizing or changing an already accepted retry payload.
    for (const match of data.queue) {
      if (data.cards.some(card => card.client_match_id === match.client_match_id)) continue;
      data.cards.push({ client_match_id: match.client_match_id, result: match.result,
        duration_ms: match.game_duration_ms, submitted: false, eligible: true, closed: false, ranking: null,
        draft: { nickname: data.nickname_auto === true ? '' : match.nickname, victory_message: match.victory_message || '' },
        ...(match.approved !== false ? { acceptedInput: { nickname: match.nickname, victory_message: match.victory_message },
          wasSavedNickname: data.nickname_auto !== true } : {}), cardStatus: '' });
    }
    data.cards = data.cards.slice(-20);
    settlement = data.cards.findLast(card => !card.complete) || null;
    const completed = new Set([...data.queue, ...data.cards].map(x => x.client_match_id));
    function state() { return { nickname: data.nickname, victory_message: data.victory_message,
      player_id: data.player_id, persistent, status, rows, progress, settlement, settlementReady, pending: data.queue.length,
      savedNickname: data.confirmed_profile?.registered === true, profileReady: profileKnown,
      profileSubmitting: !!profileRequest, submitting: !!pendingRequest || !!combinedRequest }; }
    function remember() {
      if (settlement) {
        data.cards = data.cards.filter(card => card.client_match_id !== settlement.client_match_id);
        data.cards.push({ ...settlement }); data.cards = data.cards.slice(-20);
      }
      save();
    }
    function emit(text) { if (text !== undefined) status = text; remember(); onChange(state()); }
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
    function adoptProfile(result) {
      if (result.profile_contract_version !== 2 || typeof result.registered !== 'boolean') throw new Error('资料服务正在更新，请稍后重试');
      if (result.registered) Object.assign(data, profile(result.nickname, result.victory_message), { nickname_auto: false });
      data.confirmed_profile = { registered: result.registered, version: 2 };
      profileKnown = true;
    }
    function syncProfile() {
      if (identityRequest) return identityRequest;
      const revision = identityRevision;
      identityRequest = (async () => {
        try {
          const result = await rpc('qqt_get_profile', { p_player_id: data.player_id, p_player_secret: data.secret });
          if (revision !== identityRevision) return profileKnown;
          adoptProfile(result);
          emit(); return true;
        } catch (_) {
          if (revision !== identityRevision) return profileKnown;
          profileKnown = false;
          if (settlement) settlement.cardStatus = '资料服务暂时无法连接，请重试刷新';
          emit('资料服务暂时无法连接，请重试刷新'); return false;
        }
      })().finally(() => { identityRequest = null; });
      return identityRequest;
    }
    function successText(card) {
      return card.refreshFailed ? '提交成功，排行榜刷新失败，可重试刷新'
        : card.pendingEdits ? '提交成功！新的感言已保留到下一局'
        : card.profileSuperseded ? '提交成功！感言沿用较新一局的内容' : '提交成功！';
    }
    async function refresh() {
      const revision = ++refreshRevision;
      try {
        const responseRows = await rpc('qqt_leaderboard', {});
        if (!Array.isArray(responseRows)) throw new Error('排行榜响应格式错误');
        if (revision !== refreshRevision) return false;
        rows = responseRows;
        if (settlement) {
          settlement.refreshFailed = false;
          if (settlement.complete && settlement.cardStatus?.includes('排行榜刷新失败')) settlement.cardStatus = successText(settlement);
        }
        emit('排行榜已更新'); return true;
      } catch (error) {
        if (revision !== refreshRevision) return false;
        if (settlement) settlement.refreshFailed = true;
        emit(settlement?.submitted ? '提交成功，排行榜刷新失败，可重试刷新' : '排行榜暂时没连上，请重试刷新');
        return false;
      }
    }
    function drain() {
      if (pendingRequest) return pendingRequest;
      pendingRequest = (async () => {
        try {
          data.queue = data.queue.filter(match => Number.isFinite(Date.parse(match.completed_at)) &&
            now() - Date.parse(match.completed_at) <= 7 * 86400000);
          save();
          do {
            while (data.queue.some(match => match.approved !== false)) {
              const index = data.queue.findIndex(match => match.approved !== false);
              const { approved, ...match } = data.queue[index];
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
              const currentCard = settlement?.client_match_id === match.client_match_id;
              const card = currentCard ? settlement : data.cards.find(value => value.client_match_id === match.client_match_id);
              if (card) {
                // The queued payload carries the terminal result; the original
                // match metadata passed to begin() intentionally does not.
                const receipt = { ...card, submitted: true,
                  ranking: validRanking(result.match_rank, match) ? result.match_rank : null,
                  closed: currentCard ? false : card.closed, upgraded: result.client_match_id === match.client_match_id && result.match_upgraded === true,
                  level: result.level };
                if (currentCard) settlement = receipt;
                else data.cards = data.cards.map(value => value.client_match_id === match.client_match_id ? receipt : value);
              }
              data.progress = progress;
              data.queue.splice(index, 1); refreshRevision++; identityRevision++; save(); emit('提交成功');
              await refresh();
            }
            // A fast restart may finish another match while the prior leaderboard read
            // is pending. Drain that newly queued result before releasing this request.
          } while (data.queue.some(match => match.approved !== false));
        } catch (error) { emit('提交失败，请重试'); }
      })().finally(() => { pendingRequest = null; emit(); });
      return pendingRequest;
    }
    function begin(metadata) { return { ...metadata, client_match_id: uuid(crypto), started: now() }; }
    function finish(match, { result, gameDurationMs, autoSubmit = true }) {
      if (!match || completed.has(match.client_match_id)) return Promise.resolve(false);
      completed.add(match.client_match_id);
      settlement = { client_match_id: match.client_match_id, result, duration_ms: gameDurationMs, submitted: false, ranking: null, closed: false, eligible: false,
        draft: { nickname: data.confirmed_profile?.registered ? data.nickname : '', victory_message: data.nextDeclaration || '' }, cardStatus: '' };
      const wall = Math.round(now() - match.started);
      if (!['win','loss','draw'].includes(result) || !Number.isInteger(gameDurationMs) ||
          gameDurationMs < 1000 || gameDurationMs > 240000 || wall < Math.max(1000, gameDurationMs * .75) || wall > 3600000) {
        emit('本局无法参与排名，继续练习吧～'); return Promise.resolve(false);
      }
      if (data.queue.length >= 20) { emit('请先完成之前的提交'); return Promise.resolve(false); }
      data.total_ms += gameDurationMs;
      const { client_match_id, opponent, difficulty, seed, mode, map_id, client_version } = match;
      settlement.eligible = true;
      data.queue.push({ approved: autoSubmit, player_id: data.player_id, client_match_id, nickname: data.nickname,
        victory_message: data.victory_message, result, game_duration_ms: gameDurationMs,
        wall_duration_ms: wall, client_total_ms: data.total_ms, opponent, difficulty,
        seed, mode, map_id, client_version, completed_at: new Date(now()).toISOString() });
      save(); emit(autoSubmit ? '提交中…' : ''); return autoSubmit ? drain() : Promise.resolve(true);
    }
    function submitProfile(nick, message) {
      if (settlement?.result === 'loss') return Promise.reject(new Error('失败局不可提交资料'));
      if (profileRequest) return profileRequest;
      if (!settlement || !settlement.submitted || settlement.profileSkipped) {
        return Promise.reject(new Error('请先完成本局战绩提交'));
      }
      if (settlement.profileReceiptVersion === 2) return Promise.resolve(true);
      let values;
      try { values = profile(data.confirmed_profile?.registered ? data.nickname : nick || data.nickname, message); }
      catch (error) { return Promise.reject(error); }
      const matchId = settlement.client_match_id;
      // One immutable profile intent per match, persisted before any network write.
      settlement.profileIntent ??= { victory_message: values.victory_message };
      remember();
      profileRequest = (async () => {
        const result = await rpc('qqt_update_profile', { p_player_id: data.player_id, p_player_secret: data.secret,
          p_client_match_id: matchId, p_nickname: null, p_victory_message: settlement.profileIntent.victory_message });
        if (!result || result.saved !== true || result.profile_contract_version !== 2 || result.client_match_id !== matchId ||
            typeof result.nickname !== 'string' || typeof result.victory_message !== 'string') throw new Error('资料服务正在更新，请稍后重试');
        adoptProfile({ ...result, registered: true });
        identityRevision++;
        refreshRevision++;
        if (settlement && settlement.client_match_id === matchId) {
          settlement.profileSaved = true;
          settlement.profileReceiptVersion = 2;
          settlement.profileSuperseded = result.superseded === true;
          settlement.pendingEdits = String(settlement.draft?.victory_message || '').trim() !== settlement.profileIntent.victory_message;
          data.nextDeclaration = settlement.pendingEdits ? settlement.draft.victory_message : '';
        }
        emit('感言已提交'); await refresh();
        return true;
      })().finally(() => { profileRequest = null; emit(); });
      emit(); return profileRequest;
    }
    function submitSettlement() {
      if (!settlement || !settlement.eligible || settlement.submitted) return Promise.resolve(false);
      const match = data.queue.find(x => x.client_match_id === settlement.client_match_id);
      if (!match) return Promise.resolve(false);
      match.approved = true; save();
      const request = drain(); emit('提交中…'); return request;
    }
    function setDraft(nickname, message) {
      if (!settlement || settlement.result === 'loss') return;
      settlement.draft = { nickname: data.confirmed_profile?.registered ? data.nickname : String(nickname), victory_message: String(message) };
      if (settlement.profileReceiptVersion === 2) {
        data.nextDeclaration = String(message); settlement.pendingEdits = true;
        settlement.cardStatus = successText(settlement);
      } else settlement.cardStatus = '';
      emit();
    }
    function submitCard() {
      if (combinedRequest) return combinedRequest;
      const card = settlement;
      if (!card || card.result === 'loss' || !card.eligible || card.complete) return Promise.resolve(false);
      card.cardStatus = '提交中…';
      combinedRequest = Promise.resolve().then(async () => {
        try {
          if (!profileKnown && !await syncProfile()) throw new Error('资料服务暂时无法连接，请重试刷新');
          const values = profile(data.confirmed_profile?.registered ? data.nickname : card.draft.nickname, card.draft.victory_message);
          card.profileIntent ??= { victory_message: values.victory_message };
          if (!card.submitted) {
            const queued = data.queue.find(x => x.client_match_id === card.client_match_id);
            if (!queued) throw new Error('本局已过期，请开始新一局');
            if (queued.approved === false) {
              queued.nickname = values.nickname;
              queued.victory_message = values.victory_message;
              card.acceptedInput = { ...values };
            }
          }
          remember();
          if (!card.submitted) await submitSettlement();
          // drain replaces the settlement object with the server receipt.
          const current = settlement;
          if (!current || current.client_match_id !== card.client_match_id || !current.submitted) throw new Error('result failed');
          await submitProfile(values.nickname, values.victory_message);
          current.complete = true;
          current.cardStatus = successText(current);
          return true;
        } catch (error) {
          if (settlement?.client_match_id === card.client_match_id) settlement.cardStatus = settlement.submitted
            ? '战绩已提交，感言提交失败，请重试' : /昵称|感言|资料服务|过期/.test(error.message) ? error.message : '提交失败，请重试';
          return false;
        }
      }).finally(() => { combinedRequest = null; emit(); });
      emit(); return combinedRequest;
    }
    function openPending() {
      if (pendingRequest || profileRequest || combinedRequest) return false;
      const card = data.cards.findLast(value => !value.complete);
      if (!card) return false;
      settlement = { ...card, closed: false }; emit(); return true;
    }
    function profileActive() { return !!profileRequest || !!(settlement && settlement.result !== 'loss' && settlement.submitted && !settlement.profileSkipped && !settlement.profileSaved); }
    return { state, begin, finish, refresh, syncProfile, retry: drain, setDraft, submitCard, openPending, submitProfile, submitSettlement, profileActive,
      setSettlementReady(ready) { if (settlementReady !== !!ready) { settlementReady = !!ready; emit(); } },
      markProfileDirty() { if (settlement) settlement.profileSaved = false; emit(); },
      closeSettlement() { if (pendingRequest || profileRequest || combinedRequest) return false; if (settlement) settlement.closed = true; emit(); return true; },
      reopenSettlement() { if (settlement) settlement.closed = false; emit(); },
      submitPending() {
        if (profileActive()) return Promise.resolve(false);
        if ((!settlement || !data.queue.some(x => x.client_match_id === settlement.client_match_id)) && data.queue.length) {
          const last = data.queue[data.queue.length - 1];
          settlement = { client_match_id: last.client_match_id, result: last.result, duration_ms: last.game_duration_ms,
            submitted: false, eligible: true, closed: false, ranking: null };
        }
        for (const match of data.queue) match.approved = true;
        save(); const request = drain(); emit('提交中…'); return request;
      },
      skipProfile() { if (settlement) settlement = { ...settlement, profileSkipped: true }; emit(); }, clearSettlement() { remember(); settlement = null; emit(); },
      setProfile(nick, message) { Object.assign(data, profile(nick, message), { nickname_auto: false }); save(); emit('昵称和感言已保存'); } };
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
        : value.submitted ? '暂无排名' : value.result === 'loss' ? '' : '提交后查看排名' };
  }
  // Canvas keeps the terminal title; HTML exposes accessible actions and status.
  function visibleSettlement(state) {
    const value = state.settlement;
    return state.settlementReady !== false && !!value && !value.closed &&
      ['win', 'loss', 'draw'].includes(value.result) &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value.client_match_id || '') &&
      Number.isSafeInteger(value.duration_ms) && value.duration_ms >= 0;
  }
  function renderSettlement(document, state) {
    const el = id => document.getElementById(id), value = visibleSettlement(state) ? state.settlement : null;
    const loss = value?.result === 'loss';
    el('settlement').hidden = !value;
    const form = el('settlement-form');
    if (form) form.hidden = !value || loss;
    const submit = el('settlement-submit'), close = el('settlement-close');
    if (submit) {
      submit.disabled = loss || !value || !value.eligible || value.complete || state.submitting || state.profileReady === false;
      submit.textContent = '提交';
    }
    if (close) close.disabled = !value || state.submitting || state.profileSubmitting;
    const nickname = el('player-nickname'), nicknameField = el('nickname-field'), currentName = el('current-nickname');
    if (nicknameField) nicknameField.hidden = !value || loss || state.savedNickname || state.profileReady === false;
    if (nickname) {
      nickname.disabled = !value || loss || state.savedNickname || state.submitting || state.profileReady === false;
      nickname.required = !!value && !loss && !state.savedNickname;
    }
    const message = el('player-message');
    if (message) message.disabled = !value || loss;
    if (currentName) {
      currentName.hidden = !value || loss || !state.savedNickname;
      currentName.textContent = value && !loss && state.savedNickname ? `昵称：${state.nickname}` : '';
    }
    if (!value) {
      for (const id of ['settlement-title', 'settlement-time', 'settlement-rank', 'settlement-status']) el(id).textContent = '';
      return;
    }
    const lines = settlementText(value);
    el('settlement-title').textContent = { win: '胜利', loss: '失败', draw: '平局' }[value.result] || '本局结束';
    el('settlement-time').textContent = lines.time;
    el('settlement-rank').textContent = [lines.rank, lines.percentile].filter(Boolean).join(' · ');
    el('settlement-status').textContent = loss ? '小伙子，再沉淀沉淀吧'
      : value.cardStatus || (value.eligible ? '' : '本局无法参与排名，继续练习吧～');
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
    const form = document.getElementById('leaderboard-profile');
    if (form) form.hidden = !state.settlement || state.settlement.closed;
  }
  function mount(document, options) {
    const el = id => document.getElementById(id);
    let renderedMatch, actionRequest = null;
    const client = createClient({ ...options, onChange(state) {
      el('leaderboard-status').textContent = state.status;
      const value = state.settlement;
      if (value && renderedMatch !== value.client_match_id) {
        renderedMatch = value.client_match_id;
        el('player-nickname').value = value.draft?.nickname ?? (state.savedNickname ? state.nickname : '');
        el('player-message').value = value.draft?.victory_message ?? '';
      }
      renderSettlement(document, state);
      renderRows(document, el('leaderboard-list'), state.rows);
      el('leaderboard-empty').hidden = state.rows.length > 0;
    } });
    function runSettlementAction(action) {
      if (actionRequest) return actionRequest;
      const state = client.state(), card = state.settlement;
      if (!visibleSettlement(state)) return Promise.resolve(false);
      // Keep submit/X coalesced through the async restart, after write flags clear.
      actionRequest = Promise.resolve().then(action).then(async success => {
        if (!success || client.state().settlement?.client_match_id !== card.client_match_id) return false;
        if (options.onRestart) await options.onRestart();
        return true;
      }).finally(() => { actionRequest = null; });
      return actionRequest;
    }
    el('settlement-form').addEventListener('submit', event => {
      event.preventDefault(); return runSettlementAction(() => client.submitCard());
    });
    for (const id of ['player-nickname', 'player-message']) el(id).addEventListener('input', () =>
      client.setDraft(el('player-nickname').value, el('player-message').value));
    el('leaderboard-retry').addEventListener('click', () => { void client.refresh(); void client.syncProfile(); });
    el('settlement-close').addEventListener('click', () => runSettlementAction(() => client.closeSettlement()));
    el('pending-submit').addEventListener('click', () => client.openPending());
    renderSettlement(document, client.state());
    client.refresh();
    void client.syncProfile();
    return client;
  }

  return { createClient, profile, renderRows, renderSettlement, settlementText, renderUpgradeProfile, validRanking, mount, time, maskIp, defaultNickname };
});
