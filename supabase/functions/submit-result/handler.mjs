// Forwarded IPs are intentionally accepted even though clients may forge them.
// Parse strictly before passing to PostgreSQL inet; never log addresses or return raw values.
export function parseIp(value) {
  if (typeof value !== 'string') return null;
  const ip = value.trim();
  if (!ip || ip.length > 45 || /[^0-9a-fA-F:.]/.test(ip)) return null;
  if (!ip.includes(':')) {
    const parts = ip.split('.');
    return parts.length === 4 && parts.every(p => /^(0|[1-9][0-9]{0,2})$/.test(p) && Number(p) <= 255) ? ip : null;
  }
  try { return new URL(`http://[${ip}]/`).hostname.slice(1, -1); } catch (_) { return null; }
}
export function requestIp(headers) {
  for (const value of [headers.get('x-forwarded-for')?.split(',')[0], headers.get('cf-connecting-ip'), headers.get('x-real-ip')]) {
    const parsed = parseIp(value); if (parsed) return parsed;
  }
  return null;
}

const baseCors = {
  'access-control-allow-headers': 'apikey, authorization, content-type, x-client-info',
  'access-control-allow-methods': 'POST, OPTIONS',
  'vary': 'Origin',
};

export function createSettlementHandler({ env, createClient }) {
  return async request => {
    const headers = { ...baseCors, 'content-type': 'application/json' };
    const response = (body, status = 200) => new Response(JSON.stringify(body), { status, headers });
    try {
      const origin = request.headers.get('origin');
      const allowed = (env('CORS_ALLOWED_ORIGINS') || 'https://allwellll.github.io')
        .split(',').map(value => value.trim()).filter(Boolean);
      if (origin && !allowed.includes(origin)) return response({ error: 'origin not allowed' }, 403);
      if (origin) headers['access-control-allow-origin'] = origin;
      if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers });
      if (request.method !== 'POST') return response({ error: 'method not allowed' }, 405);
      const url = env('SUPABASE_URL'), serviceKey = env('SUPABASE_SERVICE_ROLE_KEY');
      if (!url || !serviceKey) return response({ error: 'edge function is not configured' }, 503);
      const text = await request.text();
      if (text.length > 16384) return response({ error: 'payload too large' }, 413);
      let body;
      try { body = JSON.parse(text); } catch (_) { return response({ error: 'invalid json' }, 400); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) return response({ error: 'invalid payload' }, 400);
      const payload = body.p_payload || body.payload || body;
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return response({ error: 'invalid payload' }, 400);
      const client = createClient(url, serviceKey, { auth: { persistSession: false, autoRefreshToken: false } });
      const { data, error } = await client.rpc('qqt_submit_result', { p_payload: payload });
      if (error) return response({ error: 'settlement rejected', code: error.code || null }, error.code === '42501' ? 401 : 422);
      if (!data || !Number.isInteger(data.level) || !Number.isInteger(data.games)) {
        return response({ error: 'invalid settlement response' }, 502);
      }
      const rawIp = requestIp(request.headers);
      let metadata = { metadata_recorded: false, network_metadata_recorded: false };
      if (rawIp) {
        // Metadata failure must not reject a match already committed by qqt_submit_result.
        try {
          const recorded = await client.rpc('qqt_record_player_ip', {
            p_player_id: payload.player_id, p_raw_ip: rawIp,
          });
          const masked = recorded.data?.ip_display;
          if (!recorded.error && recorded.data?.recorded === true && typeof masked === 'string' &&
              (/^\d{1,3}\.\*\.\*\.\d{1,3}$/.test(masked) || /^[0-9a-f]{1,4}:\*:\*:[0-9a-f]{1,4}$/i.test(masked))) {
            metadata = { metadata_recorded: true, network_metadata_recorded: true, ip_display: masked };
          }
        } catch (_) { /* Return valid settlement with metadata_recorded=false. */ }
      }
      // Explicit response allowlist prevents raw metadata or internal credentials leaking.
      const safe = Object.fromEntries(['level', 'points', 'wins', 'games', 'total_game_ms',
        'level_reached_ms', 'last_level_up_ms', 'client_match_id', 'match_upgraded', 'match_rank']
        .filter(key => Object.hasOwn(data, key)).map(key => [key, data[key]]));
      return response({ ...safe, ...metadata });
    } catch (_) {
      return response({ error: 'internal error' }, 500);
    }
  };
}
