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
      // No forwarded header has proven platform-controlled provenance. IP recording
      // stays disabled, so malformed/spoofed addresses cannot affect settlement.
      return response({ ...data, metadata_recorded: false, network_metadata_recorded: false });
    } catch (_) {
      return response({ error: 'internal error' }, 500);
    }
  };
}
