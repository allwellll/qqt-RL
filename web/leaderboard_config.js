'use strict';
window.QQTLeaderboardConfig = Object.freeze({
  url: 'https://ozfdtqtlwrqywwfdxfac.supabase.co',
  publishableKey: 'sb_publishable_frMJjdQUB1cSU24XahSCVg_iCVcSkaq',
  // Publish this client only after the win-record SQL/RPC upgrade is verified.
  leaderboardRpc: 'qqt_my_win_leaderboard',
  // Deployed separately from Pages. An absent/unavailable function falls back to
  // the old result RPC, which keeps the game playable but records no IP.
  submitResultUrl: 'https://ozfdtqtlwrqywwfdxfac.supabase.co/functions/v1/submit-result',
});
