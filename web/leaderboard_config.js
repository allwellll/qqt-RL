'use strict';
window.QQTLeaderboardConfig = Object.freeze({
  url: 'https://ozfdtqtlwrqywwfdxfac.supabase.co',
  publishableKey: 'sb_publishable_frMJjdQUB1cSU24XahSCVg_iCVcSkaq',
  rulesetVersion: 'bun-native-v2-face',
  // Face gameplay is live; its board is deferred. Never relabel these results.
  unavailableRulesets: Object.freeze(['bun-native-v2-face']),
  leaderboardRpc: 'qqt_my_win_leaderboard',
  // Deployed separately from Pages. An absent/unavailable function falls back to
  // the old result RPC, which keeps the game playable but records no IP.
  submitResultUrl: 'https://ozfdtqtlwrqywwfdxfac.supabase.co/functions/v1/submit-result',
});
