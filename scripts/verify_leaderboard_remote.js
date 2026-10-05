'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { execFileSync } = require('child_process');
const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync('web/leaderboard_config.js','utf8'), sandbox);
const config = sandbox.window.QQTLeaderboardConfig;
(async () => {
  const checks = [];
  for (const [endpoint, method, body] of [
    ['rpc/qqt_leaderboard','POST',{}],
    ['rpc/qqt_submit_result','POST',{p_payload:{ip:'1.2.3.4'}}],
    ['players?select=player_id&limit=1','GET'],
    ['match_results?select=client_match_id&limit=1','GET'],
  ]) {
    const args = ['--silent','--show-error','--retry','2','--max-time','30','--config','-',
      '--request',method,'--write-out','\n%{http_code}'];
    if (body !== undefined) args.push('--data-binary',JSON.stringify(body));
    const response = execFileSync('curl', args, { encoding:'utf8', stdio:['pipe','pipe','pipe'],
      input: `url = "${config.url}/rest/v1/${endpoint}"\nheader = "apikey: ${config.publishableKey}"\nheader = "Content-Type: application/json"\n` });
    const split = response.lastIndexOf('\n'), status = Number(response.slice(split+1));
    const value = JSON.parse(response.slice(0,split));
    checks.push({ endpoint, status, code: value.code || null, rows: Array.isArray(value) ? value.length : null });
  }
  const initialized = checks[0].status === 200;
  if (initialized && (checks[1].status < 400 || checks.slice(2).some(x => x.status < 400))) {
    throw new Error('unexpected remote permissions');
  }
  const evidence = { initialized, ddlExecutedByThisTask: false, validSettlementE2E: false, checks };
  const out = path.resolve('runs/leaderboard_20261005/remote.json'); fs.mkdirSync(path.dirname(out), {recursive:true});
  fs.writeFileSync(out,JSON.stringify(evidence,null,2)+'\n'); console.log(JSON.stringify(evidence,null,2));
})().catch(error => { console.error(error.message); process.exitCode=1; });
