'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const sha = process.argv[2];
if (!/^[a-f0-9]{40}$/.test(sha || '')) throw new Error('expected full deployed commit SHA');

try {
  const credential = execFileSync('git', ['credential', 'fill'], {
    input: 'url=https://github.com/allwellll/qqt-RL.git\n\n', encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
  });
  const token = credential.split('\n').find((x) => x.startsWith('password='))?.slice(9);
  if (!token) throw new Error('missing GitHub credential');
  const api = (endpoint) => JSON.parse(execFileSync('curl', [
    '--silent', '--show-error', '--fail', '--max-time', '45', '--config', '-',
  ], { input: `url = "https://api.github.com/repos/allwellll/qqt-RL/${endpoint}"\n` +
    `header = "Authorization: Bearer ${token}"\nheader = "Accept: application/vnd.github+json"\n`,
  encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }));
  const remote = api('branches/main').commit.sha;
  if (remote !== sha) throw new Error(`remote main differs: ${remote}`);
  const runs = api('actions/workflows/pages.yml/runs?branch=main&per_page=10').workflow_runs;
  const run = runs.find((r) => r.head_sha === sha);
  if (!run || run.status !== 'completed' || run.conclusion !== 'success') {
    throw new Error(`Pages commit ${sha} is ${run ? `${run.status}/${run.conclusion}` : 'not started'}`);
  }
  const jobs = api(`actions/runs/${run.id}/jobs`).jobs.map((j) => ({ name: j.name,
    conclusion: j.conclusion, steps: j.steps.map((s) => ({ name: s.name, conclusion: s.conclusion })) }));
  if (!jobs.some((j) => j.name === 'deploy' && j.conclusion === 'success') ||
      jobs.some((j) => j.conclusion !== 'success')) throw new Error('Pages jobs did not all succeed');
  const files = ['index.html', 'app.js', 'sim.js', 'bun_coop_hunter_bot.js', 'visual_renderer.js',
    'assets/native/sprites.json', 'assets/native/protection.png', 'assets/native/bird.png',
    'assets/native/trap.png', 'assets/native/syrup_pop.wav', 'assets/maps/levels.json', 'assets/maps/bun06_8.map'];
  const resources = files.map((file) => {
    const data = execFileSync('curl', ['--fail', '--silent', '--show-error', '--retry', '3', '--max-time', '45',
      `https://allwellll.github.io/qqt-RL/${file}?verify=${sha}`], { maxBuffer: 10000000, stdio: ['ignore', 'pipe', 'pipe'] });
    if (!data.equals(fs.readFileSync(path.resolve('web', file)))) throw new Error(`published resource differs: ${file}`);
    return { file, bytes: data.length, sha256: crypto.createHash('sha256').update(data).digest('hex') };
  });
  const evidence = { sha, remoteMain: remote, run: { id: run.id, url: run.html_url,
    status: run.status, conclusion: run.conclusion }, jobs, resources };
  const output = path.resolve('runs/bot_reserve_spawn_20261005/pages.json');
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify(evidence, null, 2));
} catch (e) {
  // Do not surface curl config or credential-helper output on network failures.
  console.error(e.status != null ? `verification subprocess failed (${e.status})` : e.message);
  process.exitCode = 1;
}
