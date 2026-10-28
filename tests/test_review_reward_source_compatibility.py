"""Audit-only compatibility does not change any published D reward result."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_only_approved_audit_function_differs_and_has_no_reward_callers():
    entry = json.loads((ROOT / 'scripts/review_source_hash_compatibility.json').read_text())['entries'][0]
    current = (ROOT / entry['file']).read_text()
    historical = subprocess.check_output(
        ['git', 'show', entry['published_commit'] + ':' + entry['file']], cwd=ROOT).decode()
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    assert digest(current) == entry['current_sha256']
    assert digest(historical) == entry['published_sha256']
    assert current.count(entry['new_function']) == 1
    assert current.replace(entry['new_function'], entry['old_function']) == historical
    callers = []
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in tracked:
        if not name.endswith('.py') or not name.startswith(('jax_bomb/', 'scripts/', 'qqt_rl/')):
            continue
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if ((isinstance(node, ast.Name) and node.id == entry['function']) or
                (isinstance(node, ast.Attribute) and node.attr == entry['function']) or
                (isinstance(node, ast.Constant) and node.value == entry['function'])):
                callers.append(name)
    # The only runtime reference is the separate spawn-distribution audit CLI.
    assert set(callers) == {'scripts/audit_spawn_buckets.py'}


def test_all_published_d_rewards_match_with_audit_function_forbidden(tmp_path, monkeypatch):
    from jax_bomb import bun_env, je05_bun_env, je07_bun_env

    def forbidden(*args, **kwargs):
        raise AssertionError('Reward reconstruction called spawn_bucket_ids')

    for module in (bun_env, je05_bun_env, je07_bun_env):
        monkeypatch.setattr(module, 'spawn_bucket_ids', forbidden)
    # Use repository modules only; no historical run directory or checkpoint input.
    monkeypatch.setenv('REPLAY_INPUT', str(tmp_path / 'unused-input'))
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    ledger = importlib.import_module('recompute_best_d_rewards')
    assert Path(ledger.M.__file__).resolve() == ROOT / 'scripts/recompute_review_metrics.py'
    assert Path(ledger.M.R.__file__).resolve() == ROOT / 'scripts/diagnostic_reward.py'
    for module in (bun_env, je05_bun_env, je07_bun_env):
        assert module.spawn_bucket_ids is forbidden
    script = r"""
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const P=require('./web/review_protocol'),{instrument}=require('./scripts/diagnostic_counterfactual');
const level=require('./scripts/eval_web_hunter').loadLevel();
(async()=>{
  const file=process.argv[1],folder=process.argv[2],source=fs.readFileSync(file),doc=JSON.parse(source);
  await new P.Player(doc,level).verify();
  const sim=P.replaySetup(doc.meta,level),step=instrument(sim);
  const output=path.join(folder,path.basename(file));fs.writeFileSync(output,source);
  const trace=fs.createWriteStream(output.replace(/\.json$/,'.trace.jsonl'));
  const write=async row=>{if(!trace.write(JSON.stringify(row)+'\n'))await new Promise(r=>trace.once('drain',r));};
  await write({state:P.state(sim),legal:sim.legalMask()});
  for(const row of doc.actions)await write(step(row));
  await new Promise(r=>trace.end(r));
  assert.equal(sim.t,doc.actions.length);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    rows = json.loads((ROOT / 'web/review-catalog.json').read_text())['evaluation']['replays']
    assert len(rows) == 15
    for row in rows:
        source = ROOT / 'web/review-replays' / row['file']
        original = source.read_bytes()
        subprocess.run(['node', '-e', script, str(source), str(tmp_path)], cwd=ROOT,
                       check=True, capture_output=True, text=True)
        copy = tmp_path / row['file']
        # verify=True compares every sparse reward row and the exact episode totals.
        ledger.run(copy, verify=True)
        assert source.read_bytes() == original
        assert copy.read_bytes() == original
        copy.with_suffix('.trace.jsonl').unlink()
        copy.unlink()
