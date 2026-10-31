"""Durable, bounded export/replay/browser verification after the final gate."""
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha

ROOT = Path(__file__).resolve().parents[1]
O = Path(os.environ.get('BT_PUBLICATION_BASE', 'runs/bomb_threat_v1/publish_best_20261030'))
W = Path(os.environ.get('BT_PUBLICATION_WORKTREE', '../cx_publish_best_20261030'))


def read(p):
    return json.loads(Path(p).read_text())


def run(args, log, env=None, cwd=None):
    with (O / log).open('ab') as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, check=True)


def main():
    guard = lock(O / 'PREPUBLICATION_OWNER.lock')
    while not (O / 'FINAL_SELECTION_COMPLETE.json').exists():
        atomic_json(O / 'PREPUBLICATION_STATUS.json', dict(status='WAITING_FINAL_SELECTION', unix=time.time()))
        time.sleep(5)
    selection = read(O / 'FINAL_SELECTION.json')
    assert read(O / 'FINAL_SELECTION_COMPLETE.json')['selection_sha256'] == sha(O / 'FINAL_SELECTION.json')
    if not selection['publish']:
        atomic_json(O / 'PREPUBLICATION_ACCEPTANCE.json', dict(status='RETAIN_BASELINE_NO_PUBLICATION', winner='baseline'))
        return
    prepare_command=[sys.executable, '-m', os.environ.get('BT_PUBLICATION_PREPARE_MODULE', 'scripts.best_publish_prepare_20261030')]
    if os.environ.get('BT_PUBLICATION_PREPARE_FLAG'):
        prepare_command.append(os.environ['BT_PUBLICATION_PREPARE_FLAG'])
    run(prepare_command, 'PREPARE.log')
    row = read(O / 'CATALOG_ROW.json')
    manifest = read(O / 'manifest.json')
    assert manifest['raw_tensors_unchanged']
    cpu = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    fixture = O / 'JS_ONNX.json'
    if not fixture.exists() or read(fixture)['model_sha256'] != row['sha256'] or 'reference_zero' not in read(fixture):
        run(['taskset', '-c', '140-141', 'node', 'scripts/verify_a3sr_release.js',
             str(O / row['file']), str(O / row['onnx']['file']), str(fixture)], 'JS_ONNX.log', cpu)
    parity = O / 'PYTHON_PARITY.json'
    if not parity.exists() or read(parity)['model_sha256'] != row['sha256']:
        run(['taskset', '-c', '140-141', sys.executable, 'scripts/verify_bun_release.py',
             '--checkpoint', manifest['source_ref'], '--model', str(O / row['file']),
             '--fixtures', str(fixture), '--out', str(parity)], 'PYTHON_PARITY.log', cpu)
    assert read(parity)['passed'] and read(parity)['packed_weights_exact']
    row['onnx']['reference_zero'] = read(fixture)['reference_zero']
    atomic_json(O / 'CATALOG_ROW.json', row)
    old = read(W / 'web/models.json')
    original = read(O / 'ONLINE_CATALOG_BEFORE.json')
    if old['models'][0]['id'] != row['id']:
        assert old == original, 'Isolated release catalog changed since task start'
        old['models'].insert(0, row)
    else:
        old['models'][0] = row
    atomic_json(W / 'web/models.json', old)
    assert old['models'][1:] == original['models']
    test = W / 'web/test_model_catalog.js'
    s = test.read_text()
    previous = original['models'][0]
    s = s.replace(f"assert.strictEqual(current.id,{json.dumps(previous['id'])});", f"assert.strictEqual(current.id,{json.dumps(row['id'])});")
    s = s.replace(f"assert.strictEqual(current.id,'{previous['id']}');", f"assert.strictEqual(current.id,{json.dumps(row['id'])});")
    s = s.replace(f"assert.strictEqual(current.source_sha256,'{previous['source_sha256']}');", f"assert.strictEqual(current.source_sha256,{json.dumps(row['source_sha256'])});")
    s = s.replace(f"assert.strictEqual(current.source_sha256,{json.dumps(previous['source_sha256'])});", f"assert.strictEqual(current.source_sha256,{json.dumps(row['source_sha256'])});")
    import re
    s = re.sub(r'assert\.strictEqual\(current\.score,[^;]+;', f"assert.strictEqual(current.score,{row['score']});", s)
    s = s.replace("assert.strictEqual(current.id,'bun392-a3sr-u1280');", f"assert.strictEqual(current.id,{json.dumps(row['id'])});")
    s = s.replace("assert.strictEqual(current.source_sha256,'a552a3094182db7d49d498d8ebe37cdae34fc8ba058396661b7e58c2e812699a');",
                  f"assert.strictEqual(current.source_sha256,{json.dumps(row['source_sha256'])});")
    s = s.replace('assert.strictEqual(current.score,53/256);', f"assert.strictEqual(current.score,{row['score']});")
    test.write_text(s)
    http_test = W / 'web/test_http.js'
    http_source = http_test.read_text()
    http_source = re.sub(r'assert\.strictEqual\(manifest\.models\.length,\s*\d+\);',
                        f'assert.strictEqual(manifest.models.length, {len(old["models"])});', http_source)
    http_source = re.sub(r'assert\.strictEqual\(manifest\.models\[0\]\.id,\s*[^;]+;',
                        f'assert.strictEqual(manifest.models[0].id, {json.dumps(row["id"])});', http_source)
    http_source = http_source.replace('assert.strictEqual(manifest.models.length, 12);',
                                      f'assert.strictEqual(manifest.models.length, {len(old["models"])});')
    http_source = http_source.replace("assert.strictEqual(manifest.models[0].id, 'bun392-a3sr-u1280');",
                                      f'assert.strictEqual(manifest.models[0].id, {json.dumps(row["id"])});')
    http_test.write_text(http_source)
    models = W / 'web/models'
    models.mkdir(exist_ok=True)
    for asset in (row, row['onnx']):
        target = models / asset['file']
        if not target.exists():
            target.symlink_to((O / asset['file']).resolve())
        assert sha(target) == asset['sha256']
    atomic_json(O / 'PREPUBLICATION_STATUS.json', dict(status='VERIFYING_BROWSER_TESTS_AND_REPLAY', unix=time.time()))
    # CPU-only replay runs concurrently with catalog tests/browser inference.
    r = read(O / 'REPLAY_PREPARED.json')
    replay_env = dict(cpu, BT_REPLAY_BASE=r['base'])
    logfile = (O / 'REPLAY_OWNER.log').open('ab')
    replay = None
    replay_complete = Path(r['base']) / 'RECORDING_COMPLETE.json'
    existing_replay = False
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            variables = (proc / 'environ').read_bytes().split(b'\0')
            if b'scripts.bomb_threat_a3sr_replay_owner' in cmd and ('BT_REPLAY_BASE='+r['base']).encode() in variables:
                existing_replay = True
        except OSError:
            pass
    if not replay_complete.exists() and not existing_replay:
        replay = subprocess.Popen(['taskset', '-c', '136-139', sys.executable, '-m', 'scripts.bomb_threat_a3sr_replay_owner'],
                                  env=replay_env, stdout=logfile, stderr=subprocess.STDOUT)
    try:
        run(['npm', 'test'], 'NPM_TEST.log', cpu, W)
        run(['npm', 'run', 'test:review'], 'REVIEW_TEST.log', cpu, W)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(W / 'web'))
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            run(['taskset', '-c', '140-143', 'node', 'scripts/verify_a3sr_publication_browser.js'],
                'LOCAL_BROWSER.log', dict(cpu, BT_PUBLICATION_BASE=str(O), LOCAL='1', WEB_URL=f'http://127.0.0.1:{server.server_port}/'))
        finally:
            server.shutdown()
            server.server_close()
        if replay is not None:
            assert replay.wait() == 0
        else:
            while not replay_complete.exists():
                time.sleep(5)
    finally:
        logfile.close()
    assert read(Path(r['base']) / 'RECORDING_COMPLETE.json')['status'] == 'MATCH'
    if os.environ.get('BT_REPLAY_R1_METRICS') == '1':
        run(['node', 'scripts/bomb_threat_r1_replay_metrics.js', r['base']], 'REPLAY_METRICS.log', replay_env)
    run([sys.executable, '-m', 'scripts.build_bomb_threat_a3sr_replay'], 'REPLAY_BUILD.log', replay_env)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        run(['taskset', '-c', '140-143', 'node', 'scripts/verify_bomb_threat_a3sr_replay_browser.js'],
            'REPLAY_LOCAL_BROWSER.log', dict(replay_env, BROWSER_LOCAL='1', REPLAY_URL=f'http://127.0.0.1:{server.server_port}/'+r['base']+'/'+r['html']))
    finally:
        server.shutdown()
        server.server_close()
    run(['git', '-C', str(W), 'diff', '--check'], 'DIFF_CHECK.log')
    changes = subprocess.check_output(['git', '-C', str(W), 'diff', '--name-only']).decode().splitlines()
    assert sorted(changes) == ['web/models.json', 'web/test_http.js', 'web/test_model_catalog.js'], changes
    assert sha(W / 'web/sim-stock-window-v1.js') == '2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45'
    tracked = subprocess.check_output(['git', '-C', str(W), 'ls-files', '-z']).decode().split('\0')
    findings = []
    patterns = [rb'/mnt/jpfs/afs/', rb'(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s]+/', rb'10\.186\.\d+\.\d+']
    for name in tracked:
        if not name:
            continue
        data = (W / name).read_bytes()
        if any(re.search(p, data) for p in patterns):
            findings.append(name)
    for p in (O / 'manifest.json', O / 'CATALOG_ROW.json', O / row['file']):
        if any(re.search(pattern, p.read_bytes()) for pattern in patterns):
            findings.append(p.name)
    assert not findings, findings
    assert max((W / n).stat().st_size for n in tracked if n) < 100_000_000
    atomic_json(O / 'PREPUBLICATION_ACCEPTANCE.json', dict(status='PASS', selected_policy_sha256=row['source_sha256'],
                source_ref=manifest['source_ref'], selection_sha256=sha(O / 'FINAL_SELECTION.json'),
                python_packed_tensor_exact=True, JS_ONNX='PASS', Python_fp32='PASS', local_Chromium='PASS',
                npm_test='PASS', test_review='PASS', replay_recording='MATCH', replay_browser='PASS',
                changed_files=changes, old_model_objects_preserved=len(original['models']), privacy_findings=findings,
                database_writes_blocked=True, training_untouched=True, unix=time.time()))
    atomic_json(O / 'PREPUBLICATION_STATUS.json', dict(status='READY_FOR_RELEASE', unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        message = str(e).replace(str(ROOT), '<WORKTREE>')
        atomic_json(O / 'PREPUBLICATION_FAILURE.json', dict(status='BLOCKED', reason=message, training_untouched=True, unix=time.time()))
        raise
