"""Publish CPU replay snapshots from one active DCU run, on a fixed cadence."""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.github.com/repos/allwellll/qqt-RL'
PAGES = 'https://allwellll.github.io/qqt-RL/dcu-replays/'
DIRECTORY = 'web/dcu-replays/'
START = '<!-- dcu-training-20261121:start -->'
END = '<!-- dcu-training-20261121:end -->'
NODE = ROOT / '.tools/node/bin/node'


def read(path):
    return json.loads(path.read_text())


def status(message, **fields):
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                      'event': message, **fields}, ensure_ascii=False), flush=True)


def request(token, path, body=None, method=None):
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28',
               'User-Agent': 'qqt-dcu-replays'}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    data = None if body is None else json.dumps(body).encode()
    with urlopen(Request(API + path, data=data, headers=headers, method=method), timeout=45) as response:
        return json.load(response)


def blob(token, data):
    return request(token, '/git/blobs', {
        'content': base64.b64encode(data).decode('ascii'), 'encoding': 'base64'})['sha']


def render_index(index, filename, document, final):
    start, end = index.find(START), index.find(END)
    links = [] if start < 0 else re.findall(r'<li>[^\n]+</li>', index[start:end])
    links = [link for link in links if f'href="{filename}"' not in link]
    utc = datetime.fromisoformat(document['generated_utc'])
    local_time = utc.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M')
    update = html.escape(document['candidate']['update'])
    games = len(document['replays'])
    label = f'{update} · {games} 局 · {local_time} 北京时间' + (' · 最终检查点' if final else '')
    links.insert(0, f'<li><a href="{filename}">{label}</a></li>')
    section = '\n'.join([
        START, '<section><h2>八卡对拼训练 · 2026-11-21</h2>',
        f'<a class="play" href="{filename}">播放最新录像 · {update}</a>',
        '<p>每半小时用最新保存的模型重跑固定对局，对手为 DUEL、SP、G，轮换双方座位。</p>',
        '<ul>', *links, '</ul></section>', END, '',
    ])
    if start >= 0 and end > start:
        return index[:start] + section + index[end + len(END):].lstrip('\n')
    heading = '<h1>DCU 新模型对弈录像</h1>'
    if index.count(heading) != 1:
        raise ValueError('The existing Pages replay index has no expected insertion point')
    return index.replace(heading, heading + '\n' + section, 1)


def publish(token, filename, document, player, final):
    player_sha = blob(token, player)
    for _ in range(3):
        head = request(token, '/git/ref/heads/main')['object']['sha']
        tree = request(token, f'/git/commits/{head}')['tree']['sha']
        index = request(token, f'/contents/{DIRECTORY}index.html?ref={head}')
        raw_index = base64.b64decode(index['content']).decode()
        new_index = render_index(raw_index, filename, document, final).encode()
        index_sha = blob(token, new_index)
        changes = [
            {'path': DIRECTORY + name, 'mode': '100644', 'type': 'blob', 'sha': sha}
            for name, sha in [('index.html', index_sha), (filename, player_sha)]
        ]
        new_tree = request(token, '/git/trees', {'base_tree': tree, 'tree': changes})['sha']
        if new_tree == tree:
            return head, new_index
        commit = request(token, '/git/commits', {
            'message': f'Publish DCU training replay {document["candidate"]["update"]}',
            'tree': new_tree, 'parents': [head],
        })['sha']
        try:
            request(token, '/git/refs/heads/main', {'sha': commit, 'force': False}, 'PATCH')
        except HTTPError as error:
            if error.code not in (409, 422):
                raise
            if request(token, '/git/ref/heads/main')['object']['sha'] == head:
                raise
            continue
        return commit, new_index
    raise RuntimeError('GitHub main moved during all publish attempts')


def verify_public(filename, player, index, commit):
    until = time.monotonic() + 780
    while time.monotonic() < until:
        try:
            with urlopen(f'{PAGES}{filename}?rev={commit}', timeout=15) as response:
                public_player = response.read()
            with urlopen(f'{PAGES}?rev={commit}', timeout=15) as response:
                public_index = response.read()
            if public_player == player and public_index == index:
                return
        except (URLError, TimeoutError):
            pass
        time.sleep(15)
    raise RuntimeError('Published commit has not appeared on GitHub Pages')


def export(run, out, checkpoint, first):
    result = out / 'RESULT.json'
    if not result.exists() and first:
        until = time.monotonic() + 300
        while not result.exists() and time.monotonic() < until:
            time.sleep(5)
    if not result.exists():
        if out.exists():
            raise RuntimeError(f'Existing export did not finish: {out.name}')
        with out.with_suffix('.log').open('xb') as log:
            subprocess.run(
                [sys.executable, '-u', '-m', 'scripts.export_dcu_kill_replays',
                 '--checkpoint', str(run / checkpoint), '--out', str(out),
                 '--max-seconds', '600'], stdout=log, stderr=subprocess.STDOUT,
                cwd=ROOT, check=True, timeout=660,
            )
    report = read(result)
    if report['status'] != 'PASS_CPU_FULL_GAME_EXPORT' or report['games'] != 12:
        raise ValueError('Expected 12 complete fixed-opponent games')
    player = (out / 'watch.html').read_bytes()
    if hashlib.sha256(player).hexdigest() != report['files']['watch.html']['sha256']:
        raise ValueError('Replay player digest differs from the export receipt')
    subprocess.run([str(NODE), 'scripts/verify_dcu_kill_replays.js', str(out / 'replays.json')],
                   cwd=ROOT, check=True, timeout=120)
    if first:
        browser_env = os.environ.copy()
        browser_env.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(ROOT / '.tools/replay-preview/browsers'))
        subprocess.run([str(NODE), 'scripts/verify_dcu_kill_replay_browser.js',
                        (out / 'watch.html').as_uri(), str(out)],
                       cwd=ROOT, env=browser_env, check=True, timeout=180)
    return read(out / 'replays.json'), player


def run_schedule(args, token):
    plan = read(args.run / 'PLAN.json')
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    next_due = args.first_utc.timestamp()
    last_checkpoint = None
    first = True
    while time.time() < deadline + 600:
        final = (args.run / 'STOPPED.json').exists()
        if not final and time.time() < next_due:
            time.sleep(min(15, max(0, next_due - time.time())))
            continue
        if not final and time.time() >= deadline:
            time.sleep(10)
            continue
        checkpoint = args.first if first and not final else read(args.run / 'LATEST.json')['checkpoint']
        if not re.fullmatch(r'u\d{6,}', checkpoint):
            raise ValueError('Unexpected checkpoint name')
        if checkpoint != last_checkpoint or final:
            out = args.out / checkpoint
            try:
                status('snapshot', checkpoint=checkpoint, final=final)
                document, player = export(args.run, out, checkpoint, first and checkpoint == args.first)
                if document['candidate']['update'] != checkpoint:
                    raise ValueError('Replay checkpoint mismatch')
                filename = f'training-{checkpoint}-20261121.html'
                commit, index = publish(token, filename, document, player, final)
                status('committed', checkpoint=checkpoint, commit=commit, url=PAGES + filename)
                verify_public(filename, player, index, commit)
                receipt = {'status': 'VERIFIED_PUBLIC', 'checkpoint': checkpoint, 'final': final,
                           'commit': commit, 'url': PAGES + filename,
                           'sha256': hashlib.sha256(player).hexdigest()}
                (out / 'PAGES_PUBLISH.json').write_text(json.dumps(receipt, indent=2) + '\n')
                status('published', **receipt)
                last_checkpoint = checkpoint
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                status('publish_error', checkpoint=checkpoint, detail=str(error), final=final)
                if final or first:
                    time.sleep(30)
                    continue
            if final:
                return
        first = False
        while next_due <= time.time():
            next_due += 1800
        next_due = min(next_due, deadline)
        status('waiting', next_utc=datetime.fromtimestamp(next_due, timezone.utc).isoformat())
    raise RuntimeError('Replay schedule ended without a verified final publication')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--first', required=True)
    parser.add_argument('--first-utc', type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    if args.first_utc.tzinfo is None:
        parser.error('--first-utc requires a timezone')
    args.run = args.run.resolve(strict=True)
    args.out = args.out.resolve(strict=True)
    token = sys.stdin.readline().strip()
    if not token:
        parser.error('GitHub credential must be provided through stdin')
    os.umask(0o077)
    with (args.out / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_schedule(args, token)


if __name__ == '__main__':
    main()
