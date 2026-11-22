"""Publish timestamped courier/interceptor replays from an active DCU run."""
import argparse
import base64
from datetime import datetime, timezone
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


ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.github.com/repos/allwellll/qqt-RL'
PAGES = 'https://allwellll.github.io/qqt-RL/carry-replays/'
WEB_DIR = 'web/carry-replays/'
RUN_ID = 'carry-joint-u587-u104-20261122b'


def status(event, **fields):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **fields),
                     ensure_ascii=False), flush=True)


def api(token, path, body=None, method=None):
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28',
               'User-Agent': 'qqt-dcu-carry-replays'}
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        headers['Content-Type'] = 'application/json'
    with urlopen(Request(API + path, data=data, headers=headers, method=method), timeout=60) as response:
        return json.load(response)


def git_blob(token, data):
    return api(token, '/git/blobs', {
        'content': base64.b64encode(data).decode('ascii'), 'encoding': 'base64'})['sha']


def latest_checkpoint(root, stage, fallback=None):
    candidates = []
    for path in root.glob(stage + '-s*/LATEST.json'):
        data = json.loads(path.read_text())
        if not data.get('checkpoint') or int(data.get('update', 0)) < 1:
            continue
        directory = path.parent / data['checkpoint']
        if not (directory / 'actor.pt').is_file() or not (directory / 'critic.pt').is_file():
            continue
        candidates.append((path.stat().st_mtime_ns, directory))
    if not candidates:
        if fallback is not None:
            fallback = fallback.resolve(strict=True)
            if (fallback / 'actor.pt').is_file() and (fallback / 'critic.pt').is_file():
                return fallback
        raise FileNotFoundError('No durable ' + stage + ' checkpoint is available')
    return max(candidates)[1]


def render_player(template, replay_name, document):
    defender = document['models']['defender']
    courier = document['models']['courier']
    courier_duel = defender.get('role') == 'courier' and courier.get('role') == 'courier'
    title = (f"运包模型对弈 · 运包 u{defender['update']:06d} vs 运包 u{courier['update']:06d}"
             if courier_duel else
             f"运包训练录像 · 拦截 u{defender['update']:06d} vs 运包 u{courier['update']:06d}")
    page = template.replace('<title>运包对弈录像 · u104 vs SP_web</title>',
                            '<title>' + html.escape(title) + '</title>')
    page = page.replace('<h1>u104 拦截者 vs SP_web 运包模型</h1>',
                        '<h1>' + html.escape(title) + '</h1>')
    page = page.replace('<p>map806 完整 JAX 运包规则 · CPU FP32 推理 · 两个种子各交换一次出生侧</p>',
                        '<p>map806 完整运包规则 · CPU FP32 推理 · 固定种子并交换双方座位</p>')
    page = page.replace("fetch('u104-vs-spweb-20261121.json')", "fetch('" + replay_name + "')")
    page = page.replace("const role = seat === game.defender_seat ? '拦' : '运';",
                        "const role = data.models.defender.role === 'courier' && data.models.courier.role === 'courier' ? '运' : (seat === game.defender_seat ? '拦' : '运');")
    page = page.replace("|| `${sides[game.defender_seat]}拦截 · ${sides[1-game.defender_seat]}运包`;",
                        "|| (data.models.defender.role === 'courier' && data.models.courier.role === 'courier' ? `${sides[game.defender_seat]}运包 · ${sides[1-game.defender_seat]}运包` : `${sides[game.defender_seat]}拦截 · ${sides[1-game.defender_seat]}运包`);")
    page = page.replace("$('title').textContent = `种子 ${game.seed} · ${sides[game.defender_seat]}拦截`;",
                        "$('title').textContent = data.models.defender.role === 'courier' && data.models.courier.role === 'courier' ? `种子 ${game.seed} · 运包对弈` : `种子 ${game.seed} · ${sides[game.defender_seat]}拦截`;")
    page = page.replace("$('game').add(new Option(`种子 ${row.seed} · ${sides[row.defender_seat]}拦截`, index));",
                        "$('game').add(new Option(data.models.defender.role === 'courier' && data.models.courier.role === 'courier' ? `种子 ${row.seed} · 运包对弈` : `种子 ${row.seed} · ${sides[row.defender_seat]}拦截`, index));")
    page = page.replace("const values = [row.seed, sides[1-row.defender_seat], sides[row.defender_seat],",
                        "const values = [row.seed, sides[1-row.defender_seat], data.models.defender.role === 'courier' && data.models.courier.role === 'courier' ? '运包' : sides[row.defender_seat],")
    old = "$('source').textContent = `拦截者 u104 · ${data.models.defender.sha256} | SP_web u23302 · ${data.models.courier.sha256}`;"
    new = "$('source').textContent = (data.models.defender.role === 'courier' && data.models.courier.role === 'courier') ? `运包 u${String(data.models.defender.update).padStart(6, '0')} · ${data.models.defender.sha256} | 运包 u${String(data.models.courier.update).padStart(6, '0')} · ${data.models.courier.sha256}` : `拦截者 u${String(data.models.defender.update).padStart(6, '0')} · ${data.models.defender.sha256} | 运包 u${String(data.models.courier.update).padStart(6, '0')} · ${data.models.courier.sha256}`;"
    if old not in page:
        raise ValueError('Replay player template source marker changed')
    return page.replace(old, new).encode()


def render_directory(index, archive_html, defender_update, courier_update, stamp, courier_duel=False):
    marker = '<!-- carry-joint-replay-links:start -->'
    end = '<!-- carry-joint-replay-links:end -->'
    links = []
    if marker in index and end in index:
        body = index.split(marker, 1)[1].split(end, 1)[0]
        links = re.findall(r'<li>.*?</li>', body)
    links = [row for row in links if archive_html not in row]
    label = (f'运包 u{courier_update:06d} vs 运包 u{defender_update:06d}' if courier_duel
             else f'运包 u{courier_update:06d} vs 拦截 u{defender_update:06d}')
    links.insert(0, f'<li><a href="{html.escape(archive_html)}">{stamp} · {label}</a></li>')
    section = '\n'.join([marker, '<section id="carry-training-replays">',
        f'<p><a href="latest.html">播放最新训练录像：{label}</a></p>',
        '<p>交付胜利要求把对方包子运回己方包子笼；未交付均按平局展示。</p>',
        '<ul>', *links[:24], '</ul>', '</section>', end])
    if marker in index and end in index:
        return index.split(marker, 1)[0] + section + index.split(end, 1)[1]
    heading = '<h1>u104 拦截者 vs SP_web 运包模型</h1>'
    if index.count(heading) != 1:
        raise ValueError('Could not find insertion point in carry replay index')
    return index.replace(heading, heading + '\n' + section, 1)


def publish(token, archive_html, archive_json, payload, archive_player, latest_player,
            defender_update, courier_update, stamp):
    for _ in range(4):
        head = api(token, '/git/ref/heads/main')['object']['sha']
        base_tree = api(token, '/git/commits/' + head)['tree']['sha']
        entry = api(token, '/contents/' + WEB_DIR + 'index.html?ref=' + head)
        index = base64.b64decode(entry['content']).decode()
        document = json.loads(payload)
        courier_duel = (document['models']['defender'].get('role') == 'courier' and
                        document['models']['courier'].get('role') == 'courier')
        new_index = render_directory(index, archive_html, defender_update, courier_update,
                                     stamp, courier_duel=courier_duel).encode()
        changes = []
        files = {
            'index.html': new_index,
            'latest.html': latest_player,
            'latest.json': payload,
            archive_html: archive_player,
            archive_json: payload,
        }
        for name, data in files.items():
            changes.append({'path': WEB_DIR + name, 'mode': '100644', 'type': 'blob',
                            'sha': git_blob(token, data)})
        tree = api(token, '/git/trees', {'base_tree': base_tree, 'tree': changes})['sha']
        if tree == base_tree:
            return head
        commit = api(token, '/git/commits', {
            'message': f'Publish carry training replay courier u{courier_update:06d} defender u{defender_update:06d}',
            'tree': tree, 'parents': [head]})['sha']
        try:
            api(token, '/git/refs/heads/main', {'sha': commit, 'force': False}, 'PATCH')
            return commit
        except HTTPError as error:
            if error.code not in (409, 422) or api(token, '/git/ref/heads/main')['object']['sha'] == head:
                raise
    raise RuntimeError('GitHub main moved during all publish attempts')


def verify_public(filename, payload, latest_player, archive_player, commit):
    until = time.monotonic() + 780
    urls = [(PAGES + 'latest.json?rev=' + commit, payload),
            (PAGES + 'latest.html?rev=' + commit, latest_player),
            (PAGES + filename + '?rev=' + commit, archive_player)]
    while time.monotonic() < until:
        try:
            for url, expected in urls:
                with urlopen(url, timeout=20) as response:
                    if response.read() != expected:
                        raise URLError('Pages bytes do not match published asset')
            return
        except (URLError, TimeoutError):
            time.sleep(15)
    raise RuntimeError('Published replay has not appeared byte-for-byte on GitHub Pages')


def export_snapshot(root, out, template, fallback_courier=None, fallback_interceptor=None):
    courier = latest_checkpoint(root, 'courier', fallback_courier)
    defender = latest_checkpoint(root, 'interceptor', fallback_interceptor)
    courier_meta = json.loads((courier / 'CHECKPOINT.json').read_text())
    defender_meta = json.loads((defender / 'CHECKPOINT.json').read_text())
    key = f"c{courier_meta['update']:06d}-i{defender_meta['update']:06d}-{courier_meta['actor_sha256'][:10]}-{defender_meta['actor_sha256'][:10]}"
    payload_path = out / (key + '.json')
    if payload_path.exists():
        payload = payload_path.read_bytes()
    else:
        command = [sys.executable, '-u', '-m', 'scripts.export_carry_match_replay',
                   '--courier-checkpoint', str(courier), '--checkpoint', str(defender),
                   '--out', str(payload_path)]
        env = os.environ.copy()
        env.update(JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', JAX_ENABLE_COMPILATION_CACHE='false')
        with payload_path.with_suffix('.log').open('ab') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        payload = payload_path.read_bytes()
    document = json.loads(payload)
    if document['schema'] != 'qqt.full-jax-carry-replay/v1' or len(document['replays']) != 4:
        raise ValueError('Expected four complete courier/interceptor games')
    if any(row['result']['winner'] not in (-1, 0, 1) or
           len(row['frames']) != row['result']['tick'] + 1 for row in document['replays']):
        raise ValueError('Replay contains an incomplete game')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    courier_update = int(document['models']['courier']['update'])
    defender_update = int(document['models']['defender']['update'])
    archive = f'carry-joint-{stamp}'
    archive_html, archive_json = archive + '.html', archive + '.json'
    archive_player = render_player(template, archive_json, document)
    latest_player = render_player(template, 'latest.json', document)
    return (document, payload, archive_html, archive_json, archive_player, latest_player,
            courier_update, defender_update, stamp)


def publish_payload(token, payload, template):
    document = json.loads(payload)
    if document.get('schema') != 'qqt.full-jax-carry-replay/v1' or len(document.get('replays', [])) != 4:
        raise ValueError('Seed replay must contain four complete carry games')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    courier_update = int(document['models']['courier']['update'])
    defender_update = int(document['models']['defender']['update'])
    archive = f'carry-joint-{stamp}'
    archive_html, archive_json = archive + '.html', archive + '.json'
    archive_player = render_player(template, archive_json, document)
    latest_player = render_player(template, 'latest.json', document)
    commit = publish(token, archive_html, archive_json, payload, archive_player, latest_player,
                     defender_update, courier_update, stamp)
    verify_public(archive_html, payload, latest_player, archive_player, commit)
    receipt = dict(status='VERIFIED_PUBLIC', commit=commit,
                   url=PAGES + 'latest.html', archive_url=PAGES + archive_html,
                   courier_update=courier_update, interceptor_update=defender_update,
                   games=len(document['replays']), sha256=hashlib.sha256(payload).hexdigest())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=int, default=1800)
    parser.add_argument('--final-stage', default='interceptor-s06',
                        help='last interceptor stage; publisher exits after it finishes')
    parser.add_argument('--fallback-courier', type=Path,
                        help='checkpoint used until this run produces its first courier checkpoint')
    parser.add_argument('--fallback-interceptor', type=Path,
                        help='checkpoint used until this run produces its first interceptor checkpoint')
    parser.add_argument('--seed-json', type=Path,
                        help='Publish a previously validated replay before starting the interval')
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    template = (ROOT / 'web/carry-replays/index.html').read_text()
    token = sys.stdin.readline().strip()
    if not token:
        parser.error('GitHub credential must be provided through stdin')
    os.umask(0o077)
    due = time.monotonic()
    if args.seed_json:
        receipt = publish_payload(token, args.seed_json.read_bytes(), template)
        name = datetime.now(timezone.utc).strftime('seed-%Y%m%dT%H%M%SZ.publish.json')
        (out / name).write_text(json.dumps(receipt, indent=2) + '\n')
        status('published_seed', **receipt)
        due += args.interval_seconds
    final_stage = root / args.final_stage
    while True:
        if time.monotonic() < due:
            time.sleep(min(15, due - time.monotonic()))
            continue
        try:
            result = export_snapshot(root, out, template,
                                     args.fallback_courier, args.fallback_interceptor)
            document, payload, archive_html, archive_json, archive_player, latest_player, courier_u, defender_u, stamp = result
            commit = publish(token, archive_html, archive_json, payload, archive_player, latest_player,
                             defender_u, courier_u, stamp)
            verify_public(archive_html, payload, latest_player, archive_player, commit)
            receipt = dict(status='VERIFIED_PUBLIC', commit=commit,
                           url=PAGES + 'latest.html', archive_url=PAGES + archive_html,
                           courier_update=courier_u, interceptor_update=defender_u,
                           games=len(document['replays']),
                           sha256=hashlib.sha256(payload).hexdigest())
            (out / (archive_html + '.publish.json')).write_text(json.dumps(receipt, indent=2) + '\n')
            status('published', **receipt)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, URLError) as error:
            status('publish_error', detail=str(error))
        if (final_stage / 'RESULT.json').exists() and (final_stage / 'STOPPED.json').exists():
            return
        due += args.interval_seconds
        while due <= time.monotonic():
            due += args.interval_seconds
        status('waiting', next_in_seconds=round(due - time.monotonic()))


if __name__ == '__main__':
    main()
