"""Authorized immutable release, live verification, then publication-gated reboot."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha
from scripts.a3sr_publish_github import request

ROOT = Path(__file__).resolve().parents[1]
O = Path('runs/bomb_threat_v1/publish_best_20261030')
W = Path('../cx_publish_best_20261030')


def read(p):
    return json.loads(Path(p).read_text())


def git(*args):
    return subprocess.check_output(['git', '-C', str(W), *args]).decode().strip()


def run(args, logfile, env=None):
    with (O / logfile).open('ab') as f:
        subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def upload_replay(base, html):
    path = Path(base) / html
    raw = path.read_bytes()
    result_path = Path(base) / 'UPLOAD_VERIFIED.json'
    if result_path.exists() and read(result_path)['local_sha256'] == sha(path):
        return read(result_path)
    notify = Path(shutil.which('jdme-notify'))
    uploader = notify.parent.parent / 'oss.sh'
    assert uploader.exists(), 'Authorized existing OSS upload helper unavailable'
    result = subprocess.run([str(uploader), str(path)], env=dict(os.environ, PYTHON_BIN='/usr/bin/python3'),
                            capture_output=True, text=True, timeout=180, check=True)
    urls = {}
    for line in result.stdout.splitlines():
        if 'Location:' in line:
            label, url = line.split('Location:', 1)
            urls[label.strip()] = url.strip()
    external = next((v for k, v in urls.items() if 'INNER' not in k), None)
    assert external, 'Uploader did not provide external URL'
    internal = next((v for k, v in urls.items() if 'INNER' in k), None)
    public_alias = external.replace('.jdcloud-oss.com/', '.jcloudcs.com/').replace('http://', 'https://')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    verified_url = None
    for url in list(dict.fromkeys([public_alias, external] + ([internal] if internal else []))):
        try:
            with opener.open(url, timeout=60) as response:
                check, status = response.read(), response.status
            assert status == 200 and check == raw
            verified_url = url
            if url == public_alias:
                external = public_alias
            break
        except Exception:
            if url == internal or not internal:
                raise
    record = dict(status='HTTP_BYTE_VERIFIED', url=external, verification_url=verified_url,
                  local_sha256=sha(path), remote_sha256=hashlib.sha256(check).hexdigest(),
                  bytes=len(raw), http_status=status, byte_equal=True, unix=time.time())
    atomic_json(result_path, record)
    return record


def report(row, receipt, head, actions, replay, browser):
    choice = read(O / 'FINAL_SELECTION.json')
    cleanup = read(O / 'CLEANUP_RECEIPT_20261030.json')
    name = 'publish_best_' + row['id'].replace('bun392-', '') + '_20261030.md'
    path = Path('reports') / name
    lines = ['# 最好模型与新录像发布 — 20261030', '',
             '状态：发布、Release/tag、main快进、同SHA Pages、线上完整资产与浏览器P1整局、新OSS录像回读和播放核验全部通过。训练按用户重启后恢复授权继续运行，发布不触发停训。', '',
             f"固定来源 `{read(O / 'manifest.json')['source_ref']}`，policy SHA `{row['source_sha256']}`。", '',
             '候选按原1024局复核的时间窗冻结：u1280、slot00 u2400，以及窗口内完成三档fresh的slot03 u2400；slot04/07在截止前未完成，不纳入。三者统一生产JS猎手、stock-window-v1、argmax、老师关闭、同新seed，每档256局（P0/P1各128），总2304局。hard/normal/easy seed分别从2026193000/2026194000/2026195000起。64出生几何重复，不等于256独立地图；是预声明固定矩阵上的观测选择，不宣称显著性或真人泛化。', '',
             '| 模型 | 档位 | P0胜/平/负 | P1胜/平/负 | P1胜负差 | 自伤 |',
             '|---|---|---|---|---:|---:|']
    for label, tiers in choice['results'].items():
        for tier in ('hard', 'normal', 'easy'):
            x = tiers[tier]
            seats = x['by_seat']
            a, b = seats['P0'], seats['P1']
            lines.append(f"|{label}|{tier}|{a['win']}/{a['draw']}/{a['loss']}|{b['win']}/{b['draw']}/{b['loss']}|{b['win_minus_loss']:+d}|{x['self_deaths']}|")
    lines += ['', f"最终选择 **{choice['winner']}**，满足P1每档≥u1280−5、P0三档合计≥u1280−10、自伤不增；主指标P1三档胜负差最高。完整比较与来源SHA `{O}/FINAL_SELECTION.json`。", '',
              '已知短板：仍有单方零泡、振荡和持包受砖阻等失败；固定预算结果并不代表战斗/交付/全部指标一致改善。网页真人P0、模型P1，部署仍为argmax；规则保持stock-window-v1，nativeItems/nativeTrap开启。训练bf16与网页fp32差异单列，不声称逐位相同。', '',
              'JSON权重张量逐项精确，Python fp32/JS/ONNX/Chromium八组双座位fixture数值误差<0.002且masked argmax一致；实际误差保留在原始收据。npm test、test:review、diff检查、全追踪文件及待发资产隐私检查通过；只提交模型目录和对应目录断言，旧模型对象全部保留，训练/Review/规则源码没有夹带发布。', '',
              f"- JSON SHA `{row['sha256']}`；ONNX SHA `{row['onnx']['sha256']}`。",
              '- 旧u1280 JSON SHA `736bc0c2d458682d771879be0cca80e25b90779336440beb40cbd800de0b7fed`。',
              f"- 生产提交 `{head}`；tag `{receipt['tag']}`。",
              f"- [Release]({receipt['url']})；[同head Pages Actions]({actions['html_url']})：completed/success。",
              '- 线上 `https://allwellll.github.io/qqt-RL/`；build-info/catalog首条/完整模型字节hash均与本地及Release一致。',
              '- 线上sim SHA `2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45` 不变。', '',
              '网页目录第一模型为默认模型身份，UI初始对手仍为规则Bot；未改变鉴权或自动改为模型对手。线上Chromium通过既有选择事件加载第一模型，真实模型P1完成整局且所有动作finite。线上数据库写入屏蔽。完整P1验收使用从真实Pages下载并核验SHA的模型缓存；页面、脚本、目录和ORT runtime仍来自线上，不冒称未缓存冷网络加载性能通过。', '',
              f"线上实战：{browser['evidence']['ticks']} tick，done={browser['evidence']['done']}，推理动作{len(browser['evidence']['actions'])}次，backend={browser['evidence']['backend']}；console/pageerror均无新错误。", '',
              '## 新录像', '',
              f"新页面 `{replay['html']}`，旧页保留。固定首8局与u800/u1280同seed；每席最早3胜1负、最早2个P1单方零泡、u800隔砖清单前3局同seed。录制使用旧固定hard清单，明确不替代新seed发布选择数据。完整Review动作/hash/终局、逐局原始结果一致、浏览器播放和终局seek通过；奖励未记录则明确显示未记录。外网URL保存在 `{replay['base']}/RECEIPT.json`，追踪报告使用中性占位 `<REPLAY_URL>`。", '',
              '| 入口 | seed | seat | 胜负 | 因果击杀 | 放泡 | 交付 | 选局依据 |',
              '|---:|---:|---|---|---:|---:|---:|---|']
    cfg = read(Path(replay['base']) / 'MANIFEST.json')
    for index, item in enumerate(cfg['games'], 1):
        e, seat = item['expected'], item['seat']
        outcome = '平' if e['winner'] in (None, -1) else '胜' if e['winner'] == seat else '负'
        lines.append(f"|{index}|{e['seed']}|P{seat}|{outcome}|{e['causal_kills'][seat]}|{e['placed'][seat]}|{e['primary_delivery']}|{item['rule']}|")
    lines += ['', f"录像HTML SHA `{read(Path(replay['base']) / 'PACKAGE.json')['html_sha256']}`，上传后HTTP200回读字节和SHA一致。", '',
              '## 清理与复现', '',
              '清理先清单、小批量、健康验收、全量，删除3978个完成评估的重复导出和51个历史state，合计168305000072 bytes。全部policy、COMMITTED、outcome/partial/trace/日志保留；最新完整state和R1共同起点保留。少量原清理器硬保护/历史watcher重新pin的旧state保守保留，未为清理中断训练。首批清理已结束；恢复后的追加清理单独登记，不混计释放量。逐文件清单、剩余state、目录大小见 `CLEANUP_RECEIPT_20261030.json`。', '',
              f"技术/选择/部署所有原始证据：`{O}`。复现选择 `<PYTHON> -m scripts.best_publish_selection_20261030` 和 `<PYTHON> -m scripts.best_publish_eligible_20261030`（已完成分片按hash复用）；录像 `<PYTHON> -m scripts.bomb_threat_a3sr_replay_owner` 配合 `BT_REPLAY_BASE`；发布检查 `scripts/verify_bun_release.py`、`scripts/verify_a3sr_publication_browser.js`。训练恢复验收另见 `reports/reboot_resume_20261030.md`。", '']
    path.write_text('\n'.join(lines))
    return path


def main():
    guard = lock(O / 'RELEASE_OWNER.lock')
    while not (O / 'PREPUBLICATION_ACCEPTANCE.json').exists():
        if (O / 'PREPUBLICATION_FAILURE.json').exists():
            atomic_json(O / 'RELEASE_STATUS.json', dict(status='WAITING_PREPUBLICATION_REPAIR', unix=time.time()))
        time.sleep(5)
    acceptance = read(O / 'PREPUBLICATION_ACCEPTANCE.json')
    if acceptance['status'] != 'PASS':
        raise RuntimeError('Retained baseline path requires explicit live baseline/replay revalidation; not a new release')
    row = read(O / 'CATALOG_ROW.json')
    manifest = read(O / 'manifest.json')
    tag = manifest['tag']
    env = dict(os.environ, BT_PUBLICATION_BASE=str(O), BT_PUBLICATION_WORKTREE=str(W), BT_RELEASE_TAG=tag,
               CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if not (O / 'LOCAL_COMMIT.json').exists():
        changed = git('diff', '--name-only').splitlines()
        assert sorted(changed) == ['web/models.json', 'web/test_http.js', 'web/test_model_catalog.js']
        base = git('rev-parse', 'HEAD')
        assert request('git/ref/heads/main')['object']['sha'] == base, 'Remote main advanced; preserve it and revalidate a new isolated base'
        git('add', '--', 'web/models.json', 'web/test_http.js', 'web/test_model_catalog.js')
        run(['git', '-C', str(W), '-c', 'user.name=QQT Model Release', '-c', 'user.email=model-release@users.noreply.github.com',
             'commit', '-m', 'Publish paired-validated ' + row['onnx']['meta']['name']], 'COMMIT.log')
        head = git('rev-parse', 'HEAD')
        atomic_json(O / 'LOCAL_COMMIT.json', dict(status='COMMITTED', base=base, head=head, changed_files=changed, unix=time.time()))
    head = read(O / 'LOCAL_COMMIT.json')['head']
    assert git('rev-parse', 'HEAD') == head
    tags = git('tag', '--list', tag)
    if not tags:
        run(['git', '-C', str(W), '-c', 'user.name=QQT Model Release', '-c', 'user.email=model-release@users.noreply.github.com',
             'tag', '-a', tag, head, '-m', row['onnx']['meta']['name'] + '; paired production-JS validation; stock-window-v1; raw argmax'], 'TAG.log')
    assert git('rev-list', '-n', '1', tag) == head
    if not (O / 'GIT_OBJECT_IMPORT.json').exists():
        run([sys.executable, '-m', 'scripts.a3sr_git_transport', 'import'], 'GIT_IMPORT.log', env)
    scores = read(O / 'FINAL_SELECTION.json')['comparison']
    (O / 'RELEASE.md').write_text(row['description'] + '\n\n' +
        '按预声明候选时间窗及同新seed矩阵选择；不宣称统计显著或真人泛化。旧模型保留，stock-window-v1和argmax不改。\n\n' +
        f"Policy SHA256: `{row['source_sha256']}`\nJSON SHA256: `{row['sha256']}`\nONNX SHA256: `{row['onnx']['sha256']}`\n\n" +
        'Python/JS/ONNX/Chromium一致性、全量网页和Review回归通过。源checkpoint为仓库相对路径，未包含训练代码、凭据或本机身份。\n')
    if not (O / 'RELEASE_RECEIPT.json').exists():
        run([sys.executable, '-m', 'scripts.a3sr_publish_github', 'release'], 'RELEASE_UPLOAD.log', env)
    if not (O / 'MAIN_PUSH.json').exists():
        run([sys.executable, '-m', 'scripts.a3sr_git_transport', 'main'], 'MAIN_PUSH.log', env)
    atomic_json(O / 'RELEASE_STATUS.json', dict(status='WAITING_SAME_HEAD_PAGES', head=head, unix=time.time()))
    deadline = time.monotonic() + 3600
    deployment = None
    while time.monotonic() < deadline:
        runs = request('actions/runs?head_sha=' + head + '&per_page=20')['workflow_runs']
        rows = [{k: r[k] for k in ('name', 'head_sha', 'status', 'conclusion', 'html_url')} for r in runs]
        atomic_json(O / 'ACTIONS.json', dict(head=head, runs=rows, unix=time.time()))
        pages = [r for r in rows if r['name'] == 'Deploy Bun rule-bot web to Pages']
        if pages and pages[0]['status'] == 'completed':
            assert pages[0]['conclusion'] == 'success', 'Same-head Pages failed'
            deployment = pages[0]
            break
        time.sleep(20)
    assert deployment, 'Same-head deployment did not finish within fixed deadline'
    if not (O / 'ONLINE_ASSETS.json').exists():
        run([sys.executable, '-m', 'scripts.a3sr_verify_online_assets'], 'ONLINE_ASSETS.log', env)
    assert read(O / 'ONLINE_ASSETS.json')['head'] == head
    if not (O / 'ONLINE_BROWSER_CACHED.json').exists():
        run(['taskset', '-c', '140-143', 'node', 'scripts/verify_a3sr_publication_browser.js'], 'ONLINE_BROWSER_CACHED.log',
            dict(env, WEB_URL='https://allwellll.github.io/qqt-RL/', EXPECTED_HEAD=head, VERIFIED_MODEL_CACHE='1'))
    browser = read(O / 'ONLINE_BROWSER_CACHED.json')
    assert browser['status'] == 'PASS' and browser['build']['commit'] == head
    replay = read(O / 'REPLAY_PREPARED.json')
    upload = upload_replay(replay['base'], replay['html'])
    remote_browser = Path(replay['base']) / 'REMOTE_BROWSER.json'
    if not remote_browser.exists():
        run(['taskset', '-c', '140-143', 'node', 'scripts/verify_bomb_threat_a3sr_replay_browser.js'], 'REPLAY_REMOTE_BROWSER.log',
            dict(env, BT_REPLAY_BASE=replay['base'], REPLAY_URL=upload['verification_url']))
    rb = read(remote_browser)
    assert rb['status'] == 'PASS' and rb['html_sha256'] == upload['remote_sha256']
    atomic_json(Path(replay['base']) / 'RECEIPT.json', dict(status='COMPLETED', url=upload['url'], html_sha256=upload['remote_sha256'],
                all_entries_full_review_verified=True, all_entries_terminal_seek=True, database_writes_blocked=True,
                entries=len(read(Path(replay['base']) / 'MANIFEST.json')['games']), unix=time.time()))
    receipt = read(O / 'RELEASE_RECEIPT.json')
    report_path = report(row, receipt, head, deployment, replay, browser)
    done = dict(status='COMPLETED', online_verified=True, replay_verified=True, head=head, tag=tag,
                default_model=row['id'], policy_sha256=row['source_sha256'], JSON_sha256=row['sha256'], ONNX_sha256=row['onnx']['sha256'],
                previous_JSON_sha256=read(O / 'ONLINE_CATALOG_BEFORE.json')['models'][0]['sha256'],
                release_url=receipt['url'], webpage_url='https://allwellll.github.io/qqt-RL/', replay_url=upload['url'],
                report=str(report_path), actions_url=deployment['html_url'], unix=time.time())
    atomic_json(O / 'DONE.json', done)
    with Path('docs/training_experiment_ledger.md').open('a') as f:
        f.write('\n\n### 20261030 最好模型发布终态\n\n'+row['id']+'，policy `'+row['source_sha256']+'`；固定候选配对选择，通过P1/P0/自伤护栏（局数与来源见完整选择JSON）。生产提交 `'+head+'`，tag `'+tag+'`，Pages/完整资产hash/线上P1完整局/新Review录像OSS回读播放全部PASS；旧模型/录像和stock-window-v1/argmax保留。只提交模型目录和目录断言。详见 `'+str(report_path)+'`。用户已在重启后明确恢复训练，发布完成后健康训练继续，不再安装旧停止闸门。\n')
    selected=read(O/'FINAL_SELECTION.json')['comparison']
    winner=read(O/'FINAL_SELECTION.json')['winner']
    subprocess.run(['jdme-notify', 'send', '最好模型与新录像发布核验完成：'+row['id']+'；P1合计'+str(selected[winner]['P1_sum'])+'，P0合计'+str(selected[winner]['P0_sum'])+'；tag '+tag+'；commit '+head+'；'+receipt['url']+'；录像 '+upload['url']+'。训练按恢复授权继续。'], capture_output=True, check=True)
    atomic_json(O / 'RELEASE_STATUS.json', dict(status='PUBLISHED_VERIFIED_TRAINING_CONTINUES', head=head, unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        message = str(error).replace(str(ROOT), '<WORKTREE>')
        atomic_json(O / 'RELEASE_FAILURE.json', dict(status='BLOCKED', reason=message, unix=time.time()))
        subprocess.run(['jdme-notify', 'send', '最好模型发布链技术阻塞：'+message[:240]+'。收据已写入，健康训练继续，保留现场待修复。'], capture_output=True)
        raise
