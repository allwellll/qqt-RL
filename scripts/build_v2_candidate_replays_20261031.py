"""Reuse the Review renderer with the exact versioned v2 simulator."""
import os
from pathlib import Path
import shutil
from scripts.population_run_state import atomic_json,sha

B=Path(os.environ.get('V2_REPLAY_BASE','runs/v2_fresh_selection_20261031/replay'))
os.environ['BT_REPLAY_BASE']=str(B)
asset=B/'replay_assets';asset.mkdir(exist_ok=True)
shutil.copy2('../cx_publish_r1_1805_20261030/web/review_protocol.js',asset/'review_protocol.js')
shutil.copy2('../cx_publish_r1_1805_20261030/web/review_metrics.js',asset/'review_metrics.js')
source=Path('scripts/build_bomb_threat_a3sr_replay.py').read_text()
source=source.replace("'sim-stock-window-v1.js']","'sim-bun-native-v2.js']")
source=source.replace("'review_route.js','review_protocol.js'", "'review_route.js','review_metrics.js','review_protocol.js'")
source=source.replace("s=(Path('web')/name).read_text()","s=((B/'replay_assets/review_protocol.js') if name=='review_protocol.js' else (Path('web')/name)).read_text()")
source=source.replace("else (Path('web')/name)","else ((B/'replay_assets/review_metrics.js') if name=='review_metrics.js' else (Path('web')/name))")
page=Path('scripts/bomb_threat_best_replay_page.js').read_text()
page=page.replace('2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45',sha(Path('web/sim-bun-native-v2.js')))
page=page.replace(" diagnosticTimeline();chart();draw();", " $('spawn-identity').textContent = `规则 ${doc.meta.rule_version} / ${doc.meta.spawn_side_version || '旧清单'}；P0出生 ${JSON.stringify(doc.meta.spawn_cells?.[0])} → 己笼 ${JSON.stringify(doc.meta.cages?.[0])}；P1出生 ${JSON.stringify(doc.meta.spawn_cells?.[1])} → 己笼 ${JSON.stringify(doc.meta.cages?.[1])}（行,列；从本侧去敌笼取包再返本侧）`; diagnosticTimeline();chart();draw();")
page=page.replace('采集时间 ${doc.meta.recorded_at}', '采集时间 ${doc.meta.recorded_at}；出生规则 ${doc.meta.spawn_side_version}；双方出生 ${JSON.stringify(doc.meta.spawn_cells)}；队伍笼锚 ${JSON.stringify(doc.meta.cages)}')
assert 'spawn-identity' in page
source=source.replace("scripts.append(Path('scripts/bomb_threat_best_replay_page.js').read_text())", "scripts.append(versioned_page)")
source=source.replace('已通过用户批准的网页替换标准；不宣称逐项全面不差。',
 '固定v2矩阵胜出候选；线上默认切换另以部署验收为准，不宣称全面不差。' if B.name=='final_replay' else '候选复核中，尚未晋升或切换网页默认。')
source=source.replace('stock-window-v1','bun-native-v2')
source=source.replace('<details class="meta"><summary>核验模型与选局来源</summary>','<div id="spawn-identity" class="meta"></div><details class="meta"><summary>核验模型与选局来源</summary>')
source=source.replace("sha(Path('web')/f) for f in sources", "sha((asset/f) if f in ('review_protocol.js','review_metrics.js') else Path('web')/f) for f in sources")
source=source.replace('顺序前8局、最早胜负局、P1单方零泡与旧u800砖隔开同seed均按预声明规则选择。',
 '固定首8局；另外披露筛选各席最早2胜/1负及P1最早2个单方零泡。按seed选，不用于估计胜负频率。' if B.name=='final_replay' else '每模型固定hard首局，均为P0；完整终局与原始每tick状态hash一致，不按结果挑选，不代表P1效果。')
source=source.replace("print(str(dest),dest.stat().st_size)","print(str(dest),dest.stat().st_size)")
exec(compile(source,'<versioned Review renderer>','exec'),{'__name__':'__main__','asset':asset,'versioned_page':page})
import json
p=B/'PACKAGE.json';doc=json.loads(p.read_text());doc['source_sha256'][str(asset/'review_protocol.js')]=sha(asset/'review_protocol.js')
doc['source_sha256'].pop('web/review_protocol.js',None);atomic_json(p,doc)
doc['source_sha256'][str(asset/'review_metrics.js')]=sha(asset/'review_metrics.js')
doc['source_sha256'].pop('web/review_metrics.js',None);atomic_json(p,doc)
