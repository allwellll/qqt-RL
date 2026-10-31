"""Reuse the accepted self-contained Review renderer, add five-model labels/HUD."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from html import escape
from scripts.population_run_state import atomic_json,sha

B=Path(os.environ.get('NEW_REPLAY_BASE','runs/new_model_replays_20261031'))
def read(p):return json.loads(Path(p).read_text())

def main():
    cfg=read(B/'MANIFEST.json')
    subprocess.run(['node','scripts/v2_replay_metrics_20261031.js',str(B)],check=True)
    subprocess.run(['node','scripts/new_model_replay_timeline_20261031.js'],check=True)
    cfg=read(B/'MANIFEST.json')
    subprocess.run([sys.executable,'-m','scripts.build_v2_candidate_replays_20261031'],
                   env=dict(os.environ,V2_REPLAY_BASE=str(B)),check=True)
    html=B/cfg['html'];text=html.read_text();package=read(B/'PACKAGE.json')
    rows=read(B/'OBSERVABLE_TIMELINE.json')['rows']
    for e,g,r in zip(package['entries'],cfg['games'],rows):
        e['label']=f"{g['model']['display_name']} · seed{g['seed']} · P{g['seat']} · {r['result']} · 送达{r['deliveries']} · 击杀{r['causal_kills']} · 泡{r['placed']} · 真连锁{r['true_chain_events']} · "+g['rule']
    dump=lambda x:json.dumps(x,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    a=text.index('const entries=')+len('const entries=');z=text.index(';const docs=',a)
    text=text[:a]+dump(package['entries'])+text[z:]
    independent=len({g['index'] for g in cfg['games']})
    text=re.sub(r'\d+条入口 / \d+个独立对局；[^<]+',f"{len(rows)}条入口 / {independent}个独立对局；五模型各固定hard清单前8局，再按seed补每席首1负／首1平；缺例与扫描范围在下表披露。",text,count=1)
    text=text.replace('候选复核中，尚未晋升或切换网页默认。','五组录像对比；这组示例不作胜率估计或网页替换依据。')
    if cfg.get('supplements_pending'):
        text=text.replace('五组录像对比；这组示例不作胜率估计或网页替换依据。','首批固定40局预览；负局／平局补集正在按顺序扫描，尚未完成。五组示例不作胜率估计或网页替换依据。')
    text=text.replace('每模型固定hard首局，均为P0；完整终局与原始每tick状态hash一致，不按结果挑选，不代表P1效果。','固定前8局含P0/P1；补负／平局按seed顺序选，不能用来估计胜负频率。')
    text=text.replace('R1逐局新指标','逐局路线／放泡指标')
    text=text.replace('P0振荡/持包停滞诊断时间轴（点击跳转；区段右端不含）','中文事件时间轴（点击跳转；观察到的事件，不推断意图）')
    text=text.replace("renderer.render(player.sim,player.sim.t*100,m);", "m.playerRoutes=[0,1].map(p=>({text:`泡数 ${player.sim.bombsCap[p]} · 威力 ${player.sim.blastCap[p]} · 速度 ${player.sim.spdG[p].toFixed(3)}`,value:null}));renderer.render(player.sim,player.sim.t*100,m);")
    text=text.replace('完整Review核验通过；与fresh256的胜负、放泡、击杀及终局逐项一致。','完整Review动作／状态hash／终局核验通过；胜负与逐局原始记录一致。')
    text=text.replace('重放不一致；原fresh256记录保留，不替换。','重放不一致；请查原始记录。')
    table=['<details class="meta"><summary>展开：模型、局目与真实事件统计（送达／击杀／放泡／真连锁／道具）</summary><p>模型为毛毛，对手为蓝色Hunter；双方头顶显示当前泡数、威力、实际速度。每帧100毫秒。生产JS Sim.step；真人frameStep未在本次核验范围。奖励未记录。</p><div style="overflow:auto"><table border="1" cellpadding="5"><thead><tr><th>入口／模型</th><th>seat／seed</th><th>胜负</th><th>送达</th><th>击杀</th><th>放泡</th><th>真连锁</th><th>蕉／胶释放</th><th>属性／蕉胶拾取</th><th>持包自踩蕉</th><th>选局</th></tr></thead><tbody>']
    for i,(g,r) in enumerate(zip(cfg['games'],rows)):
        table.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [f"{i+1} · {g['model']['display_name']}",f"P{g['seat']} / {g['seed']}",r['result'],r['deliveries'],r['causal_kills'],r['placed'],r['true_chain_events'],r['releases'],r['pickups'],r['own_banana_carry_contacts'],g['rule']])+'</tr>')
    table.append('</tbody></table></div><p>真连锁=原生爆炸触发射线可达的提前爆泡事件，独立同时爆炸不计；释放=实际放置而非请求，不等于道具净收益。重复入口不增加独立样本。旧R3终点与SP快照的训练量不同。</p>')
    for label,r in cfg['selection'].items():
        table.append('<p>'+escape(f"{cfg['models'][label]['display_name']}：按序扫描{r['scanned']}局；未找到的补例={r['missing']}。最多只扫描冻结256局。")+'</p>')
    table.append('</details>')
    text=text.replace('</header>',''.join(table)+'</header>',1)
    # Keep the actual player visible on first load; detailed tables remain usable.
    text=text.replace('<details class="meta" open>','<details class="meta">')
    html.write_text(text)
    package.update(html_sha256=sha(html),bytes=html.stat().st_size,entries=package['entries'],
                   independent_games=independent,models_uploaded=False,training_untouched=True)
    package['source_sha256']['scripts/build_new_model_replays_20261031.py']=sha(Path(__file__))
    atomic_json(B/'PACKAGE.json',package)
    print(json.dumps({'status':'PACKAGED','html':str(html),'entries':len(rows),'independent_games':independent,'bytes':html.stat().st_size}))

if __name__=='__main__':main()
