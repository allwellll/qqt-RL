"""Passive round14 receipts/report, updated without blocking training."""
import json
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round14 import BASE,OUT,ROOT,read


def write_report():
    from scripts.bomb_threat_round14_metrics import recount
    recount()
    a=read(OUT/'HOLDOUT_AUDIT.json');g=a['natural']['greedy'];cat=a['natural']['categorical']
    fresh=BASE/'fresh256_a3sr_u640'
    f=read(fresh/'RESULTS.json',{});progress=read(fresh/'PROGRESS.json',{})
    native=read(BASE/'eval/fresh256_a3sr_u640/native_hunter256_v1/NATIVE_HUNTER_SUMMARY.json',{})
    slots=read(BASE/'SLOTS.json',{});watch=read(BASE/'WATCHDOG_STATUS.json',{})
    incident=read(OUT/'DONOR_RETENTION_INCIDENT.json',{})
    lines=['# BOMB-THREAT round14（20261030）','',
        '本轮状态：第1项正式验收完成；第2项GPU2 fresh辅助完成、生产JS异步采集中；第3项GPU5固定预算续训；第4项精确u640复制因清理竞态阻塞；第5项仅观察。未晋升、未发布、未修改网页。',
        '', '## 1. 正式闸门与防刷泡', '',
        f"指定u640 holdout：A={a['gates']['relative']['status']}，B={a['gates']['absolute']['status']}；与编排方复算无差异。argmax {g['win']}/{g['draw']}/{g['loss']}，W-L +{g['win_minus_loss']}、击杀{g['causal_kills']}、死亡{g['deaths']}、泡/局{g['bombs_per_game']:.6f}、单方零泡{g['zero_bomb_games']}、平局击杀差+{g['draw_kill_difference']}、自伤/笼内击杀均0；categorical {cat['win']}/{cat['draw']}/{cat['loss']}。库存窗口取包26、同链交付10（10/26）；最终库存交付11不是同链交付分子。",
        f"威胁/拓扑/连锁聚合奖励占当前chunk总奖励{a['reward_monitor']['threat_reward_fraction_total']:.4%}，低于30%；它不是整条谱系或整局奖励占比。",
        '', '|分组|局数|实际泡总量|泡/局|因果击杀|', '|---|---:|---:|---:|---:|']
    for k,v in a['groups'].items():lines.append(f"|{k}|{v['games']}|{v['bombs']}|{v['mean']:.4f}|{v['causal_kills']}|")
    lines+=['', '逐局泡量（按seed顺序，完整seed/seat/ticks/击杀/死亡及分片SHA见HOLDOUT_AUDIT.json）：',
        '', ', '.join(str(r['placed']) for r in a['episodes']), '',
        '46次可归因首次控制/2881实际放泡=1.597%；45次因果击杀/2881=1.562%。这是聚合事件比率，不是逐泡命中概率。原始分片无逐泡命中ID，命中转化率列MISSING；不以全体199控制事件冒充己方命中。击杀局泡量更高，但也有无击杀高泡局，不能据均值排除无效堆泡。',
        '', '## 2. Fresh256与邻点', '',
        '冻结协议 `runs/bomb_threat_v1/fresh256_a3sr_u640/PROTOCOL.json`：hard新seed2026153900–2026154155，normal2026154900–2026155155，easy2026155900–2026156155；与dev/holdout及旧fresh不重叠。候选argmax/categorical各256、座位各128；父点及两网页历史参照同seed各argmax256，总3840局。沿用holdout规则和64出生几何池循环，不能称新地图泛化。老师关闭。B计数阈值同比扩到256，比例不变，缺数据不PASS。',
        f"CPU JS当前{progress.get('completed_shards',0)}/480分片；RESULTS collection={f.get('collection_status','RUNNING')}，最终B={f.get('status','INCOMPLETE')}。GPU2已优先领取fresh并完成三模式各256（含额外tau.5辅助），与JS引擎/对手不同，不替代B。邻点u600 holdout固定64×两模式请求已入原CPU队列，不择峰。"]
    for mode,m in native.get('natural_jax_hunter',{}).items():
        lines.append(f"- GPU2 {mode}：{m['win']}/{m['draw']}/{m['loss']}，泡/局{m['bombs_per_game']:.3f}，死亡{m['deaths']}，自伤{m['self_deaths']}；非JS闸门。")
    lines+=['', '## 3. 显式续训与恢复', '',
        '唯一变量为预算：u640→u1280，新增640更新=167772160真实环境步；每40更新原子存盘/异步评估，320更新chunk自动接续。lr7.5e-5、熵.005、env2048×128、minibatch8192、Bank.25、A3奖励及随机出生配方均不变；不重做Critic预热。donor next-step、Adam、完整env/aux/RNG、ring精确匹配后推进。',
        '来源state SHA `3d67aaab903b27c23f4018ea91b1ae62ded632c67d7da569b35b4aa6003234d7`；policy SHA `7c5c2d535218b38e5c61e263c23556f787e3a2784c3486b76b25862938fe015c`。u640 policy/原子marker/评估不覆盖。']
    for slot in ('5','7','1'):
        entry=slots[slot];run=entry['run'];out=BASE/'formal'/run
        p=read(out/'PROGRESS.json',{});r=p.get('last',{});accept=read(OUT/f'SLOT{int(slot):02d}_FIRST_FORMAL.json',{})
        lines.append(f"- slot{slot} `{run}`：u{p.get('iterations','pending')}，finite={r.get('finite')}，Actor/Critic L2={r.get('update_l2')}，env/s={262144/r['cycle_seconds'] if r.get('cycle_seconds') else None}；round14首正式验收={accept.get('status','PENDING')}。")
    lines+=['', '## 4. Slot07复制阻塞', '',
        '授权在slot07当前完整chunk原子边界复制指定u640完整状态，配方A3SR、独立target RNG在donor parity后rekey，旧线末状态保留。实现和回归已备妥，尚未执行复制。',
        '清理竞态：GPU5读取u640并验parity后保存u641/u642，旧常驻清理器按旧“已有真实更新即可释放source”规则删除u640 state.pkl。发现后已停清理并新增持久donor/替换线末点保护及测试。该遗漏本应在启动前处理；不声称精确复制已完成，不把u642标成u640。',
        f"事件状态：{incident.get('status','UNKNOWN')}；原始删除收据在 `runs/iter_pbt_v1/RETENTION_DELETIONS.jsonl`，本次收据 `runs/bomb_threat_v1/iter2/a3sr/round14/DONOR_RETENTION_INCIDENT.json`。当前仓库runs同文件大小副本搜索未找到；明确来源修订尚待用户答复，slot07当前健康训练保留。保护回归后清理器已重启并验证u641/u642和slot07最新原子state被保护。", '',
        '## 5. 观察与owner', '',
        'slot00指定最近4点W-L +6/+1/+3/−3、单方零泡8–10，记录退化趋势；本轮不对slot00/03/04/06改配方或复制。',
        f"watchdog本轮开始前148，观察数分钟未增长；本轮主动owner-only reload后{watch.get('restarts')}，owner_count={watch.get('owner_count')}。历史连续重启日志明确是被动A3S验收访问已清理state列表触发IndexError；当前retained_atomic逻辑已规避。slot01 c27的前两更新4481/4482是协议2-update smoke，不是训练崩溃；之后独立fresh恢复exact并持续正式更新。", '',
        '测试：round14/A3SR/PBT/闸门/fresh相关focused通过；清理保护9项通过，扩展调度/原生/闸门30项通过。训练worker未发信号；owner-only reload利用持久run锁收养在跑worker。未创建重复训练owner，未恢复旧O5/R2。', '',
        '复现：`.venv/bin/python -m scripts.bomb_threat_round14`（仅在donor state仍在时freeze）；`.venv/bin/python -m scripts.bomb_threat_a3sr_fresh`（按accepted SHA续接，不重跑完整分片）；`.venv/bin/python -m scripts.bomb_threat_round14_report`（被动回填报告）。完整续训命令与seed见对应protocol/LAUNCH，原子/恢复/首更新见round14目录。']
    target=ROOT/'reports/bomb_threat_v1_round14_20261030.md'
    tmp=target.with_suffix('.md.tmp');tmp.write_text('\n\n'.join(lines)+'\n');tmp.replace(target)
    atomic_json(OUT/'EXECUTION_STATUS.json',dict(status='TRAINING_AND_FRESH_RUNNING_SLOT07_COPY_BLOCKED',
        steps={'1':'COMPLETE','2':f.get('collection_status','RUNNING'),'3':'RUNNING','4':'BLOCKED_EXACT_DONOR_MISSING','5':'COMPLETE_OBSERVATION_ONLY'},
        report=str(target.relative_to(ROOT)),report_sha256=sha(target),fresh_completed_shards=progress.get('completed_shards',0),
        owner_count=watch.get('owner_count'),unix=time.time()))
    return target


if __name__=='__main__':print(write_report().relative_to(ROOT))
