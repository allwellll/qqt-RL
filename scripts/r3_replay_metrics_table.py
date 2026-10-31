"""Local replay metric table; missing replay URLs stay explicitly missing."""
import html
import json
from pathlib import Path


def main():
    out=Path('runs/r3_items_route_20261030');rows=[]
    headers=['模型/集/模式/席位','局数','胜/平/负','五类拾取','香蕉/胶释放','终局泡/威/速',
        '持包自踩蕉','完整滑行/省tick估算/被炸','胶敌触发/50tick击杀/取包/截尾','自踩胶',
        '两笼初终BFS/最少砖','自身因果通路砖','存活泡mean/P90/max','批次mean/max',
        '实际连锁/批次数','0泡空窗/存活tick','空爆/实际爆炸','完整回放']
    records=[]
    baseline=out/'BASELINE_COMPLETE.json'
    if baseline.exists():
        for model,cs in json.loads(baseline.read_text())['results'].items():
            for c,ss in cs.items():
                for s,v in ss.items():records.append((f'{model}/{c}/greedy/{s}',v))
    for p in sorted((out/'evaluation').glob('*/COMPLETE.json')):
        for c,ms in json.loads(p.read_text())['metrics'].items():
            for m,ss in ms.items():
                for s,v in ss.items():records.append((f'{p.parent.name}/{c}/{m}/{s}',v))
    for label,v in records:
        values=[label,v['games'],[v['wins'],v['draws'],v['losses']],v['pickup_events'],v['release_counts'],v['terminal_attributes'],
            v['carrying_own_banana_contacts'],[v['banana_completed'],v['banana_estimated_saved_ticks'],v['banana_blast_within30']],
            [v['enemy_glue_contacts'],v['glue_subsequent_kill'],v['glue_subsequent_pickup'],v['glue_incomplete_followup']],v['glue_self_contacts'],
            v['routes'],v['own_causal_shortest_path_bricks'],[v['live_bomb_mean'],v['live_bomb_p90'],v['live_bomb_max']],
            [v['live_batch_mean'],v['live_batch_max']],[v['chain_events'],v['chain_batch_counts']],
            [v['zero_bomb_ticks'],v['alive_ticks']], [v.get('empty_resolved_bombs','缺数据'),v.get('resolved_bombs','缺数据')],
            '本诊断未录制回放；逐局JSON不可冒称可播放录像']
        rows.append('<tr>'+''.join('<td>'+html.escape(json.dumps(x,ensure_ascii=False) if not isinstance(x,str) else x)+'</td>' for x in values)+'</tr>')
    page='<!doctype html><meta charset="utf-8"><title>R3 回放指标表（本地）</title><style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px;max-width:360px;overflow-wrap:anywhere}td{vertical-align:top}</style><h1>R3 回放指标表</h1><p>本地验收页，未发布。香蕉为持包自踩加速；省tick是同路线原生速度估算。胶后续结果为观察性，不是因果效果。终局三属性与路径按每局保留；Sim.step范围，不代表真人frameStep。</p><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    (out/'replay_metrics_table.html').write_text(page)
    print('runs/r3_items_route_20261030/replay_metrics_table.html',len(rows),'rows')


if __name__=='__main__':main()
