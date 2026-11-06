"""Render the dated HTML/SVG optimization timeline from reviewed numeric evidence."""
import argparse
import html
import json
import re
from pathlib import Path

DATE = '20261106'
OLD = Path('reports/mainline_timeline_20261028.html')
EVIDENCE = Path('reports/mainline_timeline_evidence_20261106.json')
DATA = Path('docs/assets/mainline_timeline_20261106.json')
SVG = Path('docs/assets/mainline_timeline_20261106.svg')
HTML = Path('reports/mainline_timeline_20261106.html')


def plain(value):
    value = re.sub(r'<span[^>]*>', ' · ', value)
    return re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]+>', '', value))).strip()


def legacy_nodes():
    source = OLD.read_text()
    starts = list(re.finditer(r'<div class="n ([^"]*)">', source))
    out = []
    for i, match in enumerate(starts):
        block = source[match.start():starts[i+1].start() if i+1 < len(starts) else source.index('<div class="note">')]
        fields = {k: plain(re.search(r'<div class="'+k+r'">(.*?)</div>', block).group(1)) for k in ('d','t','r','e')}
        fields['t'] = fields['t'].replace('当前最佳', '10-28最佳')
        out.append(dict(date=fields['d'], title=fields['t'], change=fields['r'].removeprefix('做了什么'),
                        effect=fields['e'].removeprefix('效果'), kind='pub' if match[1]=='best' else match[1],
                        source='mainline_timeline_20261028.html'))
        if fields['t'].startswith('JE10 r19'):
            out.append(dict(cut='10-19～10-25 支线未晋升；从 A 起换出生点与评估脚本，对拼数不与前段横比。'))
    return out


def nodes(evidence, online):
    original = evidence['strong_original']['G_frozen']['greedy']
    retest = evidence['strong_retest']['results']['L2_S05g_u019302']['greedy']['retest']
    sampled = evidence['strong_retest']['results']['L2_S05g_u019302']['categorical']['retest']
    return legacy_nodes() + [
        dict(cut='10-28 后：R1→R2 自由搬包/归属势差→R3 可见道具与本侧出生→R4 对拼；改用 bun-native-v2。'),
        dict(date='11-01',title='K1 u5282 · 已发布',kind='pub',
             change='在 R3-a u4002 上把对拼课程 10%→30%；保留三技能与本侧出生。',
             effect='P1 三档胜负差 +359（旧 +338）；hard holdout 111胜/9负/8平，旧113/10/5。',
             source='publish_k1_u5282_20261101.md'),
        dict(date='11-04',title='five-slot u9809 · 网页发布线',kind='pub',
             change='五槽独立技能/观测接入；配对补评后选 u9809，u9969 与 K1 保留回滚。',
             effect='v2 hard holdout P1 16胜/0负/0平；交付15、携包死0；因果杀5，旧K1为9。',
             source='web_model_promotion_20261104.md'),
        dict(cut='路线切换：以下看三技能 G 对拼训练线；u9809 是同期网页发布点，不能当作 G 的父 checkpoint。'),
        dict(date='11-04',title='G_frozen u18262 · 冻结强对手',kind='key',
             change='G 联赛完整状态续训；u18102→多区域30%/自然70%，保留真实 Bank 与原生地图。',
             effect='原强评估双座自对弈 argmax 25胜/25负/38平，score .500；采样 .386。',
             source='mainline_timeline_evidence_20261106.json'),
        dict(cut='强评估口径：对手固定 G_frozen argmax；44场景×双座，greedy 与 categorical 分列，不能与猎手胜数横比。'),
        dict(date='11-05',title='CC/CX 联赛 v2 · peer 对手 + 自模仿',kind='',
             change='current/ring/固定历史 + 活跃 peer；SIL 只监督正优势放泡，bot floor 后续固定。',
             effect='早期 S05_sil u18662 对 G：P0 argmax 19胜/8负/17平；单座探索，不当双座确认。',
             source='mainline_timeline_evidence_20261106.json'),
        dict(date='11-06',title='L2_S05g u019302 · 已复测的强评估最佳',kind='best',
             change='从 L2_S05 u19142 换 seed，SIL=.5、固定 bot floor；同 checkpoint 换新 seed 复测。',
             effect=f"原 argmax {original['wins']}/{original['losses']}/{original['draws']}，score {original['score']:.4f}；复测 {retest['wins']}/{retest['losses']}/{retest['draws']}，score {retest['score']:.4f}。",
             extra=f"复测 categorical {sampled['wins']}/{sampled['losses']}/{sampled['draws']}，score {sampled['score']:.4f}；同批 G 为 .5227，不能称全模式提升。",
             source='mainline_timeline_evidence_20261106.json'),
        dict(date='11-06',title='u019382 / u019462 · 同配方复现',kind='',
             change='L2_S05i/j 从健康 u019302 换独立 seed，保留奖励、权重与完整 Actor/Critic/Adam。',
             effect='独立候选 holdout 尚待完整终态；FTN/ATK/SP 等支线未晋升，不能按最新 update 选模。',
             source='../docs/combat_league_v2_training_path_20261106.md'),
        dict(cut='网页快筛改为 v2：模型固定 P1，5 个实际网页对手；新物理与旧模拟器 SHA 分开，不复用成新局。'),
        dict(date='11-06',title='最新线上选模 · '+online,kind='pub',
             change='发布由独立 owner 完成能力、长静止与网页实测门；保留旧模型供回滚。',
             effect='截至本图核验，线上仍为 five-slot u9809；新候选未据不完整结果晋升。' if online=='待定' else '已发布身份以 README 与发布报告为准。',
             source='mainline_timeline_evidence_20261106.json'),
    ]


def wrap(text, maximum=76):
    out=[];line='';width=0
    for char in text:
        n=2 if ord(char)>255 else 1
        if width+n>maximum:out.append(line);line='';width=0
        line+=char;width+=n
    if line:out.append(line)
    return out


def render_svg(rows):
    y=122;parts=[]
    def text(x,y,value,size=17,color='#334155',weight='normal'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(value)}</text>')
    for row in rows:
        if 'cut' in row:
            lines=wrap(row['cut'],103)
            parts.append(f'<path d="M30 {y}H1050" stroke="#d99b28" stroke-dasharray="7 5"/>')
            for i,line in enumerate(lines):text(35,y+24+i*23,line,16,'#926415')
            y+=40+len(lines)*23;continue
        lines=[('做了什么  '+row['change'],'#475569'),('效果  '+row['effect'],'#14683c')]
        if row.get('extra'):lines.append((row['extra'],'#a34b20'))
        body=[(line,c) for value,c in lines for line in wrap(value,96)]
        h=55+len(body)*24
        color={'best':'#269856','key':'#d74c44','pub':'#d99723'}.get(row['kind'],'#387ab1')
        parts.append(f'<path d="M119 {y}V{y+h+20}" stroke="#cbd8e7" stroke-width="3"/>')
        parts.append(f'<circle cx="119" cy="{y+22}" r="7" fill="{color}"/>')
        parts.append(f'<rect x="144" y="{y}" width="901" height="{h}" rx="9" fill="white" stroke="{color if row["kind"]=="best" else "#dce3ec"}" stroke-width="{2 if row["kind"]=="best" else 1}"/>')
        text(32,y+28,row['date'],17,'#536277','bold')
        text(162,y+30,row['title'],19,'#172e49','bold')
        for i,(line,c) in enumerate(body):text(162,y+59+i*24,line,16,c)
        y+=h+18
    text(34,y+20,'读法：胜/负/平顺序；虚线表示评估或路线切换。支线不展开，未完成不记晋升。',17,'#536277')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{y+55}" viewBox="0 0 1080 {y+55}" role="img" aria-labelledby="title desc"><title id="title">主线优化与发布轨迹 · 2026-11-06</title><desc id="desc">it831 至 D、K1、五槽发布、G_frozen 和联赛v2最佳；不同评估口径以虚线分隔。</desc><rect width="100%" height="100%" fill="#f5f8fc"/><g font-family="Noto Sans CJK SC,Microsoft YaHei,PingFang SC,sans-serif"><text x="32" y="43" font-size="27" fill="#172e49" font-weight="bold">QQ堂 RL · 主线优化与发布轨迹</text><text x="34" y="77" font-size="17" fill="#536277">2026-11-06 · 最佳训练点与线上发布分列 · 数字可追溯原始评估</text>{"".join(parts)}</g></svg>\n'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--online',default='待定')
    args=parser.parse_args()
    evidence=json.loads(EVIDENCE.read_text());rows=nodes(evidence,args.online)
    DATA.parent.mkdir(parents=True,exist_ok=True)
    DATA.write_text(json.dumps(dict(date='2026-11-06',online=args.online,evidence='../../'+str(EVIDENCE),nodes=rows),indent=2,ensure_ascii=False)+'\n')
    SVG.write_text(render_svg(rows))
    style=re.search(r'<style>(.*?)</style>',OLD.read_text(),re.S).group(1)
    cards=[]
    for row in rows:
        if 'cut' in row:cards.append('<div class="cutline">'+html.escape(row['cut'])+'</div>');continue
        cards.append('<div class="n '+row['kind']+'"><div class="d">'+html.escape(row['date'])+'</div><div class="c"><div class="t">'+html.escape(row['title'])+'</div><div class="r"><b>做了什么</b>'+html.escape(row['change'])+'</div><div class="e"><b>效果</b>'+html.escape(row['effect'])+'</div>'+('<div class="r">'+html.escape(row['extra'])+'</div>' if row.get('extra') else '')+'<div class="r"><a href="'+html.escape(row['source'])+'">数据来源</a></div></div></div>')
    HTML.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>主线优化轨迹 2026-11-06</title><style>'+style+'</style></head><body><h1>QQ堂 RL 主线优化与发布轨迹：it831 → 联赛 v2</h1><div class="sub">2026-11-06 更新。保留旧主干，延续重要优化与发布里程碑；虚线标出评估口径和训练路线切换。<br>最佳训练点与线上发布分列；数字顺序为胜/负/平。<a href="mainline_timeline_evidence_20261106.json">证据摘录与原始SHA</a> · <a href="../docs/assets/mainline_timeline_20261106.svg">静态图</a></div><div class="tl">'+''.join(cards)+'</div><div class="note">旧对拼数为难/普/易胜局，不是胜/负/平。跨口径不画连续胜率曲线。FTN、ATK、SP及更早未晋升支线不展开。新线上模型待发布owner验收后回填；不把探索最大值称为显著改善。</div></body></html>\n')
    print(f'Rendered {sum("cut" not in row for row in rows)} milestones: {HTML}, {SVG}')


if __name__=='__main__':main()
