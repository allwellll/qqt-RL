"""Saved real collector-state audit; no invented historic reset counters."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,pickle,time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha
from scripts.bun_spawn_side import side_counts

B=Path('runs/v2_ownside_selection_20261031')
def read(p):return json.loads(Path(p).read_text())
def main():
    rows=[];files={}
    old_roots=[Path('runs/v2_fresh_selection_20261031'),Path('runs/bun_native_v2_20261030')]
    for root in old_roots:
        for f in sorted(root.glob('*spawns.json')):
            d=read(f)
            if 'spawn_cells' in d:
                rows.append(dict(kind='manifest',source=str(f),sha256=sha(f),**side_counts(d['spawn_cells'])))
    for family,root,slotsfile in [('R1','runs/bomb_threat_v1','SLOTS.json'),('R2','runs/bun_native_v2_20261030','R2_SLOTS.json'),('R3','runs/r3_items_route_20261030','R3_SLOTS.json')]:
        root=Path(root)
        if family=='R1':
            # Active slot manifest name is stable across prior scheduling revisions.
            candidates=[root/'SLOTS.json',root/'TRAIN_SLOTS.json',root/'ITER2_SLOTS.json']
            slotsfile=next((p.name for p in candidates if p.exists()),slotsfile)
        slots=read(root/slotsfile) if (root/slotsfile).exists() else {}
        if not slots:
            folders=sorted((root/'formal').glob('bt_slot*'),key=lambda p:p.stat().st_mtime,reverse=True)
            chosen={}
            for f in folders:
                slot=f.name.split('_c')[0]
                if slot not in chosen and (f/'updates.jsonl').exists():chosen[slot]={'run':f.name}
            slots=chosen
        for arm,e in slots.items():
            if not isinstance(e,dict) or 'run' not in e:continue
            folder=root/'formal'/e['run'];cps=sorted(folder.glob('u*/COMMITTED.json'))
            if not cps:continue
            cm=read(cps[-1]);cp=cps[-1].parent
            if not (cp/'state.pkl').exists():continue
            h=sha(cp/'state.pkl');assert h==cm['state_sha256']
            with (cp/'state.pkl').open('rb') as stream:d=pickle.load(stream)
            st=d['states'];ax=d['aux'];homes=np.floor(np.asarray(st.bun_spawn_pos).reshape(-1,2,2)).astype(int)
            lesson=np.asarray(st.lesson).reshape(-1)
            source=np.asarray(ax.get('provenance_source',np.full(lesson.size,-1))).reshape(-1)
            modes=np.asarray(ax.get('a3sr',{}).get('spawn_mode',np.full(lesson.size,-1))).reshape(-1)
            raw=B/'impact'/f'{family}_{arm}_u{cm["update"]:06d}.npz';raw.parent.mkdir(exist_ok=True)
            np.savez_compressed(raw,homes=homes,lesson=lesson,bank_source=source,random_brick_mode=modes)
            subsets={f'lesson_{int(k)}':side_counts(homes[lesson==k]) for k in np.unique(lesson)}
            for label,mask in [('genuine_bank',source>=0),('natural_or_designated',source<0),('random_brick',modes==1),('unchanged_layout',modes==0)]:
                if mask.any():subsets[label]=side_counts(homes[mask])
            rows.append(dict(kind='actual_saved_collector_snapshot',family=family,arm=arm,source=str(cp),state_sha256=h,
                update=cm['update'],**side_counts(homes),subsets=subsets,raw=str(raw),raw_sha256=sha(raw),
                limitation='snapshot episode-weighted, not historical reset-frequency estimator; bucket label was not persisted and overlapping geometry cannot identify it'))
            del d,st,ax
    atomic_json(B/'SPAWN_SIDE_IMPACT.json',dict(status='COMPLETE_WITH_EXPLICIT_COVERAGE_LIMIT',rows=rows,
        unavailable=['historic exact per-bucket reset counts: bucket id and birth RNG not stored'],unix=time.time()))
    lines=['# 出生侧影响与修复（20261031）','','旧 v2 选型与预览作废，仅留对照；网页原生分组出生无此错侧问题。P0列≤6、P1列≥8；列7单列为非本侧。',
        '实际训练以完整原子 checkpoint 中存储的 bun_spawn_pos 统计（本局复活家点），非当前走位；这是episode存续加权快照，不冒充全部历史reset率。Bank当前位置可能在敌半场，这是合法取包轨迹，不予传送。',
        '历史分桶ID/出生RNG未落盘，无法无歧义恢复每个重叠bucket的实际选择次数；按lesson、真实Bank/自然来源、随机去砖模式分列可复算分母，缺失项保留。','',
        '|来源|样本|P0错侧|P1错侧|双方本侧|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["source"]}|{r["samples"]}|{r["wrong_per_player"][0]}|{r["wrong_per_player"][1]}|{r["both_own"]}|')
    lines+=['','逐lesson和来源分层、原始数组与SHA见 `runs/v2_ownside_selection_20261031/SPAWN_SIDE_IMPACT.json`。',
        '修复为独立 opt-in `own-side-native-v1`：基础fresh各lesson按原生本队四点出生；随机去砖保留，随机出生候选限本队四点；真实Bank当前物理位置不变，只迁移历史错侧复活点。R1不改，R2/R3仅在完整chunk后继迁移，Actor/Critic/Adam/ring保留，首迁移明确重建env/aux/RNG，随后全state精确续接。',
        '新dev/holdout/fresh/coop采用互不相交新seed，16原生组合分块平衡；不声称256独立地图。旧结果不混入新选型。网页发布冻结直到新矩阵和原发布门槛全部通过。']
    Path('reports/spawn_side_fix_20261031.md').write_text('\n'.join(lines)+'\n')
    print('spawn impact saved',len(rows))
if __name__=='__main__':main()
