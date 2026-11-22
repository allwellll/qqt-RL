"""Bundle frozen simulator, production sprites and recorded actions in one HTML."""
import base64
import json
import mimetypes
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def bundle(document,output):
    web=ROOT/'web'
    level=next(x for x in json.loads((web/'assets/maps/levels.json').read_text()) if x.get('qqt_id')==806)
    meta={p:json.loads((web/p).read_text()) for p in ('assets/maps/elements.json','assets/item/items.json','assets/native/sprites.json')}
    elements,items,native=meta.values()
    ids={abs(v) for layer in level.get('layers',[]) for v in layer if v}
    files={level.get('bg','assets/bg/抢包子.png'),'assets/角色4×4精灵图.png','assets/角色c4×4.png','assets/bomb-custom/经典黄泡泡.png','assets/shadow.png','assets/point.png'}
    files.update('assets/flame/flame_'+n+'.png' for n in ['C_1','C_2']+[f'{d}_{i}' for d in 'UDLR' for i in range(1,7)])
    files.update(x['file'] for x in items.values())
    files.update(elements[str(i)]['file'] for i in ids if str(i) in elements)
    files.update(x['file'] for actor in native['actors'].values() for x in actor['actions'].values())
    files.update(x['file'] for x in native['effects'].values())
    images={f:'data:'+str(mimetypes.guess_type(f)[0])+';base64,'+base64.b64encode((web/f).read_bytes()).decode() for f in sorted(files)}
    sources=[]
    for name in ('sim-bun-native-v4-facing.js','kill_opening_items.js','self_respawn_preview.js','r3_model_inputs.js','visual_renderer.js'):
        source=(web/name).read_text()
        if name=='visual_renderer.js':
            assert source.count('image.src = src;')==1
            source=source.replace('image.src = src;','image.src = window.QQT_EMBED_IMAGES[src];')
        sources.append(base64.b64encode(source.encode()).decode())
    payload=dict(document=document,level=level,images=images,metadata=meta,sources=sources)
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    template=(web/'dcu_kill_replay_template.html').read_text()
    assert template.count('/*__PAYLOAD__*/null')==1
    Path(output).write_text(template.replace('/*__PAYLOAD__*/null',encoded))
