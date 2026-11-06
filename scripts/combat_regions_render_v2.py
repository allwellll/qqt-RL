"""Compact complete-episode GIFs from verified native action replay frames."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main(path):
    root=Path(path);rows=json.loads((root/'DRAWING_MANIFEST.json').read_text());out=Path('reports/combat_regions_replays_20261104');out.mkdir(exist_ok=True)
    index=[]
    for row in rows:
        d=json.loads(Path(row['frames']).read_text());meta=d['meta'];images=[]
        for frame in d['frames']:
            image=Image.new('RGB',(480,470),'#f7f9fc');draw=ImageDraw.Draw(image)
            draw.text((12,8),f"{row['region']} | frozen parent u17999 | P{meta['learner_seat']} {meta['policy']}",fill='#172033')
            draw.text((12,28),f"tick {frame['t']}/{row['ticks']} | 5x replay | red P0 / blue P1",fill='#172033')
            for r in range(13):
                for c in range(15):
                    i=r*15+c;box=(15+c*30,55+r*30,44+c*30,84+r*30)
                    color='#b9c5d5' if frame['wall'][i] else '#a77a4e' if frame['brick'][i] else '#edf2f8'
                    draw.rectangle(box,fill=color,outline='#dbe3ed')
                    if frame['fuse'][i]>0:draw.ellipse((20+c*30,60+r*30,39+c*30,79+r*30),fill='#8b5cf6')
                    if frame['fire'][i]:draw.rectangle((18+c*30,58+r*30,41+c*30,81+r*30),fill='#fbbf24')
            for p in (0,1):
                if not frame['alive'][p]:continue
                y,x=frame['pos'][2*p:2*p+2];cx,cy=15+x*30,55+y*30
                draw.ellipse((cx-9,cy-9,cx+9,cy+9),fill='#ef4444' if p==0 else '#2563eb',outline='white',width=2)
                if frame['trapped'][p]>0:draw.ellipse((cx-14,cy-14,cx+14,cy+14),outline='#06b6d4',width=3)
                if frame['carried'][p]>=0:draw.rectangle((cx-4,cy-4,cx+4,cy+4),fill='#fbbf24')
            status='draw/timeout' if frame['winner'] is None else f"winner P{frame['winner']}"
            draw.text((15,450),('complete: '+status) if frame['done'] else 'Full action sequence; sparse drawing every 5 ticks',fill='#172033')
            images.append(image)
        file=out/(row['region']+'_parent.gif');images[0].save(file,save_all=True,append_images=images[1:],duration=100,loop=0,optimize=True)
        index.append(dict(region=row['region'],gif=str(file),source_review=row['source_review'],verified=True,ticks=row['ticks'],winner=row['winner']))
    (out/'INDEX.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(index,indent=2))


if __name__=='__main__':main(sys.argv[1])
