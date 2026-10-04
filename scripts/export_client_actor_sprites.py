#!/usr/bin/env python3
"""Export native layered actors and syrup effects from the v1.3.2 client."""
import configparser
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

from PIL import Image

from export_client_item_sprites import decode_dimg, read_archive


def main():
    root = Path(sys.argv[1])
    out = Path(__file__).resolve().parents[1] / 'web/assets/native'
    out.mkdir(parents=True, exist_ok=True)
    entries = read_archive(root)
    manifest = {'source': 'kuuhaku1314/qqtang v1.3.2 QQTang-Local.zip', 'actors': {}, 'effects': {}}
    zorder = configparser.ConfigParser()
    zorder.read(root / 'object/player/player_z.ini', encoding='gb18030')
    with (root / 'data/object.pkg').open('rb') as pkg:
        def decode(name):
            offset, expanded, compressed = entries[name.lower()]
            pkg.seek(offset)
            data = zlib.decompress(pkg.read(compressed))
            if len(data) != expanded:
                raise ValueError(name)
            directions = struct.unpack_from('<I', data, 20)[0]
            return decode_dimg(data, all_directions=True), directions

        for role, key, label in [(1, 'pipi', '皮皮'), (9, 'maomao', '毛毛')]:
            config = configparser.ConfigParser()
            config.read(root / f'object/player/player{role}.ini', encoding='gb18030')
            actor = {'roleId': role, 'label': label, 'actions': {}}
            for action in ('stand', 'walk'):
                layers = {}
                for part, value in config[action].items():
                    name = f'object\\{part}\\{part}{value}_{action}.img'
                    if name.lower() not in entries:
                        continue
                    (w, h, cx, cy, frames), dirs = decode(name)
                    layers[part] = (frames, len(frames) // dirs)
                count = max(n for _, n in layers.values())
                strip = Image.new('RGBA', (85 * count, 340))
                # Native frames face right/up/left/down; renderer rows are down/left/right/up.
                for direction, suffix in enumerate(('down', 'left', 'right', 'up')):
                    source_direction = (3, 2, 0, 1)[direction]
                    order = sorted(layers, key=lambda p: int(zorder[action].get(f'{p}_z_{suffix}', 0)))
                    for frame in range(count):
                        canvas = Image.new('RGBA', (100, 100))
                        for part in order:
                            frames, n = layers[part]
                            x, y, image = frames[source_direction * n + frame % n]
                            if image is not None:
                                canvas.alpha_composite(image, (x, y))
                        strip.alpha_composite(canvas.crop((7, -2, 92, 83)), (frame * 85, direction * 85))
                filename = f'{key}_{action}.png'
                strip.save(out / filename, optimize=True)
                actor['actions'][action] = {'file': f'assets/native/{filename}', 'w': 85, 'h': 85,
                                            'frames': count, 'directions': 4}
                if action == 'stand':
                    portrait = strip.crop((0, 0, 85, 85))
                    portrait = portrait.crop(portrait.getbbox())
                    portrait.save(out / f'{key}_portrait.png', optimize=True)
            manifest['actors'][key] = actor
        for key, name in [('trap', 'misc111_trigger'), ('pop', 'misc111_die')]:
            (w, h, cx, cy, frames), _ = decode(f'object\\misc\\{name}.img')
            strip = Image.new('RGBA', (w * len(frames), h))
            for i, (x, y, image) in enumerate(frames):
                if image is not None:
                    strip.alpha_composite(image, (i * w + x - cx, y - cy))
            strip.save(out / f'{key}.png', optimize=True)
            manifest['effects'][key] = {'file': f'assets/native/{key}.png', 'w': w, 'h': h,
                                        'frames': len(frames), 'source': f'object/misc/{name}.img'}
        sound = root / 'sound/X12_01.wav'
        (out / 'syrup_pop.wav').write_bytes(sound.read_bytes())
        manifest['sound'] = {'file': 'assets/native/syrup_pop.wav', 'source': 'sound/X12_01.wav',
                             'sha256': hashlib.sha256(sound.read_bytes()).hexdigest()}
    (out / 'sprites.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
