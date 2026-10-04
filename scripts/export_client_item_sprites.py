#!/usr/bin/env python3
"""Export original QQTang scene-item sprites (object.pkg DIMG) to web PNG strips.

Source: QQTang-Local.zip (kuuhaku1314/qqtang v1.3.2) runtime/client-patched.
Usage: python scripts/export_client_item_sprites.py <client-patched root> [out_dir]
"""
import json
import struct
import sys
import zlib
from pathlib import Path

from PIL import Image

# web key -> object.pkg path. Scene IDs follow upstream battleengine/pickup.go
# and clientdata/sceneelement NativeBattleActionPickup.
SPRITES = {
    "bomb": "object\\item\\item1_stand.img",          # scene 1 bomb capacity
    "power": "object\\item\\item2_stand.img",         # scene 2 bomb power
    "speed": "object\\item\\item3_stand.img",         # scene 3 speed
    "random": "object\\item\\item5_stand.img",        # scene 5 random attribute
    "bomb_super": "object\\item\\item6_stand.img",    # scene 6
    "power_super": "object\\item\\item7_stand.img",   # scene 7
    "speed_super": "object\\item\\item8_stand.img",   # scene 8
    "banana_pickup": "object\\item\\item23_stand.img",  # scene 23 -> action 42
    "glue_pickup": "object\\item\\item25_stand.img",    # scene 25 -> action 43
    "fast_shoe": "object\\item\\Item47_stand.img",      # scene 47 fast movement
    "banana_field": "object\\item\\item42_stand.img",   # placed action 42
    "glue_field": "object\\item\\item43_stand.img",     # placed action 43
    "bun": "object\\item\\item11_stand.img",            # rule-3 bun
}


def read_archive(root):
    index = (root / "data" / "object.idx").read_bytes()
    version, count, _, _ = struct.unpack_from("<4I", index, 0)
    if version != 100:
        raise ValueError(f"object.idx version {version}")
    entries, offset = {}, 16
    for _ in range(count):
        (length,) = struct.unpack_from("<H", index, offset)
        offset += 2
        path = index[offset:offset + length].decode("gbk", "replace")
        offset += length
        _, pkg_offset, expanded, compressed = struct.unpack_from("<4I", index, offset)
        offset += 16
        entries[path.replace("/", "\\").lstrip("\\").lower()] = (pkg_offset, expanded, compressed)
    return entries


def decode_dimg(data, all_directions=False):
    if data[:8] != b"QQF\x1aDIMG":
        raise ValueError("not a QQF/DIMG resource")
    _, _, frame_count, directions = struct.unpack_from("<4I", data, 8)
    # Header (cx, cy) is the frame-space origin of the declared canvas; frames
    # are positioned relative to the cell origin, so canvas = frame - (cx, cy).
    cx, cy, width, height = struct.unpack_from("<2i2I", data, 24)
    offset, frames = 40, []
    for _ in range(frame_count):
        _, x, y, mode = struct.unpack_from("<IiiI", data, offset)
        offset += 16
        if mode == 0:
            frames.append((x, y, None))
            continue
        fw, fh, _ = struct.unpack_from("<3I", data, offset)
        offset += 12
        if fw == 0 and fh == 0:
            frames.append((x, y, None))
            continue
        n = fw * fh
        if mode in (3, 0x11000000):
            rgb = struct.unpack_from(f"<{n}H", data, offset)
            alpha = data[offset + 2 * n: offset + 3 * n]
            pixels = [((v >> 11) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31,
                       (min(a, 32) * 255 + 16) // 32) for v, a in zip(rgb, alpha)]
            offset += 3 * n
        elif mode == 8:
            raw = data[offset:offset + 4 * n]
            pixels = [(raw[i + 2], raw[i + 1], raw[i], raw[i + 3]) for i in range(0, 4 * n, 4)]
            offset += 4 * n
        elif mode == 16:
            raw = data[offset:offset + 3 * n]
            pixels = [(raw[i + 2], raw[i + 1], raw[i], 255) for i in range(0, 3 * n, 3)]
            offset += 3 * n
        else:
            raise ValueError(f"pixel mode {mode:#x}")
        image = Image.new("RGBA", (fw, fh))
        image.putdata(pixels)
        frames.append((x, y, image))
    return width, height, cx, cy, frames if all_directions else frames[: frame_count // directions]


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "web" / "assets" / "item"
    out.mkdir(parents=True, exist_ok=True)
    entries = read_archive(root)
    manifest = {}
    with open(root / "data" / "object.pkg", "rb") as pkg:
        for key, path in SPRITES.items():
            pkg_offset, expanded, compressed = entries[path.lower()]
            pkg.seek(pkg_offset)
            data = zlib.decompress(pkg.read(compressed))
            if len(data) != expanded:
                raise ValueError(f"{path} expanded to {len(data)}, want {expanded}")
            width, height, cx, cy, frames = decode_dimg(data)
            # Keep per-frame offsets so the native bobbing survives.
            strip = Image.new("RGBA", (width * len(frames), height))
            for index, (x, y, image) in enumerate(frames):
                if image is not None:
                    strip.alpha_composite(image, (index * width + x - cx, y - cy))
            strip.save(out / f"{key}.png", optimize=True)
            manifest[key] = {"file": f"assets/item/{key}.png", "w": width, "h": height, "ox": cx, "oy": cy,
                             "frames": len(frames), "source": path.replace("\\", "/")}
    (out / "items.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: (v["w"], v["h"], v["frames"]) for k, v in manifest.items()}))


if __name__ == "__main__":
    main()
