# convert_sly_to_yolo.py
# Converts Supervisely-format RELLIS-3D download → YOLO detection format
# Run from your project root: python convert_sly_to_yolo.py

import json, os, shutil, base64, zlib
import numpy as np
from pathlib import Path
from PIL import Image
import io

# RELLIS-3D class name → YOLO index
CLASS_MAP = {
    'grass': 0,    'tree': 1,     'bush': 2,     'dirt': 3,
    'mud': 4,      'concrete': 5, 'sky': 6,      'water': 7,
    'puddle': 8,   'pole': 9,     'fence': 10,   'barrier': 11,
    'log': 12,     'vehicle': 13, 'building': 14,'person': 15,
    'asphalt': 16, 'rubble': 17,  'void': 18
}


def decode_bitmap(data_b64, origin, img_w, img_h):
    """Decode Supervisely bitmap mask → YOLO bbox (cx, cy, w, h) normalised."""
    raw  = zlib.decompress(base64.b64decode(data_b64))
    mask = np.array(Image.open(io.BytesIO(raw)))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    ox, oy = origin          # [col_offset, row_offset]
    x1 = (ox + c0) / img_w
    y1 = (oy + r0) / img_h
    x2 = (ox + c1) / img_w
    y2 = (oy + r1) / img_h
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def pts_to_bbox(pts, img_w, img_h):
    """Convert list of [x, y] exterior points → YOLO bbox."""
    pts = np.array(pts)
    x1, y1 = pts.min(axis=0) / [img_w, img_h]
    x2, y2 = pts.max(axis=0) / [img_w, img_h]
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def convert(src: str, dst: str):
    src, dst = Path(src), Path(dst)

    print(f"\nConverting: {src}  →  {dst}\n")
    total_imgs = 0

    for split in ['train', 'val', 'test']:
        # Supervisely uses ann/ and img/ subdirectories
        ann_dir = src / split / 'ann'
        img_dir = src / split / 'img'

        # Fallback naming conventions
        if not ann_dir.exists():
            ann_dir = src / split / 'annotations'
            img_dir = src / split / 'images'
        if not ann_dir.exists():
            print(f"  [{split}] not found — skipping")
            continue

        out_imgs = dst / 'images' / split
        out_lbls = dst / 'labels' / split
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_lbls.mkdir(parents=True, exist_ok=True)

        ann_files = sorted(ann_dir.glob('*.json'))
        print(f"  [{split}]  {len(ann_files)} images found")

        skipped = 0
        for ann_file in ann_files:
            ann    = json.loads(ann_file.read_text())
            img_h  = ann['size']['height']
            img_w  = ann['size']['width']

            # ann filename is e.g. frame_00001.jpg.json  →  stem = frame_00001.jpg
            img_name = ann_file.stem
            stem     = Path(img_name).stem      # frame_00001

            # Copy image
            src_img = img_dir / img_name
            if src_img.exists():
                shutil.copy2(src_img, out_imgs / img_name)
            else:
                skipped += 1

            # Build YOLO label lines
            lines = []
            for obj in ann.get('objects', []):
                cls_name = obj.get('classTitle', '').lower()
                cls_idx  = CLASS_MAP.get(cls_name)
                if cls_idx is None:
                    continue                    # unknown class — skip

                gtype = obj.get('geometryType', '')
                bbox  = None

                if gtype == 'bitmap':
                    bbox = decode_bitmap(
                        obj['bitmap']['data'],
                        obj['bitmap']['origin'],
                        img_w, img_h
                    )

                elif gtype == 'rectangle':
                    ext  = obj['points']['exterior']
                    # rectangle exterior = [[x1,y1],[x2,y2]]
                    rect = [ext[0], [ext[1][0], ext[0][1]],
                            ext[1], [ext[0][0], ext[1][1]]]
                    bbox = pts_to_bbox(rect, img_w, img_h)

                elif gtype == 'polygon':
                    bbox = pts_to_bbox(obj['points']['exterior'], img_w, img_h)

                if bbox:
                    cx, cy, bw, bh = bbox
                    # Clamp to [0, 1] to guard against rounding errors
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (out_lbls / f"{stem}.txt").write_text('\n'.join(lines))

        total_imgs += len(ann_files)
        if skipped:
            print(f"           ⚠  {skipped} images missing from img/ — labels still written")
        print(f"           ✓  done")

    print(f"\n✅  Converted {total_imgs} images total")
    print(f"    Output → {dst}")
    print(f"\nNext step: make sure rellis.yaml points to:")
    print(f"    path: ./{dst}")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    convert(
        src='data/rellis_sly_real',   # ← folder you unzipped into
        dst='data/rellis'                # ← new YOLO-formatted output
    )
