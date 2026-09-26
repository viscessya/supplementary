# -*- coding: utf-8 -*-
"""potong_dan_hapus_latar.py -- smartphone photographs -> four-view input for Hunyuan3D-2mv (Section 4.4).

Steps, per view (front/left/back/right):
  1. Coarse mask of the whole photograph with BiRefNet-general-lite (ONNX), via segmentasi_birefnet_lite.py.
     Largest connected component -> bounding box of the pump.
  2. One square side for all four views: S = round(1.5 x the longest bounding-box side over the four views).
     The square is centred on the pump and shifted, if needed, so it stays inside the photograph (no padding).
     Crop, resize to 1024 x 1024 (Lanczos) -> latar_asli/<view>.png (real background).
  3. Fine mask of the 1024 crop with the same model. Threshold 0.5 -> largest component -> fill enclosed holes
     (e.g. the fan-cover slots, which are not see-through). Soft BiRefNet alpha is kept only in a 2-pixel band at
     the edge; the interior is opaque. Composite on white (255) -> latar_putih/<view>.png (Hunyuan3D-2mv input).

Provenance: on 26 Sep 2026 these steps were first run interactively. This script re-implements them. Fed the
coarse and fine masks of that run (--masker-kasar, --masker-halus), it reproduces the eight committed 1024 x 1024
images in inputs/ pixel for pixel; a fresh BiRefNet run on the same crop also returned a bit-identical mask.

Memory: on a 7-8 GB machine, BiRefNet on a full 4032 x 3024 photograph can be killed for lack of memory.
Photographs listed in --setengah are segmented from a half-resolution copy (coarse mask only); in the run behind
the paper this applied to IMG_4011.

Usage:
  python potong_dan_hapus_latar.py --model BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx \
      --foto <folder with IMG_*.png> --out <folder> \
      --view front=IMG_4013 left=IMG_4011 back=IMG_4012 right=IMG_4010 --setengah IMG_4011
(HEIC originals were converted to PNG first with pillow-heif, applying the EXIF orientation.)
"""
import argparse, os, subprocess, sys, tempfile, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, "segmentasi_birefnet_lite.py")

def masker(model, src, dst):
    for _ in range(6):  # a memory-killed run (exit -9) is retried; the result is deterministic
        r = subprocess.run([sys.executable, SEG, model, src, dst], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst): break
        time.sleep(5)
    if r.returncode != 0 or not os.path.exists(dst):
        raise SystemExit(f"segmentation failed for {src} (exit {r.returncode}); try --setengah for this photo")

def komponen_terbesar(b):
    lab, n = ndi.label(b)
    if n > 1:
        uk = ndi.sum(b, lab, range(1, n + 1)); b = lab == (1 + int(np.argmax(uk)))
    return b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--foto", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--view", nargs=4, required=True, help="front=IMG_x left=IMG_y back=IMG_z right=IMG_w")
    ap.add_argument("--setengah", nargs="*", default=[]); ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--masker-kasar", default=None, help="folder with existing kasar_<IMG>.png coarse masks (skip step 1 segmentation)")
    ap.add_argument("--masker-halus", default=None, help="folder with existing halus_<view>.png fine masks (skip step 3 segmentation)")
    a = ap.parse_args()
    views = dict(v.split("=") for v in a.view)
    for d in ("latar_asli", "latar_putih", "_masker"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
    tmp = tempfile.mkdtemp()
    bb = {}
    for v, nama in views.items():
        src = os.path.join(a.foto, nama + ".png"); dst = os.path.join(a.out, "_masker", f"kasar_{nama}.png")
        if a.masker_kasar:
            dst = os.path.join(a.masker_kasar, f"kasar_{nama}.png")
        elif nama in a.setengah:
            im = Image.open(src); w, h = im.size
            half = os.path.join(tmp, nama + "_setengah.png"); im.resize((w // 2, h // 2)).save(half)
            masker(a.model, half, dst + ".setengah.png")
            Image.open(dst + ".setengah.png").resize((w, h), Image.BILINEAR).save(dst)
        else:
            masker(a.model, src, dst)
        b = komponen_terbesar(np.asarray(Image.open(dst).convert("L")) > 127)
        ys, xs = np.where(b); bb[v] = (xs.min(), ys.min(), xs.max(), ys.max())
    L = max(max(q[2] - q[0], q[3] - q[1]) for q in bb.values()); S = int(round(a.margin * L))
    print(f"longest side {L} px -> square {S} px")
    for v, nama in views.items():
        im = Image.open(os.path.join(a.foto, nama + ".png")).convert("RGB"); q = bb[v]
        cx, cy = (q[0] + q[2]) / 2, (q[1] + q[3]) / 2
        x0, y0 = int(round(cx - S / 2)), int(round(cy - S / 2))
        x0 = min(max(x0, 0), im.width - S); y0 = min(max(y0, 0), im.height - S)
        pa = os.path.join(a.out, "latar_asli", v + ".png")
        im.crop((x0, y0, x0 + S, y0 + S)).resize((a.res, a.res), Image.LANCZOS).save(pa)
        if a.masker_halus:
            mh = os.path.join(a.masker_halus, f"halus_{v}.png")
        else:
            mh = os.path.join(a.out, "_masker", f"halus_{v}.png"); masker(a.model, pa, mh)
        rgb = np.asarray(Image.open(pa).convert("RGB"), dtype=np.float32)
        al0 = np.asarray(Image.open(mh).convert("L"), dtype=np.float32) / 255
        bf = ndi.binary_fill_holes(komponen_terbesar(al0 > 0.5))
        al = np.where(ndi.binary_dilation(bf, iterations=2), al0, 0.0)
        al = np.where(ndi.binary_erosion(bf, iterations=2), 1.0, al)
        out = rgb * al[..., None] + 255 * (1 - al[..., None])
        Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(os.path.join(a.out, "latar_putih", v + ".png"))
        print(v, nama, "crop at", (x0, y0))

if __name__ == "__main__":
    main()
