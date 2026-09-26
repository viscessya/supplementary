#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-evaluation gate: opens every generated mesh of every arm and checks that it is a usable mesh
before evaluate_mesh.py is run on it. Read-only; safe to repeat.

The first output line reports the library versions in use. trimesh must be 4.12.2
(requirements-pinned.txt), because volumetric IoU differs between trimesh versions.

Outputs: a summary on screen, eval/gerbang_mesh_n30.csv (per-file statistics), and eval/gerbang_mesh_n30.txt
(full log with library versions). Exit code 1 if any file fails, 0 otherwise.
Usage: python eval/gerbang_mesh_n30.py
"""
import csv
import glob
import os
import re
import sys
import platform
import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LENGAN = [
    "hunyuan_m130_elev0",
    "hunyuan_m130_elev20",
    "hunyuan_sv_m130",
    "hunyuan_sv_withmethod",
    "hunyuan_mv_withmethod_elev20",
    "hunyuan_material_v1_elev20",
    "hunyuan_persp_v4_elev20",
    "hunyuan_diagonal_v1_elev20",
    "hunyuan_backdrop_v1_elev20",
    "trellis_sv_m130_elev0",
    "trellis_mv_m130",
    "trellis_sv_nometode",
    "triposr_sv_m130_elev0",
    "triposr_sv_nometode",
    "hunyuan_m130_elev10",
    "hunyuan_m130_elev30",
    "hunyuan_m130_elev40",
    "hunyuan_mv_withmethod_elev0",
    "hunyuan_mv_withmethod_elev10",
    "hunyuan_mv_withmethod_elev30",
    "hunyuan_mv_withmethod_elev40",
    "hunyuan_material_v1_elev0",
    "hunyuan_material_v1_elev10",
    "hunyuan_material_v1_elev30",
    "hunyuan_material_v1_elev40",
    "hunyuan_persp_v4_elev0",
    "hunyuan_persp_v4_elev10",
    "hunyuan_persp_v4_elev30",
    "hunyuan_persp_v4_elev40",
    "hunyuan_diagonal_v1_elev0",
    "hunyuan_diagonal_v1_elev10",
    "hunyuan_diagonal_v1_elev30",
    "hunyuan_diagonal_v1_elev40",
    "hunyuan_backdrop_v1_elev0",
    "hunyuan_backdrop_v1_elev10",
    "hunyuan_backdrop_v1_elev30",
    "hunyuan_backdrop_v1_elev40",
]
N_DIHARAP = 30
POLA = re.compile(r"^obj\d{2}\.glb$")

PASANGAN = [
    ("trellis_sv_m130_elev0", "trellis_sv_nometode"),
    ("triposr_sv_m130_elev0", "triposr_sv_nometode"),
    ("hunyuan_sv_withmethod", "hunyuan_sv_m130"),
    ("hunyuan_mv_withmethod_elev20", "hunyuan_m130_elev20"),
    ("hunyuan_material_v1_elev20", "hunyuan_m130_elev20"),
    ("hunyuan_persp_v4_elev20", "hunyuan_m130_elev20"),
    ("hunyuan_diagonal_v1_elev20", "hunyuan_m130_elev20"),
    ("hunyuan_backdrop_v1_elev20", "hunyuan_m130_elev20"),
]

catatan = []


def cetak(s=""):
    print(s)
    catatan.append(s)


def glb_objek(d):
    """Hanya objNN.glb. Varian _bersih/_solid TIDAK ikut — pola longgar
    'obj*.glb' pernah membuat lengan 8 objek terbaca 24."""
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if POLA.match(f))


def main():
    try:
        import numpy
        import scipy
        import trimesh
    except ImportError as e:
        sys.exit(f"BERHENTI: pustaka tidak ada ({e}).\n"
                 f"   Interpreter ini tidak memuat pustaka yang dikunci. Pakai lingkungan requirements-pinned.txt:\n"
                 f"   python eval/gerbang_mesh_n30.py")

    cetak("=" * 78)
    cetak("GERBANG MESH n=30")
    cetak("=" * 78)
    cetak(f"waktu      : {datetime.datetime.now().isoformat(timespec='seconds')}")
    cetak(f"python     : {sys.version.split()[0]}  ({platform.platform()})")
    cetak(f"executable : {sys.executable}")
    cetak(f"numpy      : {numpy.__version__}   (terkunci: 2.5.1)")
    cetak(f"scipy      : {scipy.__version__}   (terkunci: 1.18.0)")
    cetak(f"trimesh    : {trimesh.__version__}   (terkunci: 4.12.2)")
    if trimesh.__version__ != "4.12.2":
        cetak("")
        cetak("PERINGATAN: trimesh BUKAN 4.12.2. requirements-pinned.txt menyebut")
        cetak("   volume_iou berbeda antar versi. Jangan pakai angka dari run ini")
        cetak("   untuk manuskrip sebelum versinya dibereskan.")
    cetak("")

    baris = []
    gagal = []

    for lengan in LENGAN:
        d = os.path.join(BASE, "results", lengan)
        berkas = glb_objek(d)
        diharap = [f"obj{i:02d}.glb" for i in range(1, N_DIHARAP + 1)]
        hilang = sorted(set(diharap) - set(berkas))
        asing = sorted(set(berkas) - set(diharap))

        cetak(f"■ {lengan}")
        cetak(f"    berkas objNN.glb : {len(berkas)}/{N_DIHARAP}")
        if hilang:
            cetak(f"    HILANG        : {hilang}")
            gagal.append(f"{lengan}: {len(hilang)} berkas hilang")
        if asing:
            cetak(f"    TIDAK DIKENAL : {asing}")
            gagal.append(f"{lengan}: nama tidak dikenal {asing}")

        rusak = []
        for nama in berkas:
            p = os.path.join(d, nama)
            r = {"lengan": lengan, "berkas": nama,
                 "byte": os.path.getsize(p),
                 "verteks": "", "face": "", "luas": "",
                 "diagonal_bbox": "", "status": ""}
            try:
                m = trimesh.load(p, force="mesh")
                if isinstance(m, trimesh.Scene):
                    m = trimesh.util.concatenate([g for g in m.geometry.values()])
                nf = len(m.faces)
                r["verteks"] = len(m.vertices)
                r["face"] = nf
                r["luas"] = round(float(m.area), 6)
                r["diagonal_bbox"] = round(
                    float(numpy.linalg.norm(m.bounding_box.extents)), 6)
                if nf == 0:
                    r["status"] = "NOL_FACE"
                    rusak.append(f"{nama}: nol face")
                elif float(m.area) <= 0:
                    r["status"] = "LUAS_NOL"
                    rusak.append(f"{nama}: luas permukaan nol")
                else:
                    r["status"] = "OK"
            except Exception as e:
                r["status"] = f"GAGAL_MUAT: {type(e).__name__}"
                rusak.append(f"{nama}: gagal dimuat — {type(e).__name__}: {e}")
            baris.append(r)

        ok = [b for b in baris if b["lengan"] == lengan and b["status"] == "OK"]
        if ok:
            vs = [b["verteks"] for b in ok]
            fs = [b["face"] for b in ok]
            cetak(f"    bisa dimuat      : {len(ok)}/{len(berkas)}")
            cetak(f"    verteks          : min {min(vs):>9,}  median {sorted(vs)[len(vs)//2]:>9,}  max {max(vs):>9,}")
            cetak(f"    face             : min {min(fs):>9,}  median {sorted(fs)[len(fs)//2]:>9,}  max {max(fs):>9,}")
        if rusak:
            for x in rusak:
                cetak(f"    {x}")
            gagal.append(f"{lengan}: {len(rusak)} berkas bermasalah")
        cetak("")

    cetak("=" * 78)
    cetak("PERBANDINGAN VERTEKS/FACE — dengan metode vs tanpa metode")
    cetak("=" * 78)
    cetak("Ini calon angka manuskrip (Sec. 4.2). Sifatnya DESKRIPTIF: ia menjelaskan")
    cetak("apa yang terjadi pada mesh, bukan membuktikan metode lebih baik. Klaim")
    cetak("kualitas tetap butuh metrik terhadap ground truth, bukan jumlah face.")
    cetak("")
    peta = {(b["lengan"], b["berkas"]): b for b in baris if b["status"] == "OK"}
    for dgn, tanpa in PASANGAN:
        pasang = []
        for i in range(1, N_DIHARAP + 1):
            nama = f"obj{i:02d}.glb"
            a, b_ = peta.get((dgn, nama)), peta.get((tanpa, nama))
            if a and b_:
                pasang.append((nama, a["face"], b_["face"]))
        if not pasang:
            cetak(f"■ {dgn} vs {tanpa}: tidak ada pasangan lengkap")
            continue
        turun = sum(1 for _, fa, fb in pasang if fb < fa)
        rasio = sorted(fb / fa for _, fa, fb in pasang if fa > 0)
        med = rasio[len(rasio) // 2]
        cetak(f"■ {dgn}  vs  {tanpa}   (n={len(pasang)} pasang)")
        cetak(f"    face tanpa-metode LEBIH KECIL pada : {turun}/{len(pasang)} objek")
        cetak(f"    rasio face (tanpa/dengan)          : min {rasio[0]:.3f}  median {med:.3f}  max {rasio[-1]:.3f}")
        ekstrem = sorted(pasang, key=lambda t: t[2] / t[1] if t[1] else 9e9)[:5]
        cetak(f"    5 penurunan terbesar               : " +
              ", ".join(f"{n[:5]} {fb/fa:.2f}x" for n, fa, fb in ekstrem))
        cetak("")

    out_csv = os.path.join(BASE, "eval", "gerbang_mesh_n30.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)

    cetak("=" * 78)
    if gagal:
        cetak("GERBANG TIDAK LOLOS")
        for g in gagal:
            cetak(f"   - {g}")
        cetak("")
        cetak("JANGAN lanjut ke evaluasi. Perbaiki dulu berkas yang bermasalah.")
    else:
        cetak("OK GERBANG LOLOS")
        cetak(f"   {len(baris)} berkas, ketujuh lengan 30/30, semua bisa dimuat,")
        cetak("   tidak ada nol-face dan tidak ada luas-nol.")
        cetak("")
        cetak("   Lanjut:  python eval/jalankan_evaluasi_n30.py")
    cetak("=" * 78)
    cetak(f"rincian per berkas : eval/gerbang_mesh_n30.csv")

    out_txt = os.path.join(BASE, "eval", "gerbang_mesh_n30.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(catatan) + "\n")
    print(f"catatan lengkap    : eval/gerbang_mesh_n30.txt")

    sys.exit(1 if gagal else 0)


if __name__ == "__main__":
    main()
