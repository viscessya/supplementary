#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-specified analysis of the second-seed replication (seed 123) of the positive result in Table 4
(perspective projection at 30 and 40 degrees) and of the single gray-background difference at 40 degrees.

Thresholds, tests, and predictions were fixed before any seed-123 mesh was generated
(written plan dated 20 Sep 2026; its "Table 3" is Table 4 in the manuscript).

Inputs:  eval/hasil_seed_tabel3.csv (seed 123), eval/hasil_evaluasi_n30.csv (seed 42)
Output:  eval/analisis_seed_tabel3.txt
Usage:   python eval/analisis_seed_tabel3.py
"""
import csv
import os
import sys

import numpy as np
import scipy
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_S123 = os.path.join(BASE, "eval", "hasil_seed_tabel3.csv")
CSV_S42 = os.path.join(BASE, "eval", "hasil_evaluasi_n30.csv")
KELUARAN = os.path.join(BASE, "eval", "analisis_seed_tabel3.txt")

OBJEK = ["obj%02d" % i for i in range(1, 31)]
METRIK = [("chamfer", -1, "Chamfer"),
          ("fscore@0.01", 1, "F@0.01"),
          ("fscore@0.02", 1, "F@0.02"),
          ("normal_consistency", 1, "Normal cons."),
          ("volume_iou", 1, "Volume IoU")]

UJI = [
    ("R1", "Perspektif vs raw, 30 derajat",
     "hunyuan_persp_v4_elev30_seed123", "hunyuan_m130_elev30_seed123",
     "hunyuan_persp_v4_elev30", "hunyuan_m130_elev30"),
    ("R2", "Perspektif vs raw, 40 derajat",
     "hunyuan_persp_v4_elev40_seed123", "hunyuan_m130_elev40_seed123",
     "hunyuan_persp_v4_elev40", "hunyuan_m130_elev40"),
    ("R3", "Latar abu-abu vs raw, 40 derajat",
     "hunyuan_backdrop_v1_elev40_seed123", "hunyuan_m130_elev40_seed123",
     "hunyuan_backdrop_v1_elev40", "hunyuan_m130_elev40"),
]

GERBANG_MIN_BEDA = 28
GERBANG_TOL = 1e-6

catatan = []


def tulis(s=""):
    print(s)
    catatan.append(s)


def muat(path, wajib=True):
    if not os.path.exists(path):
        if wajib:
            sys.exit("BERHENTI: tidak ada %s" % path)
        return {}
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[(r["model"], r["gt"][:5])] = r
    return d


def nilai(d, model, metr):
    hilang = [o for o in OBJEK if (model, o) not in d]
    if hilang:
        sys.exit("BERHENTI: lengan %s kurang %d objek (%s ...). Jalankan evaluasi dulu."
                 % (model, len(hilang), ", ".join(hilang[:5])))
    return np.array([float(d[(model, o)][metr]) for o in OBJEK])


def holm(p):
    urut = sorted(range(len(p)), key=lambda i: p[i])
    keluar = [0.0] * len(p)
    jalan = 0.0
    for k, i in enumerate(urut):
        jalan = max(jalan, (len(p) - k) * p[i])
        keluar[i] = min(jalan, 1.0)
    return keluar


def satu_baris(d, varian, dasar):
    """Kembalikan daftar (nama, rerata_dasar, rerata_varian, menang, p, holm_p)."""
    mentah, bagian = [], []
    for metr, arah, nama in METRIK:
        a = nilai(d, dasar, metr)
        b = nilai(d, varian, metr)
        p = stats.wilcoxon(a, b, method="exact").pvalue
        menang = int(np.sum((b - a) * arah > 0))
        mentah.append(p)
        bagian.append([nama, a.mean(), b.mean(), menang, p])
    for baris, h in zip(bagian, holm(mentah)):
        baris.append(h)
    return bagian


def cetak_baris(judul, bagian):
    tulis("     %s" % judul)
    for nama, ma, mb, menang, p, h in bagian:
        vonis = "BETTER" if (h < 0.05 and menang > 15) else ("WORSE" if h < 0.05 else "n.s.")
        tulis("       %-13s dasar %.4f -> varian %.4f | varian menang %2d/30 | p=%.4f Holm=%.4f  %s"
              % (nama, ma, mb, menang, p, h, vonis))


def main():
    d123 = muat(CSV_S123)
    d42 = muat(CSV_S42)

    tulis("=" * 78)
    tulis("ANALISIS REPLIKASI SEED UNTUK TABLE 3 (R3-S)")
    tulis("uji, gerbang, dan prediksi didaftarkan di")
    tulis("rencana tertulis 20 Sep, sebelum generasi")
    tulis("scipy %s | numpy %s" % (scipy.__version__, np.__version__))
    tulis("=" * 78)
    tulis()

    lengan123 = sorted({m for (m, _) in d123})
    tulis("[G2] Kelengkapan lengan seed 123")
    kurang = False
    for lengan in ["hunyuan_m130_elev30_seed123", "hunyuan_m130_elev40_seed123",
                   "hunyuan_persp_v4_elev30_seed123", "hunyuan_persp_v4_elev40_seed123",
                   "hunyuan_backdrop_v1_elev40_seed123"]:
        n = sum(1 for o in OBJEK if (lengan, o) in d123)
        tulis("     %-36s %2d/30 %s" % (lengan, n, "OK" if n == 30 else "BELUM LENGKAP"))
        kurang = kurang or n != 30
    if kurang:
        sys.exit("BERHENTI: ada lengan yang belum 30/30. Jangan mengutip angka parsial.")
    tulis()

    tulis("[G1] Seed benar-benar berganti: Chamfer seed 123 vs seed 42, per objek")
    tulis("     ambang: minimal %d/30 objek berbeda lebih dari %g" % (GERBANG_MIN_BEDA, GERBANG_TOL))
    pasangan = [("hunyuan_m130_elev30_seed123", "hunyuan_m130_elev30"),
                ("hunyuan_m130_elev40_seed123", "hunyuan_m130_elev40"),
                ("hunyuan_persp_v4_elev30_seed123", "hunyuan_persp_v4_elev30"),
                ("hunyuan_persp_v4_elev40_seed123", "hunyuan_persp_v4_elev40"),
                ("hunyuan_backdrop_v1_elev40_seed123", "hunyuan_backdrop_v1_elev40")]
    g1_lulus = True
    for baru, lama in pasangan:
        a = nilai(d123, baru, "chamfer")
        b = nilai(d42, lama, "chamfer")
        beda = int(np.sum(np.abs(a - b) > GERBANG_TOL))
        ok = beda >= GERBANG_MIN_BEDA
        g1_lulus = g1_lulus and ok
        tulis("     %-36s berbeda %2d/30 | median |dCD| %.5f  %s"
              % (baru, beda, float(np.median(np.abs(a - b))), "LULUS" if ok else "GAGAL"))
    if not g1_lulus:
        tulis()
        tulis("     => GAGAL: SEED_OVERRIDE tidak mengenai salinan runtime workflow.")
        tulis("        Ini kegagalan mekanis, bukan temuan. Perbaiki dulu, jangan analisis.")
        with open(KELUARAN, "w", encoding="utf-8") as f:
            f.write("\n".join(catatan) + "\n")
        sys.exit(1)
    tulis("     => LULUS")
    tulis()

    hasil = {}
    for kode, label, v123, d123_dasar, v42, d42_dasar in UJI:
        tulis("[%s] %s" % (kode, label))
        b42 = satu_baris(d42, v42, d42_dasar)
        b123 = satu_baris(d123, v123, d123_dasar)
        cetak_baris("seed  42 (sudah di naskah, dihitung ulang di sini)", b42)
        cetak_baris("seed 123 (BARU)", b123)
        hasil[kode] = (b42, b123)
        tulis()

    tulis("=" * 78)
    tulis("[PREDIKSI] dicocokkan otomatis dengan yang tertulis di rencana 20 Sep")
    tulis("=" * 78)

    def lolos(bagian):
        return [nm for nm, ma, mb, mn, p, h in bagian if h < 0.05 and mn > 15]

    r2 = lolos(hasil["R2"][1])
    p1 = len(r2) == 5
    tulis("  P1 R2 (40 derajat) signifikan kelima metrik   : %d/5 (%s) -> %s"
          % (len(r2), ", ".join(r2) if r2 else "-", "TERBUKTI" if p1 else "MELESET"))

    r1 = lolos(hasil["R1"][1])
    p2 = len(r1) >= 3
    tulis("  P2 R1 (30 derajat) signifikan minimal 3 dari 5: %d/5 (%s) -> %s"
          % (len(r1), ", ".join(r1) if r1 else "-", "TERBUKTI" if p2 else "MELESET"))

    def selisih_cd(bagian):
        for nm, ma, mb, mn, p, h in bagian:
            if nm == "Chamfer":
                return ma - mb
        return float("nan")

    d30 = selisih_cd(hasil["R1"][1])
    d40 = selisih_cd(hasil["R2"][1])
    p3 = d40 > d30
    tulis("  P3 keunggulan Chamfer 40 > 30 derajat          : 40deg %.4f vs 30deg %.4f -> %s"
          % (d40, d30, "TERBUKTI" if p3 else "MELESET"))

    nc = [b for b in hasil["R3"][1] if b[0] == "Normal cons."][0]
    p4 = not (nc[5] < 0.05 and nc[3] > 15)
    tulis("  P4 R3 normal consistency TIDAK signifikan      : Holm p=%.4f, menang %d/30 -> %s"
          % (nc[5], nc[3], "TERBUKTI" if p4 else "MELESET"))

    tulis()
    tulis("=" * 78)
    tulis("KONSEKUENSI UNTUK NASKAH (aturan keputusan rencana 20 Sep)")
    tulis("=" * 78)
    if p1:
        tulis("  Aturan 1: klaim perspektif DIPERTAHANKAN, ditambah keterangan bahwa")
        tulis("            keunggulan di 40 derajat bertahan pada seed kedua.")
        tulis("            Kalimat 'but that observation rests on seed 42' di Discussion diganti.")
    else:
        tulis("  Aturan 2: klaim perspektif DICABUT dari Discussion. Sec. 4.3 menyatakan")
        tulis("            keunggulan seed-42 tidak bertahan pada seed kedua.")
        tulis("            JANGAN menambah seed ketiga untuk mencari hasil yang lebih enak.")
    if p1 and not p2:
        tulis("  Aturan 3: 40 derajat bertahan, 30 derajat tidak. Tulis keduanya.")
    if not p4:
        tulis("  Aturan 4: latar abu-abu pada normal consistency BERTAHAN di seed kedua.")
        tulis("            Naik status dari anomali menjadi temuan kecil yang perlu disebut.")
    else:
        tulis("  Aturan 4: latar abu-abu tidak bertahan. Laporkan satu anak kalimat sebagai")
        tulis("            artefak multiplisitas.")
    tulis("  Aturan 6: hasil utama naskah (elevasi, 3 seed, n=30) tidak tersentuh analisis ini.")

    with open(KELUARAN, "w", encoding="utf-8") as f:
        f.write("\n".join(catatan) + "\n")
    print()
    print("ditulis: %s" % KELUARAN)


if __name__ == "__main__":
    main()
