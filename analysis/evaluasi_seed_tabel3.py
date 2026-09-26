#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluates the five seed-123 arms that replicate the positive result in Table 4 (perspective
projection at 30 and 40 degrees) and the gray-background difference at 40 degrees.

Writes only eval/hasil_seed_tabel3.csv (rows appended; pairs already present are skipped).
The written plan is dated 20 Sep 2026.

Usage: python eval/evaluasi_seed_tabel3.py [--cek]
"""
import csv
import glob
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_KELUARAN = os.path.join(BASE, "eval", "hasil_seed_tabel3.csv")
EVALUATOR = os.path.join(BASE, "eval", "evaluate_mesh.py")

OBJEK = ["obj%02d" % i for i in range(1, 31)]
LENGAN = [
    "hunyuan_m130_elev30_seed123",
    "hunyuan_m130_elev40_seed123",
    "hunyuan_persp_v4_elev30_seed123",
    "hunyuan_persp_v4_elev40_seed123",
    "hunyuan_backdrop_v1_elev40_seed123",
]


def ground_truth(oid):
    k = sorted(glob.glob(os.path.join(BASE, "dataset", "%s_*.stl" % oid)))
    if len(k) != 1:
        sys.exit("BERHENTI: %d ground truth cocok dataset/%s_*.stl (harus tepat 1)" % (len(k), oid))
    return k[0]


def daftar_tugas():
    tugas, hilang = [], {l: [] for l in LENGAN}
    for lengan in LENGAN:
        for oid in OBJEK:
            p = os.path.join(BASE, "results", lengan, "%s.glb" % oid)
            if os.path.exists(p):
                tugas.append((lengan, oid, p))
            else:
                hilang[lengan].append(oid)
    return tugas, hilang


def sudah_ada():
    if not os.path.exists(CSV_KELUARAN):
        return set()
    with open(CSV_KELUARAN, newline="", encoding="utf-8") as f:
        return {(r["model"], r["gt"][:5]) for r in csv.DictReader(f)}


def main():
    cek = "--cek" in sys.argv[1:]
    if not os.path.exists(EVALUATOR):
        sys.exit("BERHENTI: evaluator tidak ada: %s" % EVALUATOR)
    tugas, hilang = daftar_tugas()
    selesai = sudah_ada()
    sisa = [t for t in tugas if (t[0], t[1]) not in selesai]

    print("=" * 76)
    for lengan in LENGAN:
        n_ada = 30 - len(hilang[lengan])
        print("  %-36s GLB %2d/30  %s" % (lengan, n_ada, "OK" if n_ada == 30 else "BELUM LENGKAP"))
        if hilang[lengan]:
            print("      belum ada: %s" % ", ".join(hilang[lengan]))
    print("  pasangan total %d | sudah di CSV %d | sisa %d" % (len(tugas), len(tugas) - len(sisa), len(sisa)))
    print("=" * 76)
    if cek:
        print("--cek: tidak ada yang dihitung.")
        return

    t0 = time.time()
    gagal = 0
    for i, (model, oid, pred) in enumerate(sisa, 1):
        cmd = [sys.executable, EVALUATOR, "--gt", ground_truth(oid), "--pred", pred,
               "--model", model, "--csv", CSV_KELUARAN]
        t1 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            gagal += 1
            print("  [%3d/%d] %s/%s GAGAL\n%s" % (i, len(sisa), model, oid, r.stderr[-400:]))
        else:
            print("  [%3d/%d] %s/%s ok %5.1fs" % (i, len(sisa), model, oid, time.time() - t1))
        if i % 10 == 0:
            print("      ~%.1f menit lagi" % ((time.time() - t0) / i * (len(sisa) - i) / 60))

    with open(CSV_KELUARAN, newline="", encoding="utf-8") as f:
        kunci = [(r["model"], r["gt"][:5]) for r in csv.DictReader(f)]
    dup = {k for k in kunci if kunci.count(k) > 1}
    print("=" * 76)
    print("baris di CSV: %d | duplikat: %d | gagal: %d" % (len(kunci), len(dup), gagal))
    if dup or gagal:
        sys.exit("BERHENTI: ada duplikat atau kegagalan, periksa sebelum analisis.")
    print("Bersih: %s" % CSV_KELUARAN)


if __name__ == "__main__":
    main()
