#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluates the seed-123 and seed-777 reconstructions at 0 and 20 degrees for all 30 components.

Two sources are recorded under separate model names: newly generated meshes (results/<arm>/objNN.glb)
and earlier replicate runs (results/m130-elevE-seedS/objNN.glb, obj01-08, model name <arm>__e4lama).
analysis/analisis_seed_n30.py decides which are used, based on a pre-specified reproducibility gate.

Writes only eval/hasil_seed_n30.csv (rows appended; pairs already present are skipped, so an interrupted
run can be resumed). The written plan is dated 17 Sep 2026.

Usage: python eval/evaluasi_seed_n30.py [--cek]   (--cek reports status without computing)
"""
import csv
import glob
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_KELUARAN = os.path.join(BASE, "eval", "hasil_seed_n30.csv")
EVALUATOR = os.path.join(BASE, "eval", "evaluate_mesh.py")

OBJEK = [f"obj{i:02d}" for i in range(1, 31)]
LENGAN = {  # lengan kanonik -> folder E4 lama
    "hunyuan_m130_elev0_seed123": "m130-elev0-seed123",
    "hunyuan_m130_elev0_seed777": "m130-elev0-seed777",
    "hunyuan_m130_elev20_seed123": "m130-elev20-seed123",
    "hunyuan_m130_elev20_seed777": "m130-elev20-seed777",
}
OBJEK_E4 = [f"obj{i:02d}" for i in range(1, 9)]
SUFIKS_E4 = "__e4lama"


def ground_truth(oid):
    k = sorted(glob.glob(os.path.join(BASE, "dataset", f"{oid}_*.stl")))
    if len(k) != 1:
        sys.exit(f"BERHENTI: {len(k)} ground truth cocok dataset/{oid}_*.stl (harus tepat 1)")
    return k[0]


def daftar_tugas():
    tugas, hilang = [], {l: [] for l in LENGAN}
    for lengan, lama in LENGAN.items():
        for oid in OBJEK:
            p = os.path.join(BASE, "results", lengan, f"{oid}.glb")
            if os.path.exists(p):
                tugas.append((lengan, oid, p))
            else:
                hilang[lengan].append(oid)
            if oid in OBJEK_E4:
                q = os.path.join(BASE, "results", lama, f"{oid}.glb")
                if os.path.exists(q):
                    tugas.append((lengan + SUFIKS_E4, oid, q))
    return tugas, hilang


def sudah_ada():
    if not os.path.exists(CSV_KELUARAN):
        return set()
    with open(CSV_KELUARAN, newline="", encoding="utf-8") as f:
        return {(r["model"], r["gt"][:5]) for r in csv.DictReader(f)}


def main():
    cek = "--cek" in sys.argv[1:]
    if not os.path.exists(EVALUATOR):
        sys.exit(f"BERHENTI: evaluator tidak ada: {EVALUATOR}")
    tugas, hilang = daftar_tugas()
    selesai = sudah_ada()
    sisa = [t for t in tugas if (t[0], t[1]) not in selesai]

    print("=" * 72)
    for lengan in LENGAN:
        n_ada = 30 - len(hilang[lengan])
        tanda = "OK" if n_ada == 30 else "BELUM LENGKAP"
        print(f"  {lengan:28s} GLB kanonik {n_ada:2d}/30  {tanda}")
        if hilang[lengan]:
            print(f"      belum ada: {', '.join(hilang[lengan])}")
    n_e4 = sum(1 for t in tugas if t[0].endswith(SUFIKS_E4))
    print(f"  GLB E4 lama (obj01-08)       {n_e4}/32")
    print(f"  pasangan total {len(tugas)} | sudah di CSV {len(tugas) - len(sisa)} | sisa {len(sisa)}")
    print("=" * 72)
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
            print(f"  [{i:3d}/{len(sisa)}] {model}/{oid} GAGAL\n{r.stderr[-400:]}")
        else:
            print(f"  [{i:3d}/{len(sisa)}] {model}/{oid} ok {time.time() - t1:5.1f}s")
        if i % 10 == 0:
            print(f"      ~{(time.time() - t0) / i * (len(sisa) - i) / 60:.1f} menit lagi")

    with open(CSV_KELUARAN, newline="", encoding="utf-8") as f:
        kunci = [(r["model"], r["gt"][:5]) for r in csv.DictReader(f)]
    dup = {k for k in kunci if kunci.count(k) > 1}
    print("=" * 72)
    print(f"baris di CSV: {len(kunci)} | duplikat: {len(dup)} | gagal: {gagal}")
    if dup or gagal:
        sys.exit("BERHENTI: ada duplikat atau kegagalan, periksa sebelum analisis.")
    print(f"Bersih: {CSV_KELUARAN}")


if __name__ == "__main__":
    main()
