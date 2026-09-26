#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runs evaluation/evaluate_mesh.py over every arm and all 30 components and appends the rows to
eval/hasil_evaluasi_n30.csv.

Safeguards: the output CSV name is fixed; pairs already present in the CSV are skipped, so an interrupted
batch can be resumed without duplicate rows; only files named objNN.glb are evaluated; every prediction
must match exactly one reference mesh; the evaluator runs under the same Python interpreter as this script.

Usage:
    python eval/gerbang_mesh_n30.py                  (run the gate first)
    python eval/jalankan_evaluasi_n30.py [ARM ...]   (all arms, or only the arms named)
    python eval/jalankan_evaluasi_n30.py --cek       (check only, nothing is run)
Each pair takes tens of seconds (5-start ICP, 100,000 samples, 128^3 voxels).
"""
import csv
import glob
import os
import re
import subprocess
import sys
import time

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

CSV_KELUARAN = os.path.join(BASE, "eval", "hasil_evaluasi_n30.csv")
EVALUATOR = os.path.join(BASE, "eval", "evaluate_mesh.py")


def glb_objek(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if POLA.match(f))


def sudah_ada():
    """Kembalikan himpunan (model, pred) yang sudah tercatat di CSV keluaran."""
    if not os.path.exists(CSV_KELUARAN):
        return set(), 0
    selesai, rusak = set(), 0
    with open(CSV_KELUARAN, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m, p = (row.get("model") or "").strip(), (row.get("pred") or "").strip()
            if m and p:
                selesai.add((m, p))
            else:
                rusak += 1
    return selesai, rusak


def bangun_tugas(pilih):
    """Daftar (lengan, path_gt, path_pred). Berhenti kalau ada yang tidak beres —
    SEBELUM satu detik pun dihabiskan menghitung."""
    tugas, masalah = [], []
    for lengan in pilih:
        d = os.path.join(BASE, "results", lengan)
        berkas = glb_objek(d)
        if len(berkas) != N_DIHARAP:
            masalah.append(f"{lengan}: {len(berkas)} berkas objNN.glb, seharusnya {N_DIHARAP}")
            continue
        for nama in berkas:
            oid = nama[:5]                      # obj07
            kandidat = sorted(glob.glob(os.path.join(BASE, "dataset", f"{oid}_*.stl")))
            if len(kandidat) != 1:
                masalah.append(f"{lengan}/{nama}: {len(kandidat)} ground truth "
                               f"cocok dataset/{oid}_*.stl (harus tepat 1) {kandidat}")
                continue
            tugas.append((lengan, kandidat[0], os.path.join(d, nama)))
    return tugas, masalah


def gerbang_akhir():
    """Periksa CSV hasil: len(LENGAN)*30 baris, 30 pred unik per lengan, nol duplikat."""
    if not os.path.exists(CSV_KELUARAN):
        return ["CSV keluaran tidak ada"]
    with open(CSV_KELUARAN, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gerbang_akhir.jumlah_baris = len(rows)
    masalah = []
    print()
    print("=" * 78)
    print("GERBANG SEBELUM UJI STATISTIK")
    print("=" * 78)
    print(f"  total baris : {len(rows)}  (diharap {len(LENGAN) * N_DIHARAP})")
    if len(rows) != len(LENGAN) * N_DIHARAP:
        masalah.append(f"jumlah baris {len(rows)}, seharusnya {len(LENGAN) * N_DIHARAP}")
    for lengan in LENGAN:
        sub = [r for r in rows if r.get("model") == lengan]
        pred = [r.get("pred") for r in sub]
        unik = len(set(pred))
        dup = len(pred) - unik
        tanda = "OK" if (len(sub) == N_DIHARAP and dup == 0) else "*** MASALAH ***"
        print(f"  {lengan:24} baris={len(sub):3}  pred_unik={unik:3}  duplikat={dup:2}  {tanda}")
        if len(sub) != N_DIHARAP:
            masalah.append(f"{lengan}: {len(sub)} baris, seharusnya {N_DIHARAP}")
        if dup:
            masalah.append(f"{lengan}: {dup} baris duplikat")
    obj_unik = {r.get("pred") for r in rows}
    print(f"  nama pred unik keseluruhan : {len(obj_unik)}  (diharap {N_DIHARAP})")
    if len(obj_unik) != N_DIHARAP:
        masalah.append(f"{len(obj_unik)} nama pred unik, seharusnya {N_DIHARAP}")
    return masalah


def main():
    argv = [a for a in sys.argv[1:]]
    cek_saja = "--cek" in argv
    argv = [a for a in argv if a != "--cek"]
    pilih = argv if argv else LENGAN
    tak_dikenal = [a for a in pilih if a not in LENGAN]
    if tak_dikenal:
        sys.exit(f"BERHENTI: lengan tidak dikenal {tak_dikenal}\n"
                 f"   Pilihan: {', '.join(LENGAN)}")

    if not os.path.exists(EVALUATOR):
        sys.exit(f"BERHENTI: {EVALUATOR} tidak ada.")

    print("=" * 78)
    print("EVALUASI n=30")
    print("=" * 78)
    print(f"python    : {sys.executable}")
    print(f"keluaran  : {os.path.relpath(CSV_KELUARAN, BASE)}   (DIPAKU MATI)")
    print(f"lengan    : {len(pilih)}")
    print()

    tugas, masalah = bangun_tugas(pilih)
    if masalah:
        print("BERHENTI — pemeriksaan awal gagal. Tidak ada yang dijalankan:")
        for m in masalah:
            print(f"   - {m}")
        sys.exit(1)
    print(f"OK pemeriksaan awal lolos: {len(tugas)} pasangan, tiap pred punya tepat 1 ground truth")

    print()
    print("Provenans berkas per lengan (rentang tanggal):")
    import datetime as _dt
    ada_campur = False
    for lengan in pilih:
        dr = os.path.join(BASE, "results", lengan)
        ts = [os.path.getmtime(os.path.join(dr, f)) for f in glb_objek(dr)]
        if not ts:
            continue
        a, b = min(ts), max(ts)
        rentang_hari = (b - a) / 86400
        tanda = ""
        if rentang_hari > 2:
            tanda = f"   RENTANG {rentang_hari:.0f} HARI — kemungkinan dua run berbeda"
            ada_campur = True
        print(f"   {lengan:24} {_dt.date.fromtimestamp(a)} .. {_dt.date.fromtimestamp(b)}{tanda}")
    if ada_campur:
        print()
        print("   Lengan bertanda di atas berprovenans campuran. Itu TIDAK menghentikan")
        print("      skrip ini — tapi harus jadi keputusan sadar, dan kalau dipertahankan")
        print("      harus disebut di naskah. Untuk menyeragamkan: pindahkan berkas lama")
        print("      ke arsip lalu generate ulang di PC sebelum evaluasi.")
    print()

    selesai, baris_rusak = sudah_ada()
    if baris_rusak:
        print(f"PERHATIAN: {baris_rusak} baris tanpa kolom model/pred di CSV — periksa manual")
    sisa = [t for t in tugas
            if (t[0], os.path.basename(t[2])) not in selesai]
    print(f"   sudah ada di CSV : {len(tugas) - len(sisa)}")
    print(f"   akan dijalankan  : {len(sisa)}")
    print()

    if cek_saja:
        print("--cek: berhenti di sini, tidak ada yang dihitung.")
        return

    if not sisa:
        print("Tidak ada yang perlu dijalankan.")
    else:
        t0 = time.time()
        gagal = []
        for i, (lengan, gt, pred) in enumerate(sisa, 1):
            t1 = time.time()
            cmd = [sys.executable, EVALUATOR,
                   "--gt", gt, "--pred", pred,
                   "--model", lengan, "--csv", CSV_KELUARAN]
            r = subprocess.run(cmd, capture_output=True, text=True)
            dtk = time.time() - t1
            if r.returncode != 0:
                pesan = (r.stderr.strip().splitlines() or ["?"])[-1]
                print(f"[{i}/{len(sisa)}] {lengan}/{os.path.basename(pred)} GAGAL: {pesan}")
                gagal.append((lengan, os.path.basename(pred), pesan))
                continue
            lewat = time.time() - t0
            eta = lewat / i * (len(sisa) - i)
            print(f"[{i}/{len(sisa)}] {lengan}/{os.path.basename(pred)}  "
                  f"{dtk:.1f} dtk  | sisa ~{eta/60:.0f} mnt")

        print()
        if gagal:
            print(f"PERHATIAN: {len(gagal)} pasangan GAGAL:")
            for l, p, m in gagal:
                print(f"   {l}/{p}: {m}")
            print("   Jalankan skrip ini lagi — yang sudah selesai akan dilewati.")

    masalah = gerbang_akhir()
    print()
    if masalah:
        print("GERBANG TIDAK LOLOS — JANGAN lanjut ke Wilcoxon:")
        for m in masalah:
            print(f"   - {m}")
        sys.exit(1)
    print(f"OK GERBANG LOLOS — {gerbang_akhir.jumlah_baris} baris, "
          f"{len(LENGAN)} lengan x {N_DIHARAP} objek, nol duplikat.")
    print("   Boleh lanjut ke uji statistik.")


if __name__ == "__main__":
    main()
