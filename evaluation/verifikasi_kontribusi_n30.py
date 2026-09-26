#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prints every number behind the contribution claims, computed directly from eval/hasil_evaluasi_n30.csv.

No result value is typed into the script. The number of components is derived from the data and checked
to be equal across arms; the script stops otherwise. Only csv, numpy, and scipy are used. The one
pre-specified confirmatory comparison (0 vs 20 degrees, unmodified render) is reported separately from the
exploratory comparisons.

Usage: python eval/verifikasi_kontribusi_n30.py
"""

import csv
import os
import re
import sys

import numpy as np
import scipy
from scipy.stats import wilcoxon

DIR = os.path.dirname(os.path.abspath(__file__))
CSV_UTAMA = os.path.join(DIR, "hasil_evaluasi_n30.csv")
CSV_SOLID = os.path.join(DIR, "validasi_solidify_n30.csv")
CSV_POST = os.path.join(DIR, "validasi_postproc_n30.csv")

METRIK_TURUN = {"chamfer"}  # makin kecil makin baik


def muat(path):
    if not os.path.exists(path):
        sys.exit(f"tidak ada: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lengan(rows, nama):
    d = {r["gt"]: r for r in rows if r["model"] == nama}
    if not d:
        sys.exit(f"lengan '{nama}' tidak ada di CSV. Yang ada: "
                 f"{sorted(set(r['model'] for r in rows))}")
    return d


def nomor(kunci):
    m = re.match(r"obj(\d+)", kunci)
    if not m:
        sys.exit(f"nama gt tidak terduga: {kunci}")
    return int(m.group(1))


def benar(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def uji(A, B, metrik, kunci):
    """Return (rata_A, rata_B, jumlah_A_lebih_baik, p, n)."""
    x = np.array([float(A[k][metrik]) for k in kunci])
    y = np.array([float(B[k][metrik]) for k in kunci])
    p = wilcoxon(x, y, method="auto").pvalue
    baik = int((x < y).sum()) if metrik in METRIK_TURUN else int((x > y).sum())
    return x.mean(), y.mean(), baik, p, len(kunci)


def banding(rows, a, b, judul, metrik, batas_bawah=None, batas_atas=None):
    A, B = lengan(rows, a), lengan(rows, b)
    kunci = sorted(set(A) & set(B), key=nomor)
    if batas_bawah is not None:
        kunci = [k for k in kunci if nomor(k) >= batas_bawah]
    if batas_atas is not None:
        kunci = [k for k in kunci if nomor(k) <= batas_atas]
    print(f"\n### {judul}")
    print(f"    A = {a}")
    print(f"    B = {b}     n = {len(kunci)}")
    print(f"    {'metrik':22}{'A':>10}{'B':>10}{'A lbh baik':>12}{'p':>12}")
    for m in metrik:
        ma, mb, baik, p, n = uji(A, B, m, kunci)
        tanda = " *" if p < 0.05 else ""
        print(f"    {m:22}{ma:10.4f}{mb:10.4f}{baik:>7}/{n:<4}{p:12.2e}{tanda}")
    return A, B, kunci


def persen_median(A, B, metrik, kunci):
    """Perubahan persen per objek, lalu median. Dipakai untuk klaim '-12,5%'."""
    v = []
    for k in kunci:
        a, b = float(A[k][metrik]), float(B[k][metrik])
        if b != 0:
            v.append((a - b) / b * 100.0)
    return float(np.median(v))


def median_kolom(A, kunci, kolom):
    return float(np.median([float(A[k][kolom]) for k in kunci]))


print("=" * 92)
print("VERIFIKASI ANGKA PARAGRAF CONTRIBUTIONS — n=30")
print("=" * 92)
print(f"python : {sys.executable}")
print(f"numpy  : {np.__version__}")
print(f"scipy  : {scipy.__version__}")

utama = muat(CSV_UTAMA)
print(f"sumber : {os.path.basename(CSV_UTAMA)}, {len(utama)} baris")

per_lengan = {}
for r in utama:
    per_lengan.setdefault(r["model"], set()).add(r["gt"])
jumlah = {k: len(v) for k, v in per_lengan.items()}
if len(set(jumlah.values())) != 1:
    sys.exit(f"GERBANG GAGAL — jumlah objek tidak sama antar lengan: {jumlah}")
N = next(iter(jumlah.values()))
print(f"gerbang: {len(jumlah)} lengan x {N} objek, konsisten OK")

metrik_penuh = ["chamfer", "fscore@0.01", "fscore@0.02", "precision@0.01",
                "recall@0.01", "normal_consistency", "volume_iou"]

print("\n" + "=" * 92)
print("KONTRIBUSI 1 — jumlah view pada model terlatih multi-view")
print("=" * 92)
A, B, K = banding(utama, "hunyuan_m130_elev0", "hunyuan_sv_m130",
                  "Hunyuan3D-2mv: 4 view (A) vs 1 view (B)", metrik_penuh)
print(f"    {'components (median)':22}{median_kolom(A,K,'pred_components'):10.2f}"
      f"{median_kolom(B,K,'pred_components'):10.2f}", end="")
_, _, _, p_komp, _ = uji(A, B, "pred_components", K)
print(f"{'':>12}{p_komp:12.2e}")

print("\n--- hasil nol yang dilaporkan sebagai hasil nol ---")
banding(utama, "trellis_mv_m130", "trellis_sv_m130_elev0",
        "TRELLIS: 4 view (A) vs 1 view (B) — tidak terlatih untuk fusi", metrik_penuh)

print("\n" + "=" * 92)
print("KONTRIBUSI 2 — elevasi kamera")
print("=" * 92)
banding(utama, "hunyuan_m130_elev20", "hunyuan_m130_elev0",
        "Hunyuan3D-2mv 4 view: elevasi 20 derajat (A) vs 0 derajat (B)", metrik_penuh)
print("    precision@0.01 diperkirakan TEPAT DI AMBANG. Jangan menulis")
print("       'all metrics significant' kalau angkanya >= 0.05.")

print("\n" + "=" * 92)
print("EKSPLORATIF (BELUM di keluarga sekunder terdaftar) — Kandidat B: material")
print("=" * 92)
print("    Ditambahkan 2 Sep 2026. Lihat catatan internal")
print("    §8-9c untuk latar belakang lengkap dan riwayat kalibrasi cahaya yang gagal.")
print("    Ini uji EKSPLORATIF, bukan bagian keluarga sekunder Holm-Bonferroni yang")
print("       diregister di aturan_berhenti_dataset.md. Kalau dilaporkan di naskah,")
print("       harus disebut eksplisit sebagai eksploratif/follow-up, bukan konfirmatori.")
print("    CATATAN PENTING: render arm ini konsisten LEBIH TERANG dari baseline")
print("       (rata-rata +18 RGB per piksel objek, sd hanya ±4 antar-objek — jadi biasnya")
print("       konsisten, bukan berantakan, tapi tetap ADA). Kalau hasil di bawah berbeda")
print("       signifikan, itu TIDAK BOLEH langsung diklaim murni efek 'material lebih")
print("       realistis' -- kecerahan adalah variabel pengganggu parsial yang belum")
print("       terpisahkan (dua upaya kalibrasi cahaya sudah dicoba dan GAGAL memperbaiki")
print("       tanpa efek samping baru -- lihat §9c).")
banding(utama, "hunyuan_material_v1_elev20", "hunyuan_m130_elev20",
        "Hunyuan3D-2mv 4 view elev20: material semi-metalik (A) vs clay abu-abu (B)",
        metrik_penuh)

print("\n" + "=" * 92)
print("EKSPLORATIF (BELUM di keluarga sekunder terdaftar) — Kandidat A: proyeksi perspektif")
print("=" * 92)
print("    Ditambahkan 2 Sep 2026 (malam). Lihat")
print("    catatan internal untuk riwayat")
print("    lengkap (4 versi skrip render sebelum layak dipakai) dan verifikasi ukuran.")
print("    Ini uji EKSPLORATIF, bukan bagian keluarga sekunder Holm-Bonferroni yang")
print("       diregister di aturan_berhenti_dataset.md. Kalau dilaporkan di naskah,")
print("       harus disebut eksplisit sebagai eksploratif/follow-up, bukan konfirmatori.")
print("    CATATAN PENTING: 5/30 objek (obj01, obj09, obj13, obj15, obj17) kena")
print("       pengecualian pelebaran FOV keamanan anti-potong -- dampak ukuran pada 5")
print("       objek itu sebagian besar 96,5-108,3% vs baseline ortografik, 2 sudut kamera")
print("       turun ke 83,6% dan 85,8-85,9%. Kalau hasil di bawah berbeda signifikan,")
print("       TIDAK BOLEH langsung diklaim murni efek 'perspektif lebih realistis' --")
print("       untuk 5 objek itu, sebagian perbedaan bisa jadi efek ukuran-di-frame yang")
print("       sedikit berkurang, bukan efek proyeksi perspektif itu sendiri.")
banding(utama, "hunyuan_persp_v4_elev20", "hunyuan_m130_elev20",
        "Hunyuan3D-2mv 4 view elev20: proyeksi perspektif (A) vs ortografik (B)",
        metrik_penuh)

print("\n" + "=" * 92)
print("EKSPLORATIF (BELUM di keluarga sekunder terdaftar) — Kandidat C: azimuth diagonal")
print("=" * 92)
print("    Ditambahkan 3 Sep 2026. Lihat")
print("    catatan internal untuk latar belakang.")
print("    Ini uji EKSPLORATIF, bukan bagian keluarga sekunder Holm-Bonferroni yang")
print("       diregister di aturan_berhenti_dataset.md. Kalau dilaporkan di naskah,")
print("       harus disebut eksplisit sebagai eksploratif/follow-up, bukan konfirmatori.")
print("    Kamera tetap ORTOGRAFIK (bukan proyeksi baru seperti Kandidat A), material/")
print("       elevasi/margin tetap sama seperti baseline -- SATU-SATUNYA variabel yang")
print("       berubah adalah sudut azimuth (45/135/225/315 derajat menggantikan")
print("       0/90/180/270). Arah efek TIDAK diprediksi sebelumnya (didaftarkan di")
print("       rencana internal sebelum render pertama).")
banding(utama, "hunyuan_diagonal_v1_elev20", "hunyuan_m130_elev20",
        "Hunyuan3D-2mv 4 view elev20: azimuth diagonal (A) vs ortogonal (B)",
        metrik_penuh)

print("\n" + "=" * 92)
print("EKSPLORATIF (BELUM di keluarga sekunder terdaftar) — Kandidat D: latar tidak terkontrol")
print("=" * 92)
print("    Ditambahkan 3 Sep 2026. Lihat")
print("    catatan internal untuk latar belakang.")
print("    Ini uji EKSPLORATIF, bukan bagian keluarga sekunder Holm-Bonferroni yang")
print("       diregister di aturan_berhenti_dataset.md. Kalau dilaporkan di naskah,")
print("       harus disebut eksplisit sebagai eksploratif/follow-up, bukan konfirmatori.")
print("    Kamera/elevasi(20)/margin/azimuth (0/90/180/270, ortogonal) SEMUA sama")
print("       persis seperti baseline -- SATU-SATUNYA variabel yang sengaja diubah")
print("       adalah warna latar (world background Cycles): putih murni 1,0 -> abu-abu")
print("       netral 0,5 (konvensi mid-grey yang sama dgn TripoSR/TRELLIS di naskah).")
print("    VARIABEL PENGGANGGU yang SUDAH diketahui SEBELUM render pertama (bukan")
print("       ditemukan belakangan seperti kecerahan Kandidat B): world background node")
print("       di Cycles berperan GANDA -- warna latar DAN sumber cahaya ambien/tidak-")
print("       langsung. Mengubah dari putih ke abu-abu MENGURANGI SEDIKIT kontribusi")
print("       cahaya ambien pada objek dibanding baseline -- kalau hasil di bawah")
print("       berbeda signifikan, TIDAK BOLEH langsung diklaim murni efek 'latar lebih")
print("       realistis/tidak terkontrol' -- sebagian bisa jadi efek pencahayaan yang")
print("       sedikit berkurang, bukan efek warna latar itu sendiri.")
print("    Arah efek TIDAK diprediksi sebelumnya (didaftarkan di")
print("    rencana internal sebelum render pertama).")
banding(utama, "hunyuan_backdrop_v1_elev20", "hunyuan_m130_elev20",
        "Hunyuan3D-2mv 4 view elev20: latar abu-abu 0,5 (A) vs putih murni (B)",
        metrik_penuh)

print("\n" + "=" * 92)
print("KONTRIBUSI 3 — tanda tangan topologi, tanpa ground truth")
print("=" * 92)
print(f"    {'lengan':26}{'watertight':>12}{'tepi batas':>12}{'non-manifold':>14}"
      f"{'komponen':>10}")
print(f"    {'':26}{'':>12}{'(median)':>12}{'(median)':>14}{'(median)':>10}")
for nama in sorted(per_lengan):
    D = lengan(utama, nama)
    k = sorted(D, key=nomor)
    wt = sum(1 for i in k if benar(D[i]["pred_watertight"]))
    print(f"    {nama:26}{wt:>8}/{len(k):<3}"
          f"{median_kolom(D,k,'pred_boundary_edges'):>12.0f}"
          f"{median_kolom(D,k,'pred_nonmanifold_edges'):>14.0f}"
          f"{median_kolom(D,k,'pred_components'):>10.1f}")

metrik_pasca = ["volume_iou", "chamfer", "fscore@0.01", "normal_consistency"]

solid = muat(CSV_SOLID)
print(f"\nsumber : {os.path.basename(CSV_SOLID)}, {len(solid)} baris")
for a, b, judul in [
    ("trellis_sv_m130_elev0_solid", "trellis_sv_m130_elev0_pre",
     "SOLIDIFIKASI · TRELLIS (arsitektur BER-cacat dinding ganda)"),
    ("hunyuan_m130_elev20_solid", "hunyuan_m130_elev20_pre",
     "SOLIDIFIKASI · Hunyuan elev20 (arsitektur TANPA cacat itu)"),
]:
    A, B, K = banding(solid, a, b, judul, metrik_pasca)
    wt = lambda D: sum(1 for k in K if benar(D[k]["pred_watertight"]))
    print(f"    watertight            pre={wt(B):>3}/{len(K)}   post={wt(A):>3}/{len(K)}"
          f"   <-- kalau post << 30, kata 'watertight' di naskah SALAH")
    for kol in ("pred_boundary_edges", "pred_nonmanifold_edges"):
        f = lambda D: np.mean([float(D[k][kol]) for k in K])
        print(f"    {kol:24}pre={f(B):10.1f}   post={f(A):10.1f}")

post = muat(CSV_POST)
print(f"\nsumber : {os.path.basename(CSV_POST)}, {len(post)} baris")
for a, b, judul in [
    ("trellis_sv_m130_elev0_post", "trellis_sv_m130_elev0_pre",
     "PEMBERSIHAN SERPIHAN · TRELLIS (modul yang DITOLAK)"),
    ("hunyuan_m130_elev20_post", "hunyuan_m130_elev20_pre",
     "PEMBERSIHAN SERPIHAN · Hunyuan elev20"),
]:
    A, B, K = banding(post, a, b, judul, metrik_pasca)
    for kol in ("pred_boundary_edges", "pred_nonmanifold_edges"):
        f = lambda D: np.mean([float(D[k][kol]) for k in K])
        print(f"    {kol:24}pre={f(B):10.1f}   post={f(A):10.1f}")

print("\n" + "=" * 92)
print("KONTRIBUSI 4 — praproses siluet, bergantung arsitektur")
print("=" * 92)
A, B, K = banding(utama, "triposr_sv_m130_elev0", "triposr_sv_nometode",
                  "TripoSR: siluet deterministik (A) vs segmenter terlatih (B)",
                  metrik_penuh)
print(f"    chamfer, median perubahan per objek : "
      f"{persen_median(A, B, 'chamfer', K):+.1f} %   <-- angka '-12,5%' di naskah")

banding(utama, "trellis_sv_m130_elev0", "trellis_sv_nometode",
        "TRELLIS: siluet deterministik (A) vs segmenter terlatih (B)", metrik_penuh)

print("\n--- hanya objek yang DITAMBAHKAN setelah pilot (obj09 ke atas) ---")
print("    Kenapa dipisah: pilot n=8 memakai obj01-08. Subset ini menjawab")
print("    'apakah efeknya ada pada objek yang belum pernah dilihat?'")
banding(utama, "trellis_sv_m130_elev0", "trellis_sv_nometode",
        "TRELLIS, objek baru saja", ["chamfer", "fscore@0.01", "fscore@0.02",
                                     "volume_iou", "normal_consistency"],
        batas_bawah=9)
banding(utama, "triposr_sv_m130_elev0", "triposr_sv_nometode",
        "TripoSR, objek baru saja", ["chamfer", "fscore@0.01", "fscore@0.02",
                                     "volume_iou", "normal_consistency"],
        batas_bawah=9)

print("\n" + "=" * 92)
print("PILOT n=8 — klaim lama, dihitung ulang HANYA pada obj01-08")
print("=" * 92)
print("    Dipakai kalau paragraf pilot (§3 draf) jadi ditulis.")
banding(utama, "trellis_sv_m130_elev0", "trellis_sv_nometode",
        "TRELLIS siluet vs terlatih — obj01-08 saja",
        ["volume_iou", "chamfer", "fscore@0.01"], batas_atas=8)
banding(utama, "trellis_mv_m130", "trellis_sv_m130_elev0",
        "TRELLIS 4 view vs 1 view — obj01-08 saja (klaim K2b lama)",
        ["volume_iou", "chamfer", "fscore@0.01"], batas_atas=8)
print("    Angka di atas dihitung dari GLB yang DIGENERATE ULANG 22-23 Agu,")
print("       bukan dari CSV n=8 asli. Selisihnya terukur <1,5% (lihat")
print("       catatan internal), tapi bukan nol.")

print("\n" + "=" * 92)
print("BATAS p TERKECIL PADA n=30 — untuk kalimat §2.5")
print("=" * 92)
n = N
x = np.arange(1, n + 1, dtype=float)
y = np.zeros(n)
print(f"    n = {n} · dua sisi · eksak : "
      f"{wilcoxon(x, y, method='exact').pvalue:.3e}")
print("    (kalimat naskah lama menyebut 0,008 untuk n=8 — harus diganti)")

print("\n" + "=" * 92)
print("SELESAI. Kalau ada baris yang tidak cocok dengan draf, YANG BENAR ADALAH")
print("KELUARAN INI — draf dihitung dengan versi scipy berbeda.")
print("=" * 92)
