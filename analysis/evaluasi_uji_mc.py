#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extractor comparison: surface net (ComfyUI VoxelToMesh) vs marching cubes on the same decoded grids.

Evaluates both meshes of each of the 60 seed-42 reconstructions of the primary comparison with
evaluation/evaluate_mesh.py. With --analisis it runs the pre-specified analysis: determinism gate against the
original meshes, topology per extractor, geometry difference, the elevation effect under each extractor,
and automatic prediction checks. The plan and decision rules were written on 23 Sep 2026.

Inputs:  results/uji_mc_23sep/<arm>/objNN_sn.glb (surface net) and objNN_mc.ply (marching cubes),
         eval/hasil_evaluasi_n30.csv (original meshes, for the determinism gate)
Outputs: eval/hasil_uji_mc_23sep.csv, eval/analisis_uji_mc_23sep.txt

Usage: python eval/evaluasi_uji_mc.py [--cek | --analisis]
"""
import csv
import glob
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_UJI = os.path.join(BASE, "results", "uji_mc_23sep")
CSV_KELUARAN = os.path.join(BASE, "eval", "hasil_uji_mc_23sep.csv")
CSV_LAMA = os.path.join(BASE, "eval", "hasil_evaluasi_n30.csv")
TXT_ANALISIS = os.path.join(BASE, "eval", "analisis_uji_mc_23sep.txt")
EVALUATOR = os.path.join(BASE, "eval", "evaluate_mesh.py")
LENGAN = ["hunyuan_m130_elev0", "hunyuan_m130_elev20"]
OBJEK = [f"obj{i:02d}" for i in range(1, 31)]
EKSTRAKTOR = {"sn": "_sn.glb", "mc": "_mc.ply"}
METRIK = ["chamfer", "fscore@0.01", "fscore@0.02", "normal_consistency", "volume_iou"]
NOISE_FLOOR = 0.0044


def ground_truth(oid):
    k = sorted(glob.glob(os.path.join(BASE, "dataset", f"{oid}_*.stl")))
    if len(k) != 1:
        sys.exit(f"BERHENTI: {len(k)} ground truth cocok dataset/{oid}_*.stl (harus tepat 1)")
    return k[0]


def tugas():
    t = []
    for lengan in LENGAN:
        for oid in OBJEK:
            for eks, suf in EKSTRAKTOR.items():
                p = os.path.join(DIR_UJI, lengan, oid + suf)
                if os.path.exists(p):
                    t.append((f"{lengan}__{eks}", oid, p))
    return t


def baca(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluasi(cek):
    semua = tugas()
    selesai = {(r["model"], r["gt"][:5]) for r in baca(CSV_KELUARAN)}
    sisa = [x for x in semua if (x[0], x[1]) not in selesai]
    print("=" * 72)
    for lengan in LENGAN:
        for eks in EKSTRAKTOR:
            n = sum(1 for m, _, _ in semua if m == f"{lengan}__{eks}")
            print(f"  {lengan}__{eks:3s} berkas {n:2d}/30")
    print(f"  sudah di CSV {len(semua) - len(sisa)} | sisa {len(sisa)}")
    print("=" * 72)
    if cek:
        return
    t0 = time.time()
    gagal = 0
    for i, (model, oid, pred) in enumerate(sisa, 1):
        cmd = [sys.executable, EVALUATOR, "--gt", ground_truth(oid), "--pred", pred,
               "--model", model, "--csv", CSV_KELUARAN]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            gagal += 1
            print(f"  [{i:3d}/{len(sisa)}] {model}/{oid} GAGAL\n{r.stderr[-400:]}")
        else:
            print(f"  [{i:3d}/{len(sisa)}] {model}/{oid} ok")
        if i % 10 == 0:
            print(f"      ~{(time.time() - t0) / i * (len(sisa) - i) / 60:.1f} menit lagi")
    if gagal:
        sys.exit(f"BERHENTI: {gagal} evaluasi gagal.")


def analisis():
    import numpy as np
    from scipy.stats import wilcoxon

    out = []

    def p(s=""):
        print(s)
        out.append(s)

    def holm(ps):
        ps = np.asarray(ps, float)
        o = np.argsort(ps)
        adj = np.empty(len(ps))
        run = 0.0
        for i, k in enumerate(o):
            run = max(run, (len(ps) - i) * ps[k])
            adj[k] = min(1.0, run)
        return adj

    baris = {(r["model"], r["gt"][:5]): r for r in baca(CSV_KELUARAN)}
    lama = {(r["model"], r["gt"][:5]): r for r in baca(CSV_LAMA)}
    lengkap = all((f"{l}__{e}", o) in baris for l in LENGAN for e in EKSTRAKTOR for o in OBJEK)
    p("=" * 78)
    p("ANALISIS UJI B: surface net (ComfyUI) vs marching cubes, grid voxel yang sama, seed 42")
    p("=" * 78)
    if not lengkap:
        hilang = [(l, e, o) for l in LENGAN for e in EKSTRAKTOR for o in OBJEK
                  if (f"{l}__{e}", o) not in baris]
        p(f"BELUM LENGKAP: {len(hilang)} mesh belum dievaluasi, mis. {hilang[:4]}")
        p("Analisis parsial di bawah TIDAK boleh dipakai untuk keputusan.")

    p("\n[G0] Determinisme: SN ulang vs GLB lama (hasil_evaluasi_n30.csv)")
    g0_gagal = []
    for l in LENGAN:
        for o in OBJEK:
            a, b = baris.get((f"{l}__sn", o)), lama.get((l, o))
            if not a or not b:
                continue
            same_tri = a["pred_triangles"] == b["pred_triangles"]
            dcd = abs(float(a["chamfer"]) - float(b["chamfer"]))
            if not same_tri or dcd > 0.0005:
                g0_gagal.append((l, o, a["pred_triangles"], b["pred_triangles"], round(dcd, 6)))
    n_g0 = sum(1 for l in LENGAN for o in OBJEK if (f"{l}__sn", o) in baris)
    p(f"     diperiksa {n_g0} | gagal {len(g0_gagal)}")
    for x in g0_gagal[:10]:
        p(f"     GAGAL {x}")
    if g0_gagal:
        p("     => G0 GAGAL: grid voxel belum tentu sama dengan naskah. Keputusan di bawah TIDAK berlaku.")

    def ambil(l, e, kol, cast=float):
        return [cast(baris[(f"{l}__{e}", o)][kol]) for o in OBJEK if (f"{l}__{e}", o) in baris]

    p("\n[T] Topologi per ekstraktor (median, [min-maks]; watertight = jumlah mesh)")
    tot = {}
    for e in EKSTRAKTOR:
        wt_all, nm_all = 0, []
        for l in LENGAN:
            nm = ambil(l, e, "pred_nonmanifold_edges", int)
            bd = ambil(l, e, "pred_boundary_edges", int)
            cp = ambil(l, e, "pred_components", int)
            wt = sum(v == "True" for v in ambil(l, e, "pred_watertight", str))
            wt_all += wt
            nm_all += nm
            if nm:
                p(f"     {e} {l:20s} watertight {wt:2d}/{len(nm)} | non-manifold {np.median(nm):8.1f} "
                  f"[{min(nm)}-{max(nm)}] | boundary {np.median(bd):6.1f} [{min(bd)}-{max(bd)}] "
                  f"| komponen {np.median(cp):.0f}")
        tot[e] = (wt_all, float(np.median(nm_all)) if nm_all else float("nan"), len(nm_all))
        p(f"     {e} TOTAL watertight {wt_all}/{len(nm_all)} | median non-manifold {tot[e][1]:.1f}")

    p("\n[S1] Selisih geometri MC - SN pada grid yang sama (per mesh)")
    d = []
    for l in LENGAN:
        for o in OBJEK:
            a, b = baris.get((f"{l}__mc", o)), baris.get((f"{l}__sn", o))
            if a and b:
                d.append(float(a["chamfer"]) - float(b["chamfer"]))
    if d:
        d = np.array(d)
        p(f"     n {len(d)} | median |dCD| {np.median(np.abs(d)):.5f} | maks |dCD| {np.abs(d).max():.5f} "
          f"| median dCD {np.median(d):+.5f} | noise floor {NOISE_FLOOR}")

    p("\n[S2] Efek elevasi 0 vs 20 derajat per ekstraktor (Wilcoxon exact dua sisi, Holm atas 5 metrik)")
    s2 = {}
    for e in EKSTRAKTOR:
        ok = [o for o in OBJEK if (f"{LENGAN[0]}__{e}", o) in baris and (f"{LENGAN[1]}__{e}", o) in baris]
        if len(ok) < 10:
            continue
        ps, teks = [], []
        for m in METRIK:
            a = np.array([float(baris[(f"{LENGAN[0]}__{e}", o)][m]) for o in ok])
            b = np.array([float(baris[(f"{LENGAN[1]}__{e}", o)][m]) for o in ok])
            pv = wilcoxon(a, b, method="exact" if len(ok) <= 50 else "auto").pvalue
            ps.append(pv)
            naik = int(((a - b) > 0).sum()) if m == "chamfer" else int(((b - a) > 0).sum())
            teks.append((m, a.mean(), b.mean(), naik, len(ok)))
        adj = holm(ps)
        s2[e] = dict(zip(METRIK, adj))
        for (m, a0, b0, naik, n), h in zip(teks, adj):
            p(f"     {e} {m:20s} 0deg {a0:.4f} -> 20deg {b0:.4f} | membaik {naik}/{n} | Holm p {h:.4f}")

    p("\n[S3] Topologi 0 vs 20 derajat di bawah MC (Wilcoxon, deskriptif)")
    for kol in ("pred_nonmanifold_edges", "pred_boundary_edges"):
        a = np.array(ambil(LENGAN[0], "mc", kol, int))
        b = np.array(ambil(LENGAN[1], "mc", kol, int))
        if len(a) == len(b) and len(a) >= 10 and np.any(a != b):
            p(f"     {kol}: median {np.median(a):.1f} -> {np.median(b):.1f} | p {wilcoxon(a, b).pvalue:.3f}")
        elif len(a) == len(b) and len(a):
            p(f"     {kol}: median {np.median(a):.1f} -> {np.median(b):.1f} | semua pasangan sama, uji tidak berlaku")

    p("\n[KEPUTUSAN] aturan terdaftar di rencana 23 Sep")
    if not lengkap or g0_gagal:
        p("     TIDAK DIAMBIL: data belum lengkap atau G0 gagal.")
    else:
        wt_mc, nm_mc, n_mc = tot["mc"]
        if wt_mc <= 6 and nm_mc >= 100:
            k = "R-A  klaim topologi BERTAHAN: mesh MC juga tidak valid. Judul boleh dipertahankan; tambahkan kalimat hasil uji ini."
        elif wt_mc >= 30 or nm_mc == 0:
            k = "R-B  ARTEFAK EKSTRAKTOR: mesh MC dari grid yang sama sebagian besar valid. Klaim topologi turun dari judul, abstrak, dan kesimpulan."
        else:
            k = "R-C  SEBAGIAN: laporkan kedua ekstraktor; klaim topologi diturunkan menjadi temuan tentang pipeline, bukan tentang model."
        p(f"     MC watertight {wt_mc}/{n_mc}, median non-manifold {nm_mc:.1f}")
        p(f"     => {k}")
        p("\n[PREDIKSI] dicocokkan dengan rencana 23 Sep")
        p(f"     P1 R-B berlaku (MC watertight >= 30/60)          : {'TERBUKTI' if wt_mc >= 30 else 'MELESET'} ({wt_mc}/{n_mc})")
        p(f"     P2 median non-manifold MC < 10                   : {'TERBUKTI' if nm_mc < 10 else 'MELESET'} ({nm_mc:.1f})")
        if len(d):
            md = float(np.median(np.abs(d)))
            p(f"     P3 median |dCD| MC-SN < {NOISE_FLOOR}              : {'TERBUKTI' if md < NOISE_FLOOR else 'MELESET'} ({md:.5f})")
        if "mc" in s2:
            h = s2["mc"]["chamfer"]
            p(f"     P4 efek elevasi Chamfer di bawah MC Holm p < 0.05 : {'TERBUKTI' if h < 0.05 else 'MELESET'} ({h:.4f})")

    with open(TXT_ANALISIS, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nditulis: {TXT_ANALISIS}")


if __name__ == "__main__":
    if not os.path.exists(EVALUATOR):
        sys.exit(f"BERHENTI: evaluator tidak ada: {EVALUATOR}")
    if "--analisis" in sys.argv[1:]:
        analisis()
    else:
        evaluasi("--cek" in sys.argv[1:])
