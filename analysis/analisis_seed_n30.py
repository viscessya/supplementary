#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-specified analysis of the three-seed replication (seeds 42/123/777, 30 components, 0 vs 20 degrees).

Gates, tests, and predictions were fixed before the new seeds were generated
(written plan dated 17 Sep 2026). Analyses added afterwards are reported under a separate,
clearly marked post hoc heading.

Inputs:  eval/hasil_evaluasi_n30.csv (seed 42), eval/hasil_seed_n30.csv (seeds 123 and 777),
         eval/hasil_evaluasi.csv (optional: earlier replicate runs, for the evaluator-stability check)
Outputs: eval/analisis_seed_n30.txt, eval/seed_n30_per_objek.csv
         (--parsial writes *_PARSIAL.* files for dry runs; those numbers are not reported)

Usage: python eval/analisis_seed_n30.py [--parsial]
"""
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(BASE, "eval")
PARSIAL = "--parsial" in sys.argv[1:]
SUF = "_PARSIAL" if PARSIAL else ""
OUT_TXT = os.path.join(EVAL, f"analisis_seed_n30{SUF}.txt")
OUT_CSV = os.path.join(EVAL, f"seed_n30_per_objek{SUF}.csv")

OBJEK = [f"obj{i:02d}" for i in range(1, 31)]
SEEDS = [42, 123, 777]
ELEV = [0, 20]
METRIK = [("chamfer", "Chamfer", False), ("fscore@0.01", "F@0.01", True),
          ("fscore@0.02", "F@0.02", True), ("normal_consistency", "Normal cons.", True),
          ("volume_iou", "Volume IoU", True)]
PENDORONG = ["obj25", "obj24", "obj20", "obj05"]
OBJEK_13_LAMA = ["obj01", "obj02", "obj03", "obj04", "obj05", "obj06", "obj07", "obj08",
                 "obj20", "obj21", "obj22", "obj24", "obj25"]
GERBANG_OBJEK = "obj05"
GERBANG_TOL_CD = 0.0005          # ditetapkan sebelum gerbang dijalankan
SUF_E4 = "__e4lama"

L = []


def tulis(s=""):
    print(s)
    L.append(s)


def baca(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lengan(e, s):
    return f"hunyuan_m130_elev{e}" if s == 42 else f"hunyuan_m130_elev{e}_seed{s}"


def holm(ps):
    idx = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(idx):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def wilcoxon(d):
    d = np.asarray(d, float)
    ada_nol = np.any(d == 0)
    ada_seri = len(np.unique(np.abs(d[d != 0]))) < np.sum(d != 0)
    metode = "exact" if not (ada_nol or ada_seri) else "auto"
    return stats.wilcoxon(d, method=metode).pvalue, metode


def hodges_lehmann(d, conf=0.95):
    d = np.asarray(d, float)
    n = len(d)
    w = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    k = int(math.floor(n * (n + 1) / 4 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(k, 0)
    return float(np.median(w)), float(w[k]), float(w[len(w) - 1 - k])


def main():
    base = baca(os.path.join(EVAL, "hasil_evaluasi_n30.csv"))
    p_seed = os.path.join(EVAL, "hasil_seed_n30.csv")
    if not os.path.exists(p_seed):
        sys.exit("BERHENTI: eval/hasil_seed_n30.csv belum ada. Jalankan eval/evaluasi_seed_n30.py dulu.")
    seedrows = baca(p_seed)

    R = {}
    for r in base:
        for e in ELEV:
            if r["model"] == lengan(e, 42):
                R[(r["model"], r["gt"][:5])] = r
    for r in seedrows:
        R[(r["model"], r["gt"][:5])] = r

    tulis("=" * 78)
    tulis("ANALISIS SEED n30" + ("   *** MODE PARSIAL -- BUKAN UNTUK NASKAH ***" if PARSIAL else ""))
    tulis("=" * 78)

    tulis(f"\n[G1] Gerbang reprodusibilitas: {GERBANG_OBJEK}, kanonik (PC sekarang) vs E4 lama (3 Agu), toleransi |dCD| <= {GERBANG_TOL_CD}")
    g_lulus, g_ada = True, 0
    for e in ELEV:
        for s in (123, 777):
            a = R.get((lengan(e, s), GERBANG_OBJEK))
            b = R.get((lengan(e, s) + SUF_E4, GERBANG_OBJEK))
            if a is None or b is None:
                tulis(f"     {lengan(e, s):28s} belum lengkap (kanonik={'ada' if a else 'tidak'}, e4={'ada' if b else 'tidak'})")
                g_lulus = False
                continue
            g_ada += 1
            d = abs(float(a["chamfer"]) - float(b["chamfer"]))
            ok = d <= GERBANG_TOL_CD
            g_lulus &= ok
            tulis(f"     {lengan(e, s):28s} kanonik {float(a['chamfer']):.6f} | e4 {float(b['chamfer']):.6f} | |d|={d:.6f} {'LULUS' if ok else 'GAGAL'}")
    pakai_e4 = g_lulus and g_ada == 4
    tulis(f"     => {'LULUS: GLB E4 lama boleh dipakai untuk obj01-08 yang belum digenerate ulang' if pakai_e4 else 'TIDAK LULUS / belum lengkap: GLB E4 lama TIDAK dipakai'}")

    p_lama = os.path.join(EVAL, "hasil_evaluasi.csv")
    if os.path.exists(p_lama):
        lama = {(r["model"], r["gt"][:5]): float(r["chamfer"]) for r in baca(p_lama)}
        beda = [abs(float(r["chamfer"]) - lama[(r["model"][:-len(SUF_E4)], r["gt"][:5])])
                for r in seedrows if r["model"].endswith(SUF_E4)
                and (r["model"][:-len(SUF_E4)], r["gt"][:5]) in lama]
        if beda:
            tulis(f"[G1b] Evaluator sekarang vs CSV 3 Agu untuk GLB E4 yang sama: {len(beda)} pasangan, |dCD| maks {max(beda):.2e}")

    T = {}  # (oid, e, s) -> baris
    sumber = defaultdict(set)
    for oid in OBJEK:
        for e in ELEV:
            for s in SEEDS:
                r = R.get((lengan(e, s), oid))
                src = "kanonik" if s != 42 else "seed42"
                if r is None and s != 42 and pakai_e4:
                    r = R.get((lengan(e, s) + SUF_E4, oid))
                    src = "e4lama"
                if r is not None:
                    T[(oid, e, s)] = r
                    sumber[oid].add(src)
    lengkap = [o for o in OBJEK if all((o, e, s) in T for e in ELEV for s in SEEDS)]
    tulis(f"\n[G2] Kelengkapan: {len(lengkap)}/30 objek punya 6 sel (2 elevasi x 3 seed)")
    kurang = [o for o in OBJEK if o not in lengkap]
    if kurang:
        tulis(f"     kurang: {', '.join(kurang)}")
        if not PARSIAL:
            tulis("BERHENTI: data belum lengkap. Pakai --parsial hanya untuk uji jalan.")
            simpan()
            sys.exit(1)
    pakai_e4_obj = sorted(o for o in lengkap if "e4lama" in sumber[o])
    tulis(f"     objek memakai GLB E4 lama: {', '.join(pakai_e4_obj) if pakai_e4_obj else '-'}")
    objs = lengkap
    n = len(objs)

    def nilai(oid, e, s, col):
        return float(T[(oid, e, s)][col])

    def perbaikan(oid, s, col, lebih_besar_baik):
        d = nilai(oid, 20, s, col) - nilai(oid, 0, s, col)
        return d if lebih_besar_baik else -d     # positif = 20 derajat lebih baik

    tulis(f"\n[P0] Pembanding seed 42 saja (n={n}) -- harus sama dengan Table 3 kalau n=30")
    for col, lab, hb in METRIK[:1]:
        m0 = np.mean([nilai(o, 0, 42, col) for o in objs]); m20 = np.mean([nilai(o, 20, 42, col) for o in objs])
        tulis(f"     {lab}: 0deg {m0:.4f} -> 20deg {m20:.4f}, rerata perbaikan {m0 - m20:.4f}")

    tulis(f"\n[U] UJI UTAMA: rerata 3 seed per objek, Wilcoxon dua sisi, Holm atas 5 metrik (n={n})")
    ps, baris = [], []
    for col, lab, hb in METRIK:
        m0 = np.array([np.mean([nilai(o, 0, s, col) for s in SEEDS]) for o in objs])
        m20 = np.array([np.mean([nilai(o, 20, s, col) for s in SEEDS]) for o in objs])
        d = (m20 - m0) if hb else (m0 - m20)
        p, met = wilcoxon(d)
        ps.append(p)
        baris.append((lab, m0.mean(), m20.mean(), d, p, met))
    adj = holm(ps)
    for (lab, a, b, d, p, met), pa in zip(baris, adj):
        tulis(f"     {lab:13s} 0deg {a:.4f} -> 20deg {b:.4f} | membaik {int(np.sum(d > 0))}/{n} | p={p:.4f} ({met}) Holm p={pa:.4f}")
    d_cd = baris[0][3]
    hl, lo, hi = hodges_lehmann(d_cd)
    tulis(f"     Hodges-Lehmann perbaikan Chamfer: {hl:.4f} (95% CI {lo:.4f} .. {hi:.4f}); rerata {d_cd.mean():.4f}")

    tulis("\n[S1] Tiap seed terpisah: Wilcoxon + Holm atas 5 metrik")
    for s in SEEDS:
        pp = []
        for col, lab, hb in METRIK:
            d = np.array([perbaikan(o, s, col, hb) for o in objs])
            pp.append(wilcoxon(d)[0])
        aa = holm(pp)
        dcd = np.array([perbaikan(o, s, "chamfer", False) for o in objs])
        tulis(f"     seed {s:3d}: rerata perbaikan CD {dcd.mean():.4f} | Holm p " +
              " ".join(f"{lab}={x:.3f}" for (_, lab, _), x in zip(METRIK, aa)) +
              f" | lolos<0.05: {int(np.sum(aa < 0.05))}/5")

    tulis("\n[S2] Per objek: arah perbaikan Chamfer di 3 seed dan rasio efek / rentang seed")
    per_obj = []
    for o in objs:
        dd = [perbaikan(o, s, "chamfer", False) for s in SEEDS]
        c0 = [nilai(o, 0, s, "chamfer") for s in SEEDS]
        c20 = [nilai(o, 20, s, "chamfer") for s in SEEDS]
        efek = abs(np.mean(c0) - np.mean(c20))
        rentang = max(np.ptp(c0), np.ptp(c20))
        per_obj.append(dict(objek=o, dCD_s42=dd[0], dCD_s123=dd[1], dCD_s777=dd[2],
                            dCD_rerata=float(np.mean(dd)),
                            arah="3 membaik" if all(x > 0 for x in dd) else
                                 ("3 memburuk" if all(x < 0 for x in dd) else "campur"),
                            sd_cd_0=float(np.std(c0, ddof=1)), sd_cd_20=float(np.std(c20, ddof=1)),
                            rasio_efek_rentang=efek / rentang if rentang > 0 else float("inf"),
                            sumber="+".join(sorted(sumber[o]))))
    for k in ("3 membaik", "3 memburuk", "campur"):
        tulis(f"     {k:10s}: {sum(1 for x in per_obj if x['arah'] == k)}/{n}")
    tulis(f"     rasio efek/rentang > 1: {sum(1 for x in per_obj if x['rasio_efek_rentang'] > 1)}/{n}")

    urut = sorted(per_obj, key=lambda x: -x["dCD_rerata"])
    tot = sum(x["dCD_rerata"] for x in per_obj)
    top4 = [x["objek"] for x in urut[:4]]
    tulis("\n[S3] Konsentrasi (rerata 3 seed)")
    enam = ", ".join("%s %.4f" % (x["objek"], x["dCD_rerata"]) for x in urut[:6])
    tulis(f"     6 teratas: {enam} ...")
    tulis(f"     porsi 4 teratas dari total perbaikan: {sum(x['dCD_rerata'] for x in urut[:4]) / tot * 100:.0f}% (seed 42 di naskah: 85%)")
    tulis(f"     4 pendorong naskah masih 4 teratas: {sorted(top4) == sorted(PENDORONG)}")
    tanpa = np.array([x["dCD_rerata"] for x in per_obj if x["objek"] not in PENDORONG])
    tulis(f"     tanpa 4 pendorong (n={len(tanpa)}): Wilcoxon p={wilcoxon(tanpa)[0]:.4f}, membaik {int(np.sum(tanpa > 0))}/{len(tanpa)}")

    tulis("\n[S4] Sebaran antar-seed (SD Chamfer) di 0 vs 20 derajat")
    baru = [x for x in per_obj if x["objek"] not in OBJEK_13_LAMA]
    for label, sel in (("KONFIRMATORI: 17 objek yang belum pernah dicek seed", baru),
                       ("deskriptif: seluruh objek", per_obj)):
        if len(sel) < 5:
            tulis(f"     {label}: n={len(sel)} terlalu kecil, dilewati")
            continue
        dsd = np.array([x["sd_cd_0"] - x["sd_cd_20"] for x in sel])
        tulis(f"     {label} (n={len(sel)}): SD0 > SD20 pada {int(np.sum(dsd > 0))}/{len(sel)}, "
              f"median SD0 {np.median([x['sd_cd_0'] for x in sel]):.4f} vs SD20 {np.median([x['sd_cd_20'] for x in sel]):.4f}, "
              f"Wilcoxon p={wilcoxon(dsd)[0]:.4f}")

    if not PARSIAL:
        tulis("\n[PREDIKSI] dicocokkan otomatis dengan yang tertulis di rencana 17 Sep")
        tulis(f"     P1 rerata perbaikan CD 3-seed < 0.0126            : {d_cd.mean():.4f} -> {'TERBUKTI' if d_cd.mean() < 0.0126 else 'MELESET'}")
        tulis(f"     P2 uji utama Chamfer Holm p < 0.05                : {adj[0]:.4f} -> {'TERBUKTI' if adj[0] < 0.05 else 'MELESET'}")
        tulis(f"     P3a 4 pendorong tetap 4 teratas                   : {'TERBUKTI' if sorted(top4) == sorted(PENDORONG) else 'MELESET'}")
        porsi = sum(x['dCD_rerata'] for x in urut[:4]) / tot
        tulis(f"     P3b porsi 4 teratas < 85%                          : {porsi * 100:.0f}% -> {'TERBUKTI' if porsi < 0.85 else 'MELESET'}")
        if len(baru) >= 5:
            p4 = wilcoxon(np.array([x['sd_cd_0'] - x['sd_cd_20'] for x in baru]))[0]
            tulis(f"     P4 S4 konfirmatori TIDAK signifikan (p >= 0.05)   : p={p4:.4f} -> {'TERBUKTI' if p4 >= 0.05 else 'MELESET'}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_obj[0].keys()))
        w.writeheader()
        for x in per_obj:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()})
    simpan()


def simpan():
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nTersimpan: {OUT_TXT}")


if __name__ == "__main__":
    main()
