# -*- coding: utf-8 -*-
"""Surface-visibility figures and the log of every visibility test that was run.

  figur/fig_visibilitas_A_peta.png        best viewing angle on the reference surface (six components, 0 and 20 degrees)
  figur/fig_visibilitas_B_histogram.png   area share per viewing angle, four driver components vs the other 26
  figur/fig_visibilitas_C_vs_chamfer.png  (a) visible area gained vs Chamfer improvement, n = 30
                                          (b) within-component link across five elevations, with a permutation test
  eval/ringkasan_uji_visibilitas.txt      every test tried, including those that failed

Requires eval/visibilitas_per_objek.csv (from hitung_visibilitas.py). Reads dataset/ and eval/ only.
Usage: python eval/plot_peta_histogram_visibilitas.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

import hitung_visibilitas as hv

HERE = hv.HERE
ROOT = hv.ROOT
EVAL = os.path.join(ROOT, "eval")
FIGUR = os.path.join(ROOT, "figur")

ELEV = [0, 10, 20, 30, 40]
PENDORONG = ["obj25", "obj24", "obj20", "obj05"]          # manuskrip Sec. 4.1 (85% perbaikan)
PETA_OBJEK = ["obj25", "obj20", "obj05", "obj24", "obj22", "obj15"]

BIRU = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
CMAP_THETA = LinearSegmentedColormap.from_list("theta", BIRU)   # 0 deg gelap -> 90 deg terang
ORANYE = "#eb6834"
ABU_TEKS = "#52514e"
GRID_WARNA = "#e4e3df"
PERMUKAAN = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#0b0b0b", "xtick.color": ABU_TEKS, "ytick.color": ABU_TEKS,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.dpi": 300,
})


def nama_komponen():
    with open(os.path.join(EVAL, "dataset_manifest_supplementary.csv"), encoding="utf-8") as f:
        return {r["id"]: r["component"] for r in csv.DictReader(f)}


def baca_visibilitas():
    V = {}
    with open(os.path.join(HERE, "visibilitas_per_objek.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            V[(r["objek"], int(r["elevasi"]))] = {k: float(v) for k, v in r.items()
                                                  if k not in ("objek", "elevasi")}
    return V


def baca_chamfer():
    C = {e: {} for e in ELEV}
    with open(os.path.join(EVAL, "hasil_evaluasi_n30.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for e in ELEV:
                if r["model"] == f"hunyuan_m130_elev{e}":
                    C[e][r["gt"][:5]] = float(r["chamfer"])
    return C


def dcd_tiga_seed(C, objs):
    """DITAMBAHKAN 17 Sep 2026: perbaikan Chamfer 0->20 derajat dirata-rata seed 42/123/777.
    Seed 42 dari hasil_evaluasi_n30.csv (C), seed 123/777 dari hasil_seed_n30.csv
    (lengan kanonik; kalau tidak ada, lengan __e4lama -- sah karena gerbang G1 lulus).
    Mengembalikan None kalau hasil_seed_n30.csv belum ada atau belum lengkap."""
    p = os.path.join(EVAL, "hasil_seed_n30.csv")
    if not os.path.exists(p):
        return None
    R = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            R[(r["model"], r["gt"][:5])] = float(r["chamfer"])
    out = []
    for o in objs:
        d = [C[0][o] - C[20][o]]
        for s_ in (123, 777):
            v = []
            for e in (0, 20):
                m = f"hunyuan_m130_elev{e}_seed{s_}"
                x = R.get((m, o), R.get((m + "__e4lama", o)))
                if x is None:
                    return None
                v.append(x)
            d.append(v[0] - v[1])
        out.append(np.mean(d))
    return np.array(out)


def noise_floor():
    with open(os.path.join(EVAL, "noise_floor.csv"), encoding="utf-8") as f:
        return float(np.mean([float(r["chamfer"]) for r in csv.DictReader(f)]))


def proyeksi_tampilan(titik, az=35, el=30, res=520):
    """Proyeksi ortografik untuk kamera pengamat (bukan kamera rig) + z-buffer 2x2 piksel."""
    w = hv.arah_kamera(az, el)
    u, v = hv.basis_ortonormal(w)
    x, y, d = titik @ u, titik @ v, -(titik @ w)
    xi = np.clip(((x + 0.8) / 1.6 * res).astype(int), 0, res - 2)
    yi = np.clip(((y + 0.8) / 1.6 * res).astype(int), 0, res - 2)
    zbuf = np.full((res, res), np.inf)
    owner = np.full((res, res), -1)
    urut = np.argsort(-d)                       # jauh -> dekat; yang dekat menimpa
    for dx in (0, 1):
        for dy in (0, 1):
            X, Y = xi[urut] + dx, yi[urut] + dy
            zbuf[Y, X] = np.minimum(zbuf[Y, X], d[urut])
            owner[Y, X] = urut
    return owner


def warna_theta(th):
    rgb = np.ones((len(th), 3))
    ok = ~np.isnan(th)
    rgb[ok] = CMAP_THETA(np.clip(th[ok] / 90.0, 0, 1))[:, :3]
    rgb[~ok] = matplotlib.colors.to_rgb(ORANYE)
    return rgb


def figur_A(V, nama):
    """Tata letak lebar untuk naskah: kolom = objek, baris = elevasi rig (0 dan 20 derajat)."""
    import glob
    import textwrap
    stl = {os.path.basename(p).split("_")[0]: p
           for p in sorted(glob.glob(os.path.join(hv.DATASET, "obj*.stl")))}
    n = len(PETA_OBJEK)
    fig, ax = plt.subplots(2, n, figsize=(7.2, 3.30))
    for j, obj in enumerate(PETA_OBJEK):
        m = hv.muat_ternormalisasi(stl[obj])
        titik, normal = hv.sampel(m, n=900_000, seed=7)
        owner = proyeksi_tampilan(titik)
        mask = owner >= 0
        ys, xs = np.where(mask)
        pad = 10
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
        for i, e in enumerate([0, 20]):
            th = hv.theta_rig(titik, normal, e)
            img = np.ones(owner.shape + (3,))
            img[mask] = warna_theta(th[owner[mask]])
            a = ax[i, j]
            a.imshow(img[y0:y1, x0:x1], origin="lower", interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            tl = V[(obj, e)]["terlihat"] * 100
            a.set_xlabel(f"visible {tl:.0f}%", fontsize=8.96, color=ABU_TEKS, labelpad=1)
            if i == 0:
                judul = "\n".join(textwrap.wrap(nama[obj], 14))
                a.set_title(f"{obj}\n{judul}", fontsize=8.96, pad=3)
        for i, e in enumerate([0, 20]):
            ax[i, 0].set_ylabel(f"{e}°", fontsize=10.88, rotation=0, ha="right", va="center", labelpad=8)
    fig.text(0.055, 0.106, "Ground-truth CAD surfaces, not reconstructions", fontsize=9.6,
             color=ABU_TEKS, va="center")
    cax = fig.add_axes([0.52, 0.096, 0.25, 0.026])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 90), cmap=CMAP_THETA),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 30, 60, 90]); cb.outline.set_visible(False)
    cb.ax.set_xticklabels(["0\u00b0\nface-on", "30\u00b0", "60\u00b0", "90\u00b0\ngrazing"])
    cb.ax.tick_params(labelsize=9, pad=2)
    fig.patches.append(matplotlib.patches.Rectangle((0.825, 0.093), 0.022, 0.033,
                       transform=fig.transFigure, color=ORANYE))
    fig.text(0.855, 0.099, "not visible", fontsize=9.6)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.85, bottom=0.216, wspace=0.08, hspace=0.3)
    out = os.path.join(FIGUR, "fig_visibilitas_A_peta.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def figur_B(V, objs):
    kelompok = [("4 driver components", PENDORONG),
                ("26 remaining components", [o for o in objs if o not in PENDORONG])]
    kat = hv.BIN_LABEL + ["not\nvisible"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for a, (judul, sel) in zip(axs, kelompok):
        x = np.arange(len(kat))
        for k, (e, warna, geser) in enumerate([(0, "#9ec5f4", -0.2), (20, "#184f95", 0.2)]):
            nilai = [np.mean([V[(o, e)][f"luas_theta_{l}"] for o in sel]) * 100 for l in hv.BIN_LABEL]
            nilai.append(np.mean([V[(o, e)]["tak_terlihat"] for o in sel]) * 100)
            a.bar(x + geser, nilai, width=0.38, color=warna, label=f"{e}°",
                  edgecolor="white", linewidth=1.0)
        a.set_xticks(x); a.set_xticklabels([k.replace("-", "–") for k in kat], fontsize=7.5)
        a.set_title(judul, fontsize=9)
        a.yaxis.grid(True, color=GRID_WARNA, linewidth=0.6); a.set_axisbelow(True)
        a.set_xlabel("best viewing angle θ (°)")
    axs[0].set_ylabel("share of surface area (%)")
    axs[0].legend(title="rig elevation", frameon=False, fontsize=7.5, title_fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIGUR, "fig_visibilitas_B_histogram.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def figur_C(V, C, objs, nama, floor, log):
    dV = np.array([(V[(o, 20)]["terlihat"] - V[(o, 0)]["terlihat"]) * 100 for o in objs])
    dCD = np.array([C[0][o] - C[20][o] for o in objs])
    d3 = dcd_tiga_seed(C, objs)
    label_y = "Chamfer improvement, 0°→20°"
    if d3 is not None:
        dCD = d3
        label_y = "Chamfer improvement, 0°→20°\n(mean of 3 seeds)"
        log.append("[C-a] panel (a) memakai rerata 3 seed (42/123/777) dari hasil_seed_n30.csv")

    T = np.array([[V[(o, e)]["terlihat"] for e in ELEV] for o in objs])
    Cm = -np.array([[C[e][o] for e in ELEV] for o in objs])
    rho_obj = np.array([stats.spearmanr(T[i], Cm[i])[0] for i in range(len(objs))])
    rng = np.random.default_rng(1)
    sims = np.array([np.nanmedian([stats.spearmanr(T[p], Cm[i])[0]
                                   for i, p in enumerate(rng.permutation(len(objs)))])
                     for _ in range(5000)])
    obs = np.nanmedian(rho_obj)
    p_perm = float(np.mean(sims >= obs))
    log.append(f"[C-b] rho per objek (terlihat vs -CD, 5 elevasi): median={obs:.3f}, "
               f"positif {int(np.sum(rho_obj > 0))}/30; permutasi 5000x median={np.median(sims):.3f}, "
               f"p={p_perm:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    a = axs[0]
    a.axhspan(-floor, floor, color="#f0efec", zorder=0)
    a.text(-8.5, -floor * 1.4, "noise floor", fontsize=8.75, color=ABU_TEKS, ha="left", va="top")
    a.axhline(0, color="#8a8984", linewidth=0.6)
    lain = [i for i, o in enumerate(objs) if o not in PENDORONG]
    drv = [i for i, o in enumerate(objs) if o in PENDORONG]
    a.scatter(dV[lain], dCD[lain], s=22, color="#86b6ef", edgecolor="white", linewidth=0.8,
              label="other components", zorder=3)
    a.scatter(dV[drv], dCD[drv], s=30, color="#184f95", edgecolor="white", linewidth=0.8,
              label="driver components", zorder=4)
    for o in PENDORONG + ["obj15", "obj22", "obj13", "obj14"]:
        i = objs.index(o)
        geser_lbl = {"obj24": (-30, 5), "obj14": (5, -9)}.get(o, (4, 3))
        a.annotate(o, (dV[i], dCD[i]), xytext=geser_lbl, textcoords="offset points",
                   fontsize=8.75, color="#0b0b0b")
    rho, p = stats.spearmanr(dV, dCD)
    a.set_xlabel("visible surface gained, 0°→20° (percentage points)")
    a.set_ylabel(label_y)
    a.set_title("(a) Across components (n = 30)", fontsize=11.25, loc="left")
    a.text(0.02, 0.70, f"Spearman ρ = {rho:.2f}, p = {p:.4f}", transform=a.transAxes, fontsize=9.375, color=ABU_TEKS)
    a.legend(frameon=False, fontsize=8.75, loc="upper left")
    a.yaxis.grid(True, color=GRID_WARNA, linewidth=0.6); a.set_axisbelow(True)

    b = axs[1]
    nilai, hit = np.unique(np.round(sims, 3), return_counts=True)
    b.bar(nilai, hit, width=0.022, color="#9ec5f4", edgecolor="white", linewidth=0.5,
          label="shuffled components")
    b.axvline(obs, color="#184f95", linewidth=2, label=f"observed ({obs:.2f})")
    b.set_xlabel("median within-component ρ\n(visible share vs −Chamfer, 5 elevations)")
    b.set_ylabel("permutations (of 5,000)")
    b.set_title("(b) Within components", fontsize=11.25, loc="left")
    b.set_ylim(0, hit.max() * 1.65)
    b.text(0.03, 0.74, f"permutation p = {p_perm:.4f}", transform=b.transAxes, fontsize=9.375, color=ABU_TEKS)
    b.legend(frameon=False, fontsize=8.75, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGUR, "fig_visibilitas_C_vs_chamfer.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def semua_uji(V, C, objs, floor, log):
    """Semua uji yang pernah dicoba pada analisis 16 Sep dilaporkan, termasuk yang lemah."""
    F = {r["objek"]: r for r in csv.DictReader(open(os.path.join(EVAL, "fitur_objek.csv"), encoding="utf-8"))}
    dCD = np.array([C[0][o] - C[20][o] for o in objs])
    ukur = np.abs(dCD) > floor
    keep = np.array([o not in PENDORONG for o in objs])
    kandidat = {
        "delta_terlihat_0_20": [V[(o, 20)]["terlihat"] - V[(o, 0)]["terlihat"] for o in objs],
        "luas_atas_saja (fitur_objek.csv)": [float(F[o]["luas_atas_saja"]) for o in objs],
        "delta_evidensi_0_20": [V[(o, 20)]["evidensi"] - V[(o, 0)]["evidensi"] for o in objs],
        "menyerempet_atau_tak_terlihat_0 (theta>=75)": [V[(o, 0)]["luas_theta_75-90"] + V[(o, 0)]["tak_terlihat"] for o in objs],
        "evidensi_0": [V[(o, 0)]["evidensi"] for o in objs],
        "rasio_aspek (pengganggu)": [float(F[o]["rasio_aspek"]) for o in objs],
    }
    log.append(f"noise floor CD (rerata eval/noise_floor.csv) = {floor:.5f}; objek dengan |dCD|>floor = {int(ukur.sum())}/30")
    log.append("Uji lintas objek, outcome dCD = CD(0) - CD(20), Spearman. 6 kandidat dicoba -> Bonferroni 0.05/6 = 0.0083")
    for k, x in kandidat.items():
        x = np.array(x)
        a = stats.spearmanr(x, dCD); b = stats.spearmanr(x[ukur], dCD[ukur]); c = stats.spearmanr(x[keep], dCD[keep])
        log.append(f"  {k:45s} n30 rho={a[0]:+.3f} p={a[1]:.4f} | terukur n{int(ukur.sum())} rho={b[0]:+.3f} p={b[1]:.4f} "
                   f"| tanpa 4 pendorong n26 rho={c[0]:+.3f} p={c[1]:.4f}")
    E2 = np.array([[V[(o, e)]["evidensi"] for e in ELEV] for o in objs])
    Cm = -np.array([[C[e][o] for e in ELEV] for o in objs])
    rho = [stats.spearmanr(E2[i], Cm[i])[0] for i in range(30)]
    rng = np.random.default_rng(2)
    sims = [np.nanmedian([stats.spearmanr(E2[p], Cm[i])[0] for i, p in enumerate(rng.permutation(30))]) for _ in range(5000)]
    log.append(f"[pembanding C-b] rho per objek (evidensi vs -CD): median={np.nanmedian(rho):.3f}, "
               f"permutasi p={np.mean(np.array(sims) >= np.nanmedian(rho)):.4f}")
    for e in ELEV:
        log.append(f"  median lintas objek elev {e:2d}: terlihat={np.median([V[(o, e)]['terlihat'] for o in objs]):.4f} "
                   f"evidensi={np.median([V[(o, e)]['evidensi'] for o in objs]):.4f} "
                   f"CD rerata={np.mean([C[e][o] for o in objs]):.4f}")
    for nama_k, sel in [("pendorong", PENDORONG), ("lainnya", [o for o in objs if o not in PENDORONG])]:
        for e in (0, 20):
            log.append(f"  {nama_k:9s} elev {e:2d}: tak terlihat={np.mean([V[(o, e)]['tak_terlihat'] for o in sel])*100:.1f}% "
                       f"| theta 60-90={np.mean([V[(o, e)]['luas_theta_60-75'] + V[(o, e)]['luas_theta_75-90'] for o in sel])*100:.1f}% "
                       f"| theta 0-60={np.mean([sum(V[(o, e)][f'luas_theta_{l}'] for l in hv.BIN_LABEL[:4]) for o in sel])*100:.1f}%")


def uji_tiga_seed(V, C, objs, floor, log):
    """TAMBAHAN SETELAH DATA (17 Sep 2026): uji lintas objek diulang dengan dCD rerata 3 seed."""
    d3 = dcd_tiga_seed(C, objs)
    if d3 is None:
        log.append("[3seed] hasil_seed_n30.csv belum ada/lengkap -- dilewati")
        return
    F = {r["objek"]: r for r in csv.DictReader(open(os.path.join(EVAL, "fitur_objek.csv"), encoding="utf-8"))}
    dV = np.array([V[(o, 20)]["terlihat"] - V[(o, 0)]["terlihat"] for o in objs])
    keep = np.array([o not in PENDORONG for o in objs])
    ukur = np.abs(d3) > floor
    log.append("[3seed] TAMBAHAN SETELAH DATA -- dCD = rerata 3 seed (42/123/777)")
    for nama_k, x in (("delta_terlihat_0_20", dV),
                      ("luas_atas_saja", np.array([float(F[o]["luas_atas_saja"]) for o in objs]))):
        a = stats.spearmanr(x, d3); b = stats.spearmanr(x[keep], d3[keep]); c = stats.spearmanr(x[ukur], d3[ukur])
        log.append(f"  {nama_k:22s} n30 rho={a[0]:+.3f} p={a[1]:.4f} | tanpa 4 pendorong rho={b[0]:+.3f} p={b[1]:.4f} "
                   f"| terukur n{int(ukur.sum())} rho={c[0]:+.3f} p={c[1]:.4f}")
    urut = [objs[i] for i in np.argsort(-d3)]
    for o in ("obj24", "obj15", "obj22", "obj13", "obj03", "obj14"):
        i = objs.index(o)
        log.append(f"  {o}: dV={dV[i]*100:.1f} pp (peringkat {int(np.sum(dV > dV[i])) + 1}), dCD3={d3[i]:+.4f} (peringkat {urut.index(o) + 1})")


def main():
    os.makedirs(FIGUR, exist_ok=True)
    V, C, nama = baca_visibilitas(), baca_chamfer(), nama_komponen()
    objs = sorted(C[0])
    assert len(objs) == 30 and all(len(C[e]) == 30 for e in ELEV)
    floor = noise_floor()
    log = [f"CD rerata elev0 = {np.mean([C[0][o] for o in objs]):.4f} (manuskrip Table 3: 0.0390)",
           f"CD rerata elev20 = {np.mean([C[20][o] for o in objs]):.4f} (manuskrip Table 3: 0.0264)"]
    semua_uji(V, C, objs, floor, log)
    uji_tiga_seed(V, C, objs, floor, log)
    print(figur_B(V, objs))
    print(figur_C(V, C, objs, nama, floor, log))
    print(figur_A(V, nama))
    with open(os.path.join(HERE, "ringkasan_uji_visibilitas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
