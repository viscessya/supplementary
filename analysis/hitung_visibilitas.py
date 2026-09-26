# -*- coding: utf-8 -*-
"""Surface visibility of each reference mesh under the four-view rig, per elevation.

Computed from the reference meshes alone (no generated mesh is used):
  w_v        direction from the object to camera v (same convention as render/render_ortho_views.py)
  visible_v  the point faces camera v (n . w_v > 0) and is not occluded (orthographic depth buffer)
  cos_best   max of n . w_v over the cameras that see the point
  theta      arccos(cos_best) in degrees: 0 = face-on, 90 = grazing; points no camera sees are not visible
  evidensi   area-weighted mean of cos_best (not visible = 0): the cosine-weighted variant
Sampling, grid, depth tolerance, and seed are the same as in analysis/analisis_fitur_objek.py, and the
script stops if its visible shares disagree with eval/fitur_objek.csv.

Reads dataset/ and eval/fitur_objek.csv; writes eval/visibilitas_per_objek.csv.
Usage: python eval/hitung_visibilitas.py   (then python eval/plot_peta_histogram_visibilitas.py)
"""
import os
import csv
import glob
import math
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "dataset")
FITUR_LAMA = os.path.join(ROOT, "eval", "fitur_objek.csv")
OUT_CSV = os.path.join(HERE, "visibilitas_per_objek.csv")   # HERE = eval/

N_SAMPLE = 200_000
GRID = 384
DEPTH_TOL = 0.01
SEED = 42
AZIMUT = [0, 90, 180, 270]
ELEVASI = [0, 10, 20, 30, 40]

BIN_TEPI = [0, 15, 30, 45, 60, 75, 90.0001]
BIN_LABEL = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]


def arah_kamera(azim_deg, elev_deg):
    a = math.radians(azim_deg)
    e = math.radians(elev_deg)
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])


def basis_ortonormal(w):
    bantu = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(w, bantu)) > 0.9:
        bantu = np.array([1.0, 0.0, 0.0])
    u = np.cross(bantu, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v


def terlihat_satu_view(titik, normal, w, grid=GRID, tol=DEPTH_TOL):
    u, v = basis_ortonormal(w)
    x = titik @ u
    y = titik @ v
    d = -(titik @ w)
    xi = np.clip(((x + 0.75) / 1.5 * grid).astype(np.int32), 0, grid - 1)
    yi = np.clip(((y + 0.75) / 1.5 * grid).astype(np.int32), 0, grid - 1)
    idx = yi * grid + xi
    buf = np.full(grid * grid, np.inf)
    np.minimum.at(buf, idx, d)
    return ((normal @ w) > 0.0) & (d <= buf[idx] + tol)


def muat_ternormalisasi(path):
    mesh = trimesh.load(path, process=True, force="mesh")
    mesh.merge_vertices()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    m.apply_scale(1.0 / max(m.extents))
    return m


def sampel(m, n=N_SAMPLE, seed=SEED):
    titik, id_face = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(titik, float), np.asarray(m.face_normals[id_face], float)


def theta_rig(titik, normal, elev_deg, azimut=AZIMUT, grid=GRID, tol=DEPTH_TOL):
    """theta per titik (derajat); NaN = tak terlihat dari view mana pun."""
    cos_best = np.full(len(titik), -np.inf)
    for az in azimut:
        w = arah_kamera(az, elev_deg)
        vis = terlihat_satu_view(titik, normal, w, grid, tol)
        c = normal @ w
        cos_best = np.where(vis & (c > cos_best), c, cos_best)
    th = np.full(len(titik), np.nan)
    ok = np.isfinite(cos_best)
    th[ok] = np.degrees(np.arccos(np.clip(cos_best[ok], 0.0, 1.0)))
    return th


def ringkas(th):
    lihat = ~np.isnan(th)
    cosb = np.where(lihat, np.cos(np.radians(np.nan_to_num(th))), 0.0)
    hist, _ = np.histogram(th[lihat], bins=BIN_TEPI)
    r = {
        "terlihat": float(lihat.mean()),
        "evidensi": float(cosb.mean()),
        "tak_terlihat": float((~lihat).mean()),
        "theta_median_terlihat": float(np.median(th[lihat])) if lihat.any() else float("nan"),
    }
    for lab, h in zip(BIN_LABEL, hist):
        r[f"luas_theta_{lab}"] = float(h / len(th))
    return r


def main():
    berkas = sorted(glob.glob(os.path.join(DATASET, "obj*.stl")))
    baris = []
    for p in berkas:
        obj = os.path.basename(p).split("_")[0]
        print(f"  {obj} ...", flush=True)
        m = muat_ternormalisasi(p)
        titik, normal = sampel(m)
        for e in ELEVASI:
            r = {"objek": obj, "elevasi": e}
            r.update(ringkas(theta_rig(titik, normal, e)))
            baris.append(r)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        for b in baris:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in b.items()})
    print(f"Tersimpan: {OUT_CSV}")
    cek_konsistensi(baris)


def cek_konsistensi(baris):
    """Kolom `terlihat` harus sama dengan eval/fitur_objek.csv (dibulatkan 4 digit)."""
    if not os.path.exists(FITUR_LAMA):
        print("fitur_objek.csv tidak ada; cek konsistensi dilewati")
        return
    lama = {r["objek"]: r for r in csv.DictReader(open(FITUR_LAMA, encoding="utf-8"))}
    beda = []
    for b in baris:
        ref = lama.get(b["objek"], {}).get(f"terlihat_elev{b['elevasi']}")
        if ref is None:
            continue
        if abs(round(b["terlihat"], 4) - float(ref)) > 1e-4:
            beda.append((b["objek"], b["elevasi"], round(b["terlihat"], 4), float(ref)))
    print(f"Cek konsistensi vs fitur_objek.csv: {len(baris) - len(beda)}/{len(baris)} identik")
    for x in beda[:10]:
        print("   BEDA", x)


if __name__ == "__main__":
    main()
