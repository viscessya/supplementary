# -*- coding: utf-8 -*-
"""Per-object geometric features of the reference CAD meshes (results/fitur_objek.csv).

All values are recomputed from the reference meshes in dataset/: dimensions, triangle count, connected
components, boundary and non-manifold edges, Euler number and genus, aspect ratio, flatness, bounding-box
fill, area and volume relative to the convex hull, up/down/side-facing area, and the surface share visible
to the four-view rig at 0-40 degrees elevation (200,000 area-weighted samples, orthographic depth buffer).
Up axis is +Z and camera azimuths are 0/90/180/270 (front/left/back/right), as in render/render_ortho_views.py.

Usage: python eval/analisis_fitur_objek.py   (writes eval/fitur_objek.csv; see README, "Paths")
"""
import os
import csv
import math
import glob
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "dataset")
OUT_CSV = os.path.join(HERE, "fitur_objek.csv")

N_SAMPLE = 200_000      # titik sampel permukaan (berbobot luas)
GRID = 384              # resolusi depth buffer semu
DEPTH_TOL = 0.01        # 1% diagonal ternormalisasi
ELEVASI = [0, 10, 20, 30, 40]
AZIMUT = [0, 90, 180, 270]
SEED = 42


def arah_kamera(azim_deg, elev_deg):
    """Vektor satuan dari objek MENUJU kamera (Blender: azimut memutar sumbu Z)."""
    a = math.radians(azim_deg)
    e = math.radians(elev_deg)
    return np.array([
        math.sin(a) * math.cos(e),
        -math.cos(a) * math.cos(e),
        math.sin(e),
    ], dtype=float)


def basis_ortonormal(w):
    """Bangun basis (u, v, w) dengan w = arah pandang."""
    bantu = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(w, bantu)) > 0.9:
        bantu = np.array([1.0, 0.0, 0.0])
    u = np.cross(bantu, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v


def _terlihat_satu_view(titik, normal, w):
    """Mask boolean: titik mana yang terlihat dari satu arah pandang w."""
    u, v = basis_ortonormal(w)
    x = titik @ u
    y = titik @ v
    d = -(titik @ w)
    xi = np.clip(((x + 0.75) / 1.5 * GRID).astype(np.int32), 0, GRID - 1)
    yi = np.clip(((y + 0.75) / 1.5 * GRID).astype(np.int32), 0, GRID - 1)
    idx = yi * GRID + xi
    buf = np.full(GRID * GRID, np.inf)
    np.minimum.at(buf, idx, d)
    return ((normal @ w) > 0.0) & (d <= buf[idx] + DEPTH_TOL)


def fraksi_terlihat(titik, normal, elev_deg):
    """
    Fraksi luas permukaan yang terlihat dari rig 4-view pada elevasi tertentu.
    Titik disampel berbobot luas, jadi fraksi titik = fraksi luas.
    Visibilitas: titik menghadap kamera DAN tidak tertutup titik lain
    yang lebih dekat pada sel piksel yang sama (depth buffer ortografik).
    """
    terlihat = np.zeros(len(titik), dtype=bool)
    for az in AZIMUT:
        terlihat |= _terlihat_satu_view(titik, normal, arah_kamera(az, elev_deg))
    return terlihat.mean()


def analisis(path):
    nama = os.path.splitext(os.path.basename(path))[0]
    mesh = trimesh.load(path, process=True, force="mesh")
    mesh.merge_vertices()
    mesh.remove_degenerate_faces() if hasattr(mesh, "remove_degenerate_faces") else None

    dims_mm = mesh.extents.copy()
    n_tri = len(mesh.faces)
    n_komponen = mesh.body_count
    watertight = bool(mesh.is_watertight)

    _, hitung = np.unique(
        np.sort(mesh.edges, axis=1), axis=0, return_counts=True
    )
    tepi_batas = int((hitung == 1).sum())
    tepi_nonmanifold = int((hitung > 2).sum())
    euler = int(mesh.euler_number)
    genus = int(round((2 * n_komponen - euler) / 2)) if watertight else None

    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    m.apply_scale(1.0 / max(m.extents))

    hull = m.convex_hull
    rasio_luas_hull = float(m.area / hull.area)
    try:
        rasio_volume_hull = float(abs(m.volume) / abs(hull.volume))
    except Exception:
        rasio_volume_hull = float("nan")
    isi_bbox = float(abs(m.volume) / float(np.prod(m.extents)))

    dims_urut = np.sort(dims_mm)[::-1]
    rasio_aspek = float(dims_urut[0] / max(dims_urut[2], 1e-9))
    kepipihan = float(dims_urut[2] / dims_urut[0])

    rng = np.random.default_rng(SEED)
    titik, id_face = trimesh.sample.sample_surface(m, N_SAMPLE, seed=SEED)
    titik = np.asarray(titik, dtype=float)
    normal = np.asarray(m.face_normals[id_face], dtype=float)

    nz = normal[:, 2]
    hadap_atas = float((nz > 0.5).mean())
    hadap_bawah = float((nz < -0.5).mean())
    hadap_samping = float((np.abs(nz) <= 0.5).mean())

    hasil = {
        "objek": nama.split("_")[0],
        "berkas": nama,
        "dim_mm": " x ".join(f"{d:.1f}" for d in dims_mm),
        "ukuran_maks_mm": round(float(dims_mm.max()), 1),
        "segitiga_gt": n_tri,
        "komponen": n_komponen,
        "watertight": watertight,
        "tepi_batas": tepi_batas,
        "tepi_nonmanifold": tepi_nonmanifold,
        "euler": euler,
        "genus": genus,
        "rasio_aspek": round(rasio_aspek, 2),
        "kepipihan": round(kepipihan, 3),
        "isi_bbox": round(isi_bbox, 3),
        "rasio_luas_thd_hull": round(rasio_luas_hull, 3),
        "rasio_volume_thd_hull": round(rasio_volume_hull, 3),
        "luas_hadap_atas": round(hadap_atas, 4),
        "luas_hadap_bawah": round(hadap_bawah, 4),
        "luas_hadap_samping": round(hadap_samping, 4),
    }

    for e in ELEVASI:
        hasil[f"terlihat_elev{e}"] = round(float(fraksi_terlihat(titik, normal, e)), 4)
    hasil["delta_terlihat_0_20"] = round(
        hasil["terlihat_elev20"] - hasil["terlihat_elev0"], 4
    )

    sisi0 = np.zeros(len(titik), dtype=bool)
    for az in AZIMUT:
        sisi0 |= _terlihat_satu_view(titik, normal, arah_kamera(az, 0))
    atas = _terlihat_satu_view(titik, normal, np.array([0.0, 0.0, 1.0]))
    hasil["luas_terlihat_sisi0"] = round(float(sisi0.mean()), 4)
    hasil["luas_terlihat_atas"] = round(float(atas.mean()), 4)
    hasil["luas_atas_saja"] = round(float((atas & ~sisi0).mean()), 4)
    hasil["luas_tak_terlihat"] = round(float((~(sisi0 | atas)).mean()), 4)
    return hasil


def main():
    berkas = sorted(glob.glob(os.path.join(DATASET, "obj*.stl")))
    if not berkas:
        raise SystemExit(f"Tidak ada STL di {DATASET}")

    baris = []
    for p in berkas:
        print(f"  memproses {os.path.basename(p)} ...", flush=True)
        baris.append(analisis(p))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)

    kolom = ["objek", "ukuran_maks_mm", "segitiga_gt", "komponen", "genus",
             "rasio_aspek", "isi_bbox", "rasio_luas_thd_hull",
             "luas_terlihat_sisi0", "luas_atas_saja", "luas_tak_terlihat",
             "terlihat_elev20", "delta_terlihat_0_20"]
    lebar = {k: max(len(k), 8) for k in kolom}
    print()
    print(" | ".join(k.ljust(lebar[k]) for k in kolom))
    print("-|-".join("-" * lebar[k] for k in kolom))
    for b in baris:
        print(" | ".join(str(b[k]).ljust(lebar[k]) for k in kolom))
    print(f"\nTersimpan: {OUT_CSV}")


if __name__ == "__main__":
    main()
