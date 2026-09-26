#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantitative evaluation of one reconstructed mesh against its reference CAD mesh.

Metrics: Chamfer distance, F-score at tau (with precision and recall), volumetric IoU, normal consistency,
and mesh-quality indicators (watertightness, boundary and non-manifold edges, connected components,
triangle and vertex counts).
Alignment: both meshes are centred and scaled to unit bounding-box diagonal, then the prediction is aligned
by multi-start ICP before any metric is computed. Distances are in these normalized units, so tau = 0.01
means 1% of the object size.

Usage:
    python evaluate_mesh.py --gt dataset/obj01_dr500.stl --pred results/<arm>/obj01.glb --model <arm> --csv results.csv
One row of metrics is appended to the CSV (created with a header if missing).
Dependencies: trimesh, scipy, numpy, rtree (versions in requirements-pinned.txt).
"""
import argparse, csv, os, sys, datetime
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh(path):
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        sys.exit(f"Gagal memuat mesh: {path}")
    return m


def normalize(m):
    """Pusatkan di origin, skala diagonal bbox = 1. Mengembalikan mesh baru."""
    m = m.copy()
    m.apply_translation(-m.bounding_box.centroid)
    diag = np.linalg.norm(m.bounding_box.extents)
    m.apply_scale(1.0 / diag)
    return m


def sample_surface_det(m, n, rng):
    """Sampling permukaan DETERMINISTIK (area-weighted + barycentric).

    Sengaja TIDAK memakai trimesh.sample.sample_surface: di sebagian versi
    trimesh sampler itu memakai RNG internal (np.random.default_rng tanpa seed)
    yang TIDAK terpengaruh np.random.seed → sumber non-determinisme yang
    diaudit [2026-07-22]. Sampler ini hanya bergantung pada `rng` yang kita
    kontrol, jadi hasil identik antar-run untuk mesh + seed yang sama.
    Return: (points [n,3], face_normals [n,3]).
    """
    areas = m.area_faces
    total = areas.sum()
    if total <= 0:
        raise ValueError("Luas permukaan nol — mesh tidak valid untuk sampling.")
    fidx = rng.choice(len(areas), size=n, p=areas / total)
    tris = m.triangles[fidx]                    # (n,3,3)
    u = rng.random((n, 1)); v = rng.random((n, 1))
    over = (u + v) > 1.0                         # lipat ke dalam segitiga
    u[over] = 1.0 - u[over]; v[over] = 1.0 - v[over]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    pts = a + u * (b - a) + v * (c - a)
    return pts, m.face_normals[fidx]


def _icp_once(src0, tgt, tree, R0, max_iter):
    """Satu jalur ICP point-to-point dari rotasi awal R0. Return (T_total, residual)."""
    src_h = (R0 @ src0.T).T
    T_total = np.eye(4); T_total[:3, :3] = R0
    prev_err = np.inf
    for _ in range(max_iter):
        d, idx = tree.query(src_h, k=1)
        err = d.mean()
        if abs(prev_err - err) < 1e-7:
            break
        prev_err = err
        matched = tgt[idx]
        cs, cm = src_h.mean(0), matched.mean(0)   # Kabsch
        H = (src_h - cs).T @ (matched - cm)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cm - R @ cs
        src_h = (R @ src_h.T).T + t
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        T_total = T @ T_total
    return T_total, prev_err


def icp_align(pred, gt, samples=30000, max_iter=50, seed=42, n_starts=5):
    """ICP point-to-point DETERMINISTIK + MULTI-START.

    Multi-start (identitas + beberapa rotasi awal acak-terseed) lalu pilih
    residual terendah — mengatasi konvergensi ke minimum lokal berbeda pada
    objek nyaris-simetris rotasi (obj02, obj05) yang jadi sumber ketidakstabilan
    di audit [2026-07-22]. Return: (pred ter-transform, residual, transform 4x4).
    """
    rng = np.random.default_rng(seed)
    src0, _ = sample_surface_det(pred, samples, rng)
    tgt, _ = sample_surface_det(gt, samples, rng)
    tree = cKDTree(tgt)

    inits = [np.eye(3)]                            # start 1: identitas
    for _ in range(max(0, n_starts - 1)):          # start lain: rotasi acak terseed
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        inits.append(Q)

    best_T, best_err = None, np.inf
    for R0 in inits:
        T, err = _icp_once(src0, tgt, tree, R0, max_iter)
        if err < best_err:
            best_T, best_err = T, err

    out = pred.copy()
    out.apply_transform(best_T)
    return out, best_err, best_T


def sample_with_normals(m, n, rng):
    return sample_surface_det(m, n, rng)


def chamfer_fscore_nc(pred, gt, rng, n=100000, taus=(0.01, 0.02)):
    pp, pn = sample_with_normals(pred, n, rng)
    gp, gn = sample_with_normals(gt, n, rng)
    t_g, t_p = cKDTree(gp), cKDTree(pp)
    d_pg, i_pg = t_g.query(pp, k=1)   # pred -> gt
    d_gp, i_gp = t_p.query(gp, k=1)   # gt -> pred
    cd = d_pg.mean() + d_gp.mean()
    res = {"chamfer": cd}
    for tau in taus:
        prec = (d_pg < tau).mean()
        rec = (d_gp < tau).mean()
        f = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        res[f"fscore@{tau}"] = f
        res[f"precision@{tau}"] = prec
        res[f"recall@{tau}"] = rec
    nc1 = np.abs((pn * gn[i_pg]).sum(1)).mean()
    nc2 = np.abs((gn * pn[i_gp]).sum(1)).mean()
    res["normal_consistency"] = (nc1 + nc2) / 2
    return res


def volume_iou(pred, gt, pitch=1.0 / 128):
    try:
        vg = gt.voxelized(pitch).fill()
        vp = pred.voxelized(pitch).fill()
        sg = set(map(tuple, np.round(vg.points / pitch).astype(int)))
        sp = set(map(tuple, np.round(vp.points / pitch).astype(int)))
        inter = len(sg & sp); union = len(sg | sp)
        return inter / union if union else float("nan")
    except Exception as e:
        print(f"  [warn] IoU dilewati: {e}")
        return float("nan")


def mesh_quality(m):
    E = np.sort(m.edges, axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = len(m.vertices)
    adj = coo_matrix((np.ones(len(E), dtype=np.int8), (E[:, 0], E[:, 1])), shape=(nv, nv))
    n_comp = connected_components(adj, directed=False)[0]
    return {
        "triangles": len(m.faces),
        "vertices": len(m.vertices),
        "watertight": m.is_watertight,
        "boundary_edges": int((cnt == 1).sum()),
        "nonmanifold_edges": int((cnt > 2).sum()),
        "components": int(n_comp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--model", default="", help="label model (hunyuan_mv/triposr/trellis)")
    ap.add_argument("--csv", default="hasil_evaluasi.csv")
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--voxel", type=int, default=128, help="resolusi grid IoU")
    ap.add_argument("--no-icp", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed determinisme sampling+ICP (audit 2026-07-22)")
    ap.add_argument("--icp-starts", type=int, default=5,
                    help="jumlah start ICP (multi-start pilih residual terendah)")
    args = ap.parse_args()

    gt = normalize(load_mesh(args.gt))
    pred = normalize(load_mesh(args.pred))

    if not args.no_icp:
        pred, icp_err, _ = icp_align(pred, gt, seed=args.seed, n_starts=args.icp_starts)
        print(f"ICP selesai (multi-start={args.icp_starts}), residual terendah: {icp_err:.5f}")

    rng_metric = np.random.default_rng(args.seed + 12345)
    m = chamfer_fscore_nc(pred, gt, rng_metric, n=args.samples, taus=tuple(args.taus))
    m["volume_iou"] = volume_iou(pred, gt, pitch=1.0 / args.voxel)
    q = {f"pred_{k}": v for k, v in mesh_quality(pred).items()}

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "gt": os.path.basename(args.gt),
        "pred": os.path.basename(args.pred),
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in {**m, **q}.items()},
    }

    print("\n=== HASIL ===")
    for k, v in row.items():
        print(f"{k:>22}: {v}")

    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nBaris ditambahkan ke {args.csv}")


if __name__ == "__main__":
    main()
