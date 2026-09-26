#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uji_sintetis_surfnet_vs_mc.py -- synthetic pre-test, 23 Sep 2026, no GPU needed.

WHY
    Before spending GPU time, check whether ComfyUI's own surface-net extractor
    produces non-manifold edges on clean shapes. Analytic distance fields are
    extracted with ComfyUI's `voxel_to_mesh_surfnet` and with scikit-image
    marching cubes at the same threshold (0.6) on a 96^3 grid. Topology is counted
    the same way as evaluation/evaluate_mesh.py (trimesh, vertices merged).

WHERE THE SURFACE-NET CODE COMES FROM
    ComfyUI is GPL-3.0 and is NOT redistributed in this MIT-licensed repository.
    The function is read at run time from either
      --comfyui-file <path>/ComfyUI/comfy_extras/nodes_hunyuan3d.py   (your install), or
      the pinned upstream file (default), fetched from
      https://raw.githubusercontent.com/comfyanonymous/ComfyUI/b5cc8830279eae909a59de030af1e50761c36751/comfy_extras/nodes_hunyuan3d.py
    The function body is executed unchanged; only ComfyUI's progress bar is stubbed.
    The generation machine's own copy of this file was compared function by
    function on 23 Sep 2026: `voxel_to_mesh_surfnet` is identical to the pinned one.

RESULT, 23 Sep 2026 (torch 2.14, scikit-image 0.26, trimesh 5.1)
    sphere                  surface net: 0 NM, watertight      | marching cubes: 0 NM, watertight
    rotated box             surface net: 1,058 NM, not         | marching cubes: 0 NM, watertight
    plate with hole + noise surface net: 8,849 NM, not         | marching cubes: 0 NM, watertight
    (Plate WITHOUT noise: marching cubes shows 312 NM only because trimesh merges
     vertices lying exactly on grid points; without merging it has 0. Decoded
     Hunyuan fields never sit exactly on grid points, so the noisy version is the
     relevant one.)

NEEDS: torch, numpy, scikit-image, trimesh, scipy
    python uji_sintetis_surfnet_vs_mc.py [--comfyui-file PATH]
"""
import argparse
import ast
import urllib.request

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes

URL = ("https://raw.githubusercontent.com/comfyanonymous/ComfyUI/"
       "b5cc8830279eae909a59de030af1e50761c36751/comfy_extras/nodes_hunyuan3d.py")


def muat_surfnet(path=None):
    src = open(path, encoding="utf-8").read() if path else urllib.request.urlopen(URL, timeout=60).read().decode()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "voxel_to_mesh_surfnet")
    code = "\n".join(src.split("\n")[fn.lineno - 1:fn.end_lineno])

    class _PB:
        def __init__(self, n):
            pass

        def update(self, k):
            pass

    class _Utils:
        ProgressBar = _PB

    class _Comfy:
        utils = _Utils

    ns = {"torch": torch, "comfy": _Comfy}
    exec(code, ns)
    return ns["voxel_to_mesh_surfnet"]


def topologi(v, f):
    m = trimesh.Trimesh(v, f, process=True)
    E = np.sort(m.edges, axis=1)
    _, c = np.unique(E, axis=0, return_counts=True)
    return f"NM {int((c > 2).sum()):6d} | boundary {int((c == 1).sum()):4d} | watertight {m.is_watertight}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui-file", default=None)
    a = ap.parse_args()
    surfnet = muat_surfnet(a.comfyui_file)

    N, AMBANG = 96, 0.6
    g = np.linspace(-1, 1, N)
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    h = 2 / (N - 1)

    def sdf_balok(X, Y, Z, px, py, pz):
        q = np.stack([np.abs(X) - px, np.abs(Y) - py, np.abs(Z) - pz])
        return np.linalg.norm(np.maximum(q, 0), axis=0) + np.minimum(q.max(0), 0)

    a1, b1 = np.radians(30), np.radians(20)
    Xr = X * np.cos(a1) - Y * np.sin(a1)
    Yr = X * np.sin(a1) + Y * np.cos(a1)
    Yr2, Zr = Yr * np.cos(b1) - Z * np.sin(b1), Yr * np.sin(b1) + Z * np.cos(b1)
    bentuk = {
        "sphere": np.sqrt(X**2 + Y**2 + Z**2) - 0.6,
        "rotated box": sdf_balok(Xr, Yr2, Zr, 0.5, 0.3, 0.4),
        "plate with hole + noise": np.maximum(sdf_balok(X, Y, Z, 0.6, 0.3, 0.5), -(np.sqrt(X**2 + Z**2) - 0.15)),
    }
    rng = np.random.default_rng(0)
    for nama, sdf in bentuk.items():
        vox = (AMBANG - sdf / h * 0.1).astype(np.float32)   # > threshold = inside
        if "noise" in nama:
            vox = vox + rng.normal(0, 1e-4, vox.shape).astype(np.float32)
        sv, sf = surfnet(torch.from_numpy(vox), threshold=AMBANG)
        mv, mf, _, _ = marching_cubes(vox, level=AMBANG)
        print(f"{nama:24s} SURFACE NET    {topologi(sv.numpy(), sf.numpy())}")
        print(f"{'':24s} MARCHING CUBES {topologi(mv, mf)}")


if __name__ == "__main__":
    main()
