# -*- coding: utf-8 -*-
"""ComfyUI custom node "SimpanMeshMarchingCubes" for the extractor comparison.

Extracts a mesh with scikit-image marching cubes from the same decoded voxel grid that ComfyUI's
VoxelToMesh (surface net) receives, at the same threshold (0.6), and maps the vertices exactly as
voxel_to_mesh_surfnet does (+1 padding offset, centring at max(D, H, W)/2, scaling, fliplr), so that both
meshes share one coordinate frame. The grid is not modified (no padding, no smoothing).
Output: <ComfyUI output>/<filename_prefix>.ply (binary) and a small .json note.

Install: copy this folder to ComfyUI/custom_nodes/, install scikit-image into ComfyUI's Python, and restart
ComfyUI; the log then shows "[comfy_node_mc] siap" ("ready").
"""
import json
import os
import struct
import time

import numpy as np

try:
    import folder_paths  # modul ComfyUI
except ImportError:  # dipakai saat diuji di luar ComfyUI
    folder_paths = None


def tulis_ply_biner(path, verts, faces):
    verts = np.ascontiguousarray(verts, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<i4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    rec = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    rec["n"] = 3
    rec["i"] = faces
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())
        f.write(rec.tobytes())


def ekstrak_mc(arr, threshold):
    """arr: grid (D,H,W) float. Kembalikan (verts, faces) di kerangka surface-net ComfyUI."""
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(arr, level=threshold, allow_degenerate=False)
    D, H, W = arr.shape
    v_max = max(D, H, W)
    verts = verts + 1.0                      # surface net ComfyUI bekerja di koordinat grid ber-padding 1
    verts = (verts - v_max / 2.0) / (v_max / 2.0)
    verts = verts[:, ::-1]                    # torch.fliplr: (z,y,x) -> (x,y,z)
    return np.ascontiguousarray(verts), np.ascontiguousarray(faces)


def _jalur_unik(dasar):
    if not os.path.exists(dasar + ".ply"):
        return dasar
    k = 1
    while os.path.exists(f"{dasar}_{k:03d}.ply"):
        k += 1
    return f"{dasar}_{k:03d}"


class SimpanMeshMarchingCubes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "voxel": ("VOXEL",),
            "threshold": ("FLOAT", {"default": 0.6, "min": -1.0, "max": 1.0, "step": 0.01}),
            "filename_prefix": ("STRING", {"default": "uji_mc/mesh"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "simpan"
    OUTPUT_NODE = True
    CATEGORY = "uji_mc"

    def simpan(self, voxel, threshold, filename_prefix):
        out_dir = folder_paths.get_output_directory() if folder_paths else os.getcwd()
        hasil = []
        for b, x in enumerate(voxel.data):
            arr = x.detach().float().cpu().numpy()
            t0 = time.time()
            verts, faces = ekstrak_mc(arr, threshold)
            dt = time.time() - t0
            pref = filename_prefix if len(voxel.data) == 1 else f"{filename_prefix}_b{b}"
            dasar = _jalur_unik(os.path.join(out_dir, pref))
            os.makedirs(os.path.dirname(dasar), exist_ok=True)
            tulis_ply_biner(dasar + ".ply", verts, faces)
            info = {
                "grid_shape": list(arr.shape), "threshold": threshold,
                "field_min": float(arr.min()), "field_max": float(arr.max()),
                "voxel_di_atas_ambang": int((arr > threshold).sum()),
                "n_vertex": int(len(verts)), "n_face": int(len(faces)),
                "detik_mc": round(dt, 2), "skimage": __import__("skimage").__version__,
            }
            with open(dasar + ".json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=1)
            rel = os.path.relpath(dasar + ".ply", out_dir)
            hasil.append({"filename": os.path.basename(rel),
                          "subfolder": os.path.dirname(rel), "type": "output"})
        return {"ui": {"mc_mesh": hasil}}


NODE_CLASS_MAPPINGS = {"SimpanMeshMarchingCubes": SimpanMeshMarchingCubes}
NODE_DISPLAY_NAME_MAPPINGS = {"SimpanMeshMarchingCubes": "Simpan Mesh Marching Cubes (uji 23 Sep)"}
print("[comfy_node_mc] siap")
