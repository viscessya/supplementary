# -*- coding: utf-8 -*-
"""Extractor comparison, generation side. Regenerates the 60 seed-42 reconstructions of the primary
comparison and, within one prompt each, extracts the same decoded voxel grid twice:
    VoxelToMesh (surface net, threshold 0.6)       -> GLB   (must match the original mesh)
    SimpanMeshMarchingCubes (threshold 0.6)        -> PLY   (custom node in comfy_node_mc/)
The custom node must be installed (checked via /object_info); the frozen workflow is verified as in
run_hunyuan_comfy_n30.py. Output goes to results/uji_mc_23sep/ only.
The written plan is dated 23 Sep 2026.

Usage (ComfyUI running, node installed):
    python uji_marching_cubes.py --cek                 check only
    python uji_marching_cubes.py --objek obj04,obj13   gate run on two components
    python uji_marching_cubes.py                       all 60 runs
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hunyuan_comfy_n30 import (BASE, COMFY, LENGAN, _get, jalankan,  # noqa: E402
                                   muat_dan_verifikasi, node_loadimage, unggah)

LENGAN_UJI = ["hunyuan_m130_elev0", "hunyuan_m130_elev20"]   # seed 42, workflow beku
AMBANG = 0.6
KELUAR = os.path.join(BASE, "results", "uji_mc_23sep")
ID_NODE_MC = "90001"


def unduh(item, tujuan):
    q = urllib.parse.urlencode({"filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    with urllib.request.urlopen(COMFY + "/view?" + q, timeout=300) as r:
        data = r.read()
    os.makedirs(os.path.dirname(tujuan), exist_ok=True)
    with open(tujuan, "wb") as f:
        f.write(data)
    return len(data)


def siapkan_workflow(wf, lengan, oid):
    kerja = json.loads(json.dumps(wf))
    vae = [nid for nid, n in kerja.items() if n.get("class_type") == "VAEDecodeHunyuan3D"]
    vtm = [nid for nid, n in kerja.items() if n.get("class_type") == "VoxelToMesh"]
    glb = [nid for nid, n in kerja.items() if n.get("class_type") == "SaveGLB"]
    if len(vae) != 1 or len(vtm) != 1 or len(glb) != 1:
        sys.exit(f"BERHENTI: node tidak unik (VAE {vae}, VoxelToMesh {vtm}, SaveGLB {glb})")
    if kerja[vtm[0]]["inputs"]["voxel"][0] != vae[0]:
        sys.exit("BERHENTI: VoxelToMesh tidak tersambung langsung ke VAEDecodeHunyuan3D")
    kerja[glb[0]]["inputs"]["filename_prefix"] = f"uji_mc_23sep/{lengan}__{oid}_sn"
    kerja[ID_NODE_MC] = {
        "class_type": "SimpanMeshMarchingCubes",
        "inputs": {"voxel": [vae[0], 0], "threshold": AMBANG,
                   "filename_prefix": f"uji_mc_23sep/{lengan}__{oid}_mc"},
    }
    return kerja


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objek", type=str, help="subset, pisah koma (gerbang G1: obj04,obj13)")
    ap.add_argument("--cek", action="store_true")
    a = ap.parse_args()

    print("=" * 72)
    try:
        info = _get("/object_info")
    except Exception as e:
        sys.exit(f"BERHENTI: ComfyUI tidak merespons di {COMFY}: {e}")
    if "SimpanMeshMarchingCubes" not in info:
        sys.exit("BERHENTI: node SimpanMeshMarchingCubes belum terpasang. Salin folder "
                 "pc_scripts/comfy_node_mc ke ComfyUI/custom_nodes lalu restart ComfyUI.")
    print("  OK  node SimpanMeshMarchingCubes terpasang")
    wf = muat_dan_verifikasi("api_hunyuan_mv.json")
    if abs(float([n for n in wf.values() if n.get("class_type") == "VoxelToMesh"][0]
                 ["inputs"]["threshold"]) - AMBANG) > 1e-9:
        sys.exit("BERHENTI: ambang VoxelToMesh bukan 0.6")

    rdir = os.path.join(BASE, "renders_m130_elev0")
    folder_obj = sorted(d for d in os.listdir(rdir) if d.startswith("obj"))
    if a.objek:
        target = {o.strip() for o in a.objek.split(",")}
        folder_obj = [f for f in folder_obj if f.split("_")[0] in target]
    print(f"  objek: {len(folder_obj)} | lengan: {', '.join(LENGAN_UJI)} | "
          f"run: {len(folder_obj) * len(LENGAN_UJI)}")
    print("=" * 72)
    if a.cek:
        print("--cek: semua pemeriksaan lolos. Tidak ada yang dijalankan.")
        return

    log_p = os.path.join(KELUAR, "runlog.csv")
    os.makedirs(KELUAR, exist_ok=True)
    baru = not os.path.exists(log_p)
    log = open(log_p, "a", newline="", encoding="utf-8")
    w = csv.writer(log)
    if baru:
        w.writerow(["waktu", "lengan", "objek", "prompt_id", "detik", "sn_byte", "mc_byte", "catatan"])

    gagal = 0
    for lengan in LENGAN_UJI:
        cfg = LENGAN[lengan]
        li = node_loadimage(wf)
        urut = {}
        for nid, n in li.items():
            asal = str(n["inputs"].get("image", "")).lower()
            for v in cfg["views"]:
                if asal.startswith(v):
                    urut[v] = nid
        if len(urut) != 4:
            sys.exit(f"BERHENTI: pemetaan view gagal: {urut}")
        print(f"\n■ {lengan}")
        for i, folder in enumerate(folder_obj, 1):
            oid = folder.split("_")[0]
            t_sn = os.path.join(KELUAR, lengan, f"{oid}_sn.glb")
            t_mc = os.path.join(KELUAR, lengan, f"{oid}_mc.ply")
            if os.path.exists(t_sn) and os.path.exists(t_mc):
                print(f"  [{i:>2}/{len(folder_obj)}] {oid} sudah ada, dilewati")
                continue
            try:
                peta = {}
                for v in cfg["views"]:
                    src = os.path.join(BASE, cfg["render"], folder, f"{v}.png")
                    peta[urut[v]] = unggah(src, f"ujimc__{lengan}__{oid}__{v}.png")
                kerja = siapkan_workflow(wf, lengan, oid)
                pid, dtk, hasil = jalankan(kerja, peta, timeout=1200)
                out = hasil.get("outputs", {})
                mc_items = (out.get(ID_NODE_MC) or {}).get("mc_mesh") or []
                sn_items = [it for nid, o in out.items() if nid != ID_NODE_MC
                            for k in ("mesh", "result", "gltf", "3d") for it in (o.get(k) or [])
                            if isinstance(it, dict) and it.get("filename")]
                if len(mc_items) != 1 or len(sn_items) != 1:
                    raise RuntimeError(f"keluaran tidak lengkap: mc {len(mc_items)}, sn {len(sn_items)}")
                b_sn = unduh(sn_items[0], t_sn)
                b_mc = unduh(mc_items[0], t_mc)
                js = dict(mc_items[0]); js["filename"] = js["filename"][:-4] + ".json"
                unduh(js, t_mc[:-4] + ".json")
                cat = "CACHE-HIT? <3 detik" if dtk < 3 else ""
                print(f"  [{i:>2}/{len(folder_obj)}] {oid} OK {dtk:6.1f}s  "
                      f"sn {b_sn/1e6:5.1f} MB  mc {b_mc/1e6:5.1f} MB {cat}")
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lengan, oid, pid,
                            f"{dtk:.1f}", b_sn, b_mc, cat])
            except Exception as e:
                gagal += 1
                print(f"  [{i:>2}/{len(folder_obj)}] {oid} GAGAL: {e}")
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lengan, oid, "", "", 0, 0,
                            f"GAGAL: {e}"])
            log.flush()
    log.close()
    print("\n" + "=" * 72)
    print(f"gagal: {gagal} | log: {log_p}")
    print("Berikutnya: sinkronkan results/uji_mc_23sep ke Mac, lalu jalankan "
          "eval/evaluasi_uji_mc.py di Mac.")
    if gagal:
        sys.exit(1)


if __name__ == "__main__":
    main()
