# -*- coding: utf-8 -*-
"""Runs Hunyuan3D-2mv generation through the ComfyUI API for every arm, component, and elevation.

Every run is logged (component, elevation, arm, prompt_id, duration, input file names). Before any run the
exported API workflow is checked against the frozen generation parameters, and the script stops if any
differs. Views are paired by file name (front, left, back, right), not by upload order. A run that returns
in about 0 s is a ComfyUI cache hit; it is flagged and not counted. Existing meshes in results/<arm>/ are
never overwritten, so an interrupted batch can be resumed.

One-time step: export each workflow from ComfyUI (Workflow > Export (API)) as workflow/api_hunyuan_mv.json
and workflow/api_hunyuan_sv.json, or convert them with konversi_workflow_ke_api.py.

Usage (ComfyUI running):
    python run_hunyuan_comfy_n30.py --cek                      check only
    python run_hunyuan_comfy_n30.py --lengan ARM --batas 1     smoke test on one component
    python run_hunyuan_comfy_n30.py                            all runs
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

COMFY = "http://127.0.0.1:8188"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_DIHARAP = 30

BEKU = {
    "ImageOnlyCheckpointLoader": {"ckpt_name": "hunyuan3d-dit-v2-mv_fp16.safetensors"},
    "KSampler": {"seed": 42, "steps": 30, "cfg": 5.0,
                 "sampler_name": "euler", "scheduler": "normal", "denoise": 1},
    "EmptyLatentHunyuan3Dv2": {"resolution": 4096},
    "VAEDecodeHunyuan3D": {"num_chunks": 8000, "octree_resolution": 448},
    "VoxelToMesh": {"algorithm": "surface net", "threshold": 0.6},
    "ModelSamplingAuraFlow": {"shift": 1},
}

VIEWS = ["front", "left", "back", "right"]

LENGAN = {
    "hunyuan_m130_elev0":  dict(api="api_hunyuan_mv.json", render="renders_m130_elev0",  views=VIEWS),
    "hunyuan_m130_elev20": dict(api="api_hunyuan_mv.json", render="renders_m130_elev20", views=VIEWS),
    "hunyuan_sv_m130":     dict(api="api_hunyuan_sv.json", render="renders_m130_elev0",  views=["front"]),
    "hunyuan_sv_withmethod": dict(api="api_hunyuan_sv.json",
                                  render="hunyuan_inputs_withmethod_m130_elev0",
                                  views=["front"]),
    "hunyuan_mv_withmethod_elev20": dict(api="api_hunyuan_mv.json",
                                  render="hunyuan_inputs_withmethod_m130_elev20",
                                  views=VIEWS),
    "hunyuan_material_v1_elev20": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev20_material_v1",
                                  views=VIEWS),
    "hunyuan_persp_v4_elev20": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev20_persp_v4",
                                  views=VIEWS),
    "hunyuan_diagonal_v1_elev20": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev20_diagonal_v1",
                                  views=VIEWS),
    "hunyuan_backdrop_v1_elev20": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev20_backdrop_v1",
                                  views=VIEWS),

    "hunyuan_m130_elev10": dict(api="api_hunyuan_mv.json", render="renders_m130_elev10", views=VIEWS),
    "hunyuan_m130_elev30": dict(api="api_hunyuan_mv.json", render="renders_m130_elev30", views=VIEWS),
    "hunyuan_m130_elev40": dict(api="api_hunyuan_mv.json", render="renders_m130_elev40", views=VIEWS),

    "hunyuan_mv_withmethod_elev0":  dict(api="api_hunyuan_mv.json",
                                  render="hunyuan_inputs_withmethod_mv_m130_elev0",
                                  views=VIEWS),
    "hunyuan_mv_withmethod_elev10": dict(api="api_hunyuan_mv.json",
                                  render="hunyuan_inputs_withmethod_m130_elev10",
                                  views=VIEWS),
    "hunyuan_mv_withmethod_elev30": dict(api="api_hunyuan_mv.json",
                                  render="hunyuan_inputs_withmethod_m130_elev30",
                                  views=VIEWS),
    "hunyuan_mv_withmethod_elev40": dict(api="api_hunyuan_mv.json",
                                  render="hunyuan_inputs_withmethod_m130_elev40",
                                  views=VIEWS),

    "hunyuan_material_v1_elev0":  dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev0_material_v1",
                                  views=VIEWS),
    "hunyuan_material_v1_elev10": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev10_material_v1",
                                  views=VIEWS),
    "hunyuan_material_v1_elev30": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev30_material_v1",
                                  views=VIEWS),
    "hunyuan_material_v1_elev40": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev40_material_v1",
                                  views=VIEWS),

    "hunyuan_persp_v4_elev0":  dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev0_persp_v4",
                                  views=VIEWS),
    "hunyuan_persp_v4_elev10": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev10_persp_v4",
                                  views=VIEWS),
    "hunyuan_persp_v4_elev30": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev30_persp_v4",
                                  views=VIEWS),
    "hunyuan_persp_v4_elev40": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev40_persp_v4",
                                  views=VIEWS),

    "hunyuan_diagonal_v1_elev0":  dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev0_diagonal_v1",
                                  views=VIEWS),
    "hunyuan_diagonal_v1_elev10": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev10_diagonal_v1",
                                  views=VIEWS),
    "hunyuan_diagonal_v1_elev30": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev30_diagonal_v1",
                                  views=VIEWS),
    "hunyuan_diagonal_v1_elev40": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev40_diagonal_v1",
                                  views=VIEWS),

    "hunyuan_backdrop_v1_elev0":  dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev0_backdrop_v1",
                                  views=VIEWS),
    "hunyuan_backdrop_v1_elev10": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev10_backdrop_v1",
                                  views=VIEWS),
    "hunyuan_backdrop_v1_elev30": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev30_backdrop_v1",
                                  views=VIEWS),
    "hunyuan_backdrop_v1_elev40": dict(api="api_hunyuan_mv.json",
                                  render="renders_m130_elev40_backdrop_v1",
                                  views=VIEWS),

    "hunyuan_m130_elev0_seed123":  dict(api="api_hunyuan_mv.json", render="renders_m130_elev0",  views=VIEWS),
    "hunyuan_m130_elev0_seed777":  dict(api="api_hunyuan_mv.json", render="renders_m130_elev0",  views=VIEWS),
    "hunyuan_m130_elev20_seed123": dict(api="api_hunyuan_mv.json", render="renders_m130_elev20", views=VIEWS),
    "hunyuan_m130_elev20_seed777": dict(api="api_hunyuan_mv.json", render="renders_m130_elev20", views=VIEWS),

    "hunyuan_m130_elev30_seed123":     dict(api="api_hunyuan_mv.json", render="renders_m130_elev30", views=VIEWS),
    "hunyuan_m130_elev40_seed123":     dict(api="api_hunyuan_mv.json", render="renders_m130_elev40", views=VIEWS),
    "hunyuan_persp_v4_elev30_seed123": dict(api="api_hunyuan_mv.json", render="renders_m130_elev30_persp_v4", views=VIEWS),
    "hunyuan_persp_v4_elev40_seed123": dict(api="api_hunyuan_mv.json", render="renders_m130_elev40_persp_v4", views=VIEWS),
    "hunyuan_backdrop_v1_elev40_seed123": dict(api="api_hunyuan_mv.json", render="renders_m130_elev40_backdrop_v1", views=VIEWS),
}

SEED_OVERRIDE = {
    "hunyuan_m130_elev0_seed123":  123,
    "hunyuan_m130_elev0_seed777":  777,
    "hunyuan_m130_elev20_seed123": 123,
    "hunyuan_m130_elev20_seed777": 777,
    "hunyuan_m130_elev30_seed123":        123,
    "hunyuan_m130_elev40_seed123":        123,
    "hunyuan_persp_v4_elev30_seed123":    123,
    "hunyuan_persp_v4_elev40_seed123":    123,
    "hunyuan_backdrop_v1_elev40_seed123": 123,
}

OBJEK_M8F = ["obj25", "obj24", "obj20", "obj21", "obj22"]


def _get(path):
    with urllib.request.urlopen(COMFY + path, timeout=30) as r:
        return json.load(r)


def _post(path, data):
    req = urllib.request.Request(COMFY + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def unggah(path_lokal, nama_tujuan):
    """Unggah gambar ke folder input ComfyUI lewat /upload/image (multipart)."""
    batas = uuid.uuid4().hex
    with open(path_lokal, "rb") as f:
        isi = f.read()
    body = (
        f"--{batas}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{nama_tujuan}\"\r\nContent-Type: image/png\r\n\r\n".encode()
        + isi
        + f"\r\n--{batas}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode()
        + f"--{batas}--\r\n".encode()
    )
    req = urllib.request.Request(COMFY + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={batas}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["name"]


def muat_dan_verifikasi(nama_api):
    p = os.path.join(BASE, "workflow", nama_api)
    if not os.path.exists(p):
        sys.exit(
            f"BERHENTI: {p} tidak ada.\n"
            f"   Ekspor dulu dari ComfyUI: Workflow → Export (API) → simpan sebagai {nama_api}\n"
            f"   (lihat penjelasan di kepala skrip ini)")
    wf = json.load(open(p, encoding="utf-8"))
    if "nodes" in wf:
        sys.exit(f"BERHENTI: {nama_api} masih format UI, bukan API.\n"
                 f"   Pakai Workflow → Export (API), bukan Save.")

    masalah = []
    ditemukan = {}
    for nid, node in wf.items():
        ct = node.get("class_type")
        if ct in BEKU:
            ditemukan.setdefault(ct, 0)
            ditemukan[ct] += 1
            for k, v_harap in BEKU[ct].items():
                v = node.get("inputs", {}).get(k)
                if v is None:
                    masalah.append(f"{ct}.{k} tidak ada di workflow")
                    continue
                if isinstance(v_harap, (int, float)) and not isinstance(v_harap, bool):
                    try:
                        cocok = abs(float(v) - float(v_harap)) < 1e-9
                    except (TypeError, ValueError):
                        masalah.append(
                            f"{ct}.{k} = {v!r} — bukan angka, seharusnya {v_harap}. "
                            f"Ini tanda nilai widget BERGESER saat konversi.")
                        continue
                    if not cocok:
                        masalah.append(f"{ct}.{k} = {v}, seharusnya {v_harap}")
                elif v != v_harap:
                    masalah.append(f"{ct}.{k} = {v!r}, seharusnya {v_harap!r}")
    for ct in BEKU:
        if ct not in ditemukan:
            masalah.append(f"node {ct} tidak ditemukan sama sekali")
    if masalah:
        print(f"\nBERHENTI — {nama_api} TIDAK COCOK dengan parameter beku:")
        for m in masalah:
            print("   ·", m)
        sys.exit("\n   Jangan jalankan. Periksa workflow yang diekspor.")
    print(f"  OK {nama_api} cocok dengan seluruh parameter beku")
    return wf


def node_loadimage(wf):
    return {nid: n for nid, n in wf.items() if n.get("class_type") == "LoadImage"}


def jalankan(wf, peta_gambar, timeout=600, seed=None):
    """peta_gambar: {node_id: nama_berkas_di_input_comfy}. Kembalikan (prompt_id, detik, hasil).

    seed: kalau diisi, override nilai seed KSampler HANYA pada salinan runtime ini --
    tidak pernah menyentuh berkas workflow di disk atau gerbang BEKU. Dipakai khusus
    untuk lengan pengecekan variabilitas-seed (M8-F, lihat SEED_OVERRIDE); lengan baku
    memanggil jalankan() tanpa argumen ini dan tetap seed=42 seperti sebelumnya.
    """
    kerja = json.loads(json.dumps(wf))
    for nid, nama in peta_gambar.items():
        kerja[nid]["inputs"]["image"] = nama
    if seed is not None:
        for node in kerja.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = seed
    cid = str(uuid.uuid4())
    t0 = time.time()
    pid = _post("/prompt", {"prompt": kerja, "client_id": cid})["prompt_id"]
    while True:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"lewat {timeout}s tanpa selesai (prompt_id {pid})")
        h = _get(f"/history/{pid}")
        if pid in h:
            return pid, time.time() - t0, h[pid]
        time.sleep(1.5)


def ambil_glb(hasil, tujuan):
    for out in hasil.get("outputs", {}).values():
        for kunci in ("mesh", "result", "gltf", "3d"):
            for item in out.get(kunci, []) or []:
                if isinstance(item, dict) and item.get("filename"):
                    q = urllib.parse.urlencode({"filename": item["filename"],
                                                "subfolder": item.get("subfolder", ""),
                                                "type": item.get("type", "output")})
                    with urllib.request.urlopen(COMFY + "/view?" + q, timeout=180) as r:
                        data = r.read()
                    os.makedirs(os.path.dirname(tujuan), exist_ok=True)
                    with open(tujuan, "wb") as f:
                        f.write(data)
                    return item["filename"], len(data)
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengan", choices=list(LENGAN), help="jalankan satu lengan saja")
    ap.add_argument("--batas", type=int, help="hanya N objek pertama (uji asap)")
    ap.add_argument("--objek", type=str,
                    help="hanya objek ini, pisah koma (mis. obj25,obj24,obj20) -- "
                         "untuk subset M8-F; beda dari --batas yang ambil N pertama")
    ap.add_argument("--m8f", action="store_true",
                    help="jalankan keempat lengan M8-F sekaligus dengan 8 objek "
                         "OBJEK_M8F otomatis -- singkatan dari 4x --lengan+--objek "
                         "manual. Mengabaikan --lengan/--objek/--batas kalau dipakai.")
    ap.add_argument("--cek", action="store_true", help="periksa saja, tidak menjalankan")
    a = ap.parse_args()

    print("=" * 78)
    print(f"ComfyUI : {COMFY}")
    try:
        _get("/system_stats")
        print("  OK ComfyUI merespons")
    except Exception as e:
        sys.exit(f"BERHENTI: ComfyUI tidak merespons di {COMFY} — {e}")

    rdir = os.path.join(BASE, "renders_m130_elev0")
    objek = sorted(d for d in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, d))
                   and d.startswith("obj"))
    if len(objek) != N_DIHARAP:
        sys.exit(f"BERHENTI: {len(objek)} objek di renders_m130_elev0, seharusnya {N_DIHARAP}")
    print(f"  OK {len(objek)} objek terbaca dari renders_m130_elev0")

    if a.m8f:
        pilih = list(SEED_OVERRIDE)
        a.objek = ",".join(OBJEK_M8F)
        a.batas = None
        print(f"  --m8f: menjalankan {len(pilih)} lengan x {len(OBJEK_M8F)} objek "
              f"({', '.join(OBJEK_M8F)})")
    else:
        pilih = [a.lengan] if a.lengan else list(LENGAN)
    wf_cache = {}
    for nama in pilih:
        cfg = LENGAN[nama]
        if cfg["api"] not in wf_cache:
            wf_cache[cfg["api"]] = muat_dan_verifikasi(cfg["api"])
        li = node_loadimage(wf_cache[cfg["api"]])
        if len(li) != len(cfg["views"]):
            sys.exit(f"BERHENTI: {nama} butuh {len(cfg['views'])} LoadImage, "
                     f"workflow punya {len(li)}")
    print("=" * 78)

    if a.cek:
        print("\n--cek: seluruh pemeriksaan lolos. Tidak ada yang dijalankan.")
        return

    log_p = os.path.join(BASE, "logs", "hunyuan_n30_runlog.csv")
    os.makedirs(os.path.dirname(log_p), exist_ok=True)
    baru = not os.path.exists(log_p)
    log = open(log_p, "a", newline="", encoding="utf-8")
    w = csv.writer(log)
    if baru:
        w.writerow(["waktu", "lengan", "objek", "folder_render", "prompt_id",
                    "detik", "berkas_masukan", "keluaran", "byte", "catatan"])

    total_ok = total_lewat = total_gagal = 0
    harap = {}  # DITAMBAHKAN 8 Sep 2026 -- jumlah objek yang DIHARAPKAN per lengan;
    for nama in pilih:
        cfg = LENGAN[nama]
        wf = wf_cache[cfg["api"]]
        li = node_loadimage(wf)
        urut = {}
        for nid, n in li.items():
            asal = str(n["inputs"].get("image", "")).lower()
            for v in cfg["views"]:
                if asal.startswith(v):
                    urut[v] = nid
        if len(urut) != len(cfg["views"]):
            sys.exit(f"BERHENTI: tidak bisa memetakan view ke node LoadImage di {nama}. "
                     f"Terbaca: {urut}")

        daftar = objek[: a.batas] if a.batas else objek
        if a.objek:
            target = {o.strip() for o in a.objek.split(",") if o.strip()}
            daftar = [f for f in daftar if f.split("_")[0] in target]
            hilang = target - {f.split("_")[0] for f in daftar}
            if hilang:
                sys.exit(f"BERHENTI: --objek menyebut objek yang tidak ada: {sorted(hilang)}")
        harap[nama] = len(daftar)
        print(f"\n■ {nama}  ({len(daftar)} objek, sumber {cfg['render']})")
        for i, folder in enumerate(daftar, 1):
            oid = folder.split("_")[0]
            tujuan = os.path.join(BASE, "results", nama, f"{oid}.glb")
            if os.path.exists(tujuan):
                print(f"  [{i:>2}/{len(daftar)}] {oid} — sudah ada, dilewati")
                total_lewat += 1
                continue
            try:
                peta, nama_unggah = {}, []
                for v in cfg["views"]:
                    src = os.path.join(BASE, cfg["render"], folder, f"{v}.png")
                    if not os.path.exists(src):
                        raise FileNotFoundError(src)
                    tgt = f"{nama}__{oid}__{v}.png"
                    peta[urut[v]] = unggah(src, tgt)
                    nama_unggah.append(tgt)
                pid, dtk, hasil = jalankan(wf, peta, seed=SEED_OVERRIDE.get(nama))
                fn, byt = ambil_glb(hasil, tujuan)
                catatan = ""
                if dtk < 3:
                    catatan = "CACHE-HIT? durasi < 3 detik — BUKAN data, periksa manual"
                if fn is None:
                    catatan = (catatan + " · ").strip(" ·") + "GLB tidak ditemukan di keluaran"
                    total_gagal += 1
                    print(f"  [{i:>2}/{len(daftar)}] {oid} {catatan}")
                else:
                    total_ok += 1
                    print(f"  [{i:>2}/{len(daftar)}] {oid} OK {dtk:6.1f}s  {byt/1e6:5.2f} MB"
                          + (f"  {catatan}" if catatan else ""))
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), nama, oid, cfg["render"],
                            pid, f"{dtk:.1f}", "|".join(nama_unggah), tujuan, byt, catatan])
                log.flush()
            except Exception as e:
                total_gagal += 1
                print(f"  [{i:>2}/{len(daftar)}] {oid} GAGAL: {e}")
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), nama, oid, cfg["render"],
                            "", "", "", "", 0, f"GAGAL: {e}"])
                log.flush()

    log.close()
    print("\n" + "=" * 78)
    print(f"selesai {total_ok} · dilewati {total_lewat} · gagal {total_gagal}")
    print(f"log: {log_p}")
    if total_gagal:
        print("ADA YANG GAGAL — periksa log sebelum melanjutkan ke evaluasi.")
        sys.exit(1)
    print("OK Semua run selesai. Periksa jumlah GLB per lengan sebelum evaluasi:")
    for nama in pilih:
        d = os.path.join(BASE, "results", nama)
        n = len([f for f in os.listdir(d) if f.endswith(".glb")]) if os.path.isdir(d) else 0
        exp = harap.get(nama, N_DIHARAP)
        print(f"   {nama:22} {n}/{exp} {'OK' if n == exp else 'KURANG'}")


if __name__ == "__main__":
    main()
