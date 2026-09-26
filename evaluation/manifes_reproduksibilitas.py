#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collects every frozen parameter of the pipeline into one human-readable manifest and records the
package versions of the evaluation environment.

Outputs: eval/MANIFES_REPRODUKSIBILITAS.md and requirements-pinned.txt.
Chamfer distance and normal consistency were bit-identical across operating systems and trimesh versions;
volumetric IoU was not (voxelization changed between trimesh versions: 0.999163 vs 0.998983 for a
reference mesh against itself), so trimesh is pinned.

Usage: python eval/manifes_reproduksibilitas.py
"""
import datetime
import os
import platform
import shutil
import subprocess


def _versi_os():
    """Versi sistem operasi lengkap dengan nomornya.

    platform.system() saja hanya memberi 'Darwin' — tidak cukup untuk menelusuri
    pergeseran lingkungan. Di macOS, platform.mac_ver()[0] memberi mis. '15.7.4'.
    """
    if platform.system() == "Darwin":
        rilis = platform.mac_ver()[0] or "?"
        return f"macOS {rilis} (Darwin {platform.release()})"
    return f"{platform.system()} {platform.release()}"


def _versi_blender():
    """Versi Blender, kalau bisa dipanggil dari PATH.

    Render GT dibuat oleh Blender, jadi versinya bagian dari lingkungan yang
    menentukan angka — sama pentingnya dengan versi trimesh.
    """
    exe = shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"
    try:
        keluaran = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=30
        ).stdout.strip().splitlines()
        return keluaran[0].strip() if keluaran else "Blender: versi tidak terbaca"
    except Exception:
        return "Blender: TIDAK DITEMUKAN — versi tidak tercatat"
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT_MD = os.path.join(HERE, "MANIFES_REPRODUKSIBILITAS.md")
OUT_REQ = os.path.join(ROOT, "requirements-pinned.txt")

PAKET = ["numpy", "scipy", "trimesh", "pillow", "rtree", "matplotlib"]


RENDER = {
    "mesin render": "Blender Cycles, 64 samples, device GPU (Metal di M4 Pro)",
    "resolusi": "1024 x 1024",
    "jumlah view": "4 ortogonal — front / left / back / right (urutan JANGAN ditukar)",
    "latar": "putih murni (255,255,255); derau Cycles menyisakan ~5% piksel sedikit meleset",
    "albedo objek": "0.42 (abu menengah)",
    "roughness": "0.6, metallic 0.0",
    "pencahayaan": "3 area light — 60 / 25 / 25 W",
    "view transform": "Standard (BUKAN Filmic/AgX — Filmic mengompresi kontras)",
    "normalisasi objek": "dipusatkan di origin, dimensi maksimum = 1 unit",
    "kamera": "ortografik, ortho_scale = margin, jarak 5.0",
    "margin": "1.30 (seragam). 1.15 lama TIDAK AMAN — memotong obj06 & obj08 di 20 derajat",
    "skrip": "eval/render_ortho_views.py (BUKAN salinan di root)",
}

def _param_hunyuan(nama_workflow):
    """Baca parameter beku LANGSUNG dari JSON workflow ComfyUI.

    KENAPA DIBACA, BUKAN DISALIN
        Sampai 18 Agu 2026 blok ini memuat string yang ditulis tangan:
        "seed 42, octree 448, surface-net, 30 steps, CFG 5". Empat parameter beku
        TIDAK tercatat di situ:

            EmptyLatentHunyuan3Dv2.resolution = 4096
                ← catatan di workflow sendiri menyatakan parameter ini LEBIH
                  menentukan detail geometri daripada octree_resolution
            VoxelToMesh.threshold             = 0.6   ← menggeser permukaan iso
            VAEDecodeHunyuan3D.num_chunks     = 8000
            ModelSamplingAuraFlow.shift       = 1

        Penelaah tidak bisa mereproduksi dari manifes yang kurang empat parameter.
        Dan selama nilainya disalin tangan, ia akan menyimpang dari workflow
        tanpa ada yang tahu — dua salinan yang "kebetulan masih sama".

        Sekarang manifes membaca JSON-nya. Kalau workflow berubah, manifes ikut.
    """
    import json
    p = os.path.join(ROOT, "workflow", nama_workflow) if 'ROOT' in globals() else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflow", nama_workflow)
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        return f"workflow TIDAK TERBACA ({nama_workflow}): {e}"

    w = {}
    for n in d.get("nodes", []):
        v = n.get("widgets_values")
        if v:
            w.setdefault(n.get("type"), []).append(v)

    def amb(tipe, idx, bawaan="?"):
        try:
            return w[tipe][0][idx]
        except Exception:
            return bawaan

    ks = w.get("KSampler", [[None] * 7])[0]
    return (
        f"checkpoint {amb('ImageOnlyCheckpointLoader', 0)} · "
        f"seed {ks[0]} ({ks[1]}) · steps {ks[2]} · cfg {ks[3]} · "
        f"{ks[4]}/{ks[5]} · denoise {ks[6]} · "
        f"EmptyLatent resolution {amb('EmptyLatentHunyuan3Dv2', 0)} · "
        f"VAEDecode num_chunks {amb('VAEDecodeHunyuan3D', 0)}, "
        f"octree_resolution {amb('VAEDecodeHunyuan3D', 1)} · "
        f"VoxelToMesh {amb('VoxelToMesh', 0)} threshold {amb('VoxelToMesh', 1)} · "
        f"ModelSamplingAuraFlow shift {amb('ModelSamplingAuraFlow', 0)} · "
        f"CLIPVisionEncode crop {amb('CLIPVisionEncode', 0)} · "
        f"[dibaca dari workflow/{nama_workflow}]"
    )


GENERATOR = {
    "Hunyuan3D-2mv (multi-view, 4 view)":
        _param_hunyuan("3d_hunyuan3d-v2mv_eksperimen_seed42_octree448.json"),
    "Hunyuan3D-2mv (single-view)":
        _param_hunyuan("3d_hunyuan3d-v2mv_eksperimen_singleview_seed42_octree448.json")
        + "  checkpoint SENGAJA sama dengan MV — perbedaan hasil murni dari jumlah view, bukan beda model",
    "Hunyuan — identitas checkpoint (diverifikasi 18 Agu 2026)":
        "model.fp16.safetensors dari repo HuggingFace tencent/Hunyuan3D-2mv · "
        "ukuran 4.928.151.562 byte · dipasang di ComfyUI 20 Des 2025. "
        "Workflow beku 9 Juli menamainya `hunyuan3d-dit-v2-mv.safetensors`; "
        "berkas yang sama di-rename setelah Juli menjadi "
        "`hunyuan3d-dit-v2-mv_fp16.safetensors`. "
        "Untuk reproduksi, yang mengikat adalah UKURAN + repo sumber, bukan nama "
        "berkas lokal — nama lokal terbukti berubah tanpa mengubah bobot model.",
    "Hunyuan — urutan view":
        "front.png→front · left.png→left · back.png→back · right.png→right. "
        "Salah pasang TIDAK memunculkan galat; hasilnya diam-diam rusak",
    "Hunyuan — cache ComfyUI":
        "Queue ulang tanpa mengubah input mengembalikan hasil cache dalam ~0 detik. "
        "Run 0 detik BUKAN data — buang dan jalankan ulang dengan input yang benar-benar berganti",
    "_Hunyuan3D-2mv (ringkasan lama, disimpan sebagai jejak)":
        "seed 42, octree 448, surface-net, 30 steps, CFG 5 "
        "(workflow beku: 3d_hunyuan3d-v2mv_eksperimen_seed42_octree448.json) "
        "— RINGKASAN INI KURANG 4 PARAMETER, jangan dipakai untuk reproduksi",
    "TripoSR": "default resmi, marching-cubes 256, --no-remove-bg, feedforward (tanpa seed)",
    "TRELLIS": "repo resmi microsoft/TRELLIS, sampler default 25 steps, seed 42, "
               "ATTN_BACKEND=xformers, SPCONV_ALGO=native, formats=['mesh'], "
               "to_glb resmi DILEWATI (mendesimasi ~95% segitiga)",
    "perangkat keras": "RTX 4090 24 GB — semua pengukuran waktu & VRAM",
}

EVALUATOR = {
    "normalisasi": "mesh diskalakan ke diagonal bbox GT = 1, lalu ICP",
    "ICP": "multi-start (identitas + rotasi acak terseed), pilih residu terendah",
    "seed": "42 (RNG metrik dipisah: seed + 12345)",
    "Chamfer": "100.000 titik permukaan, sampler determinstik ber-seed",
    "F-score": "tau 0.01 & 0.02 dari diagonal ternormalisasi",
    "Volume IoU": "voxel grid pitch 1/128 diagonal, flood-fill interior",
    "noise floor": "PER OBJEK 0.0032-0.0058 (GT vs dirinya sendiri) — "
                   "BUKAN satu angka global 0.0047",
    "keterulangan evaluator": "8e-6 sampai 3.8e-5 antar-seed evaluator "
                              "(5 seed x 8 objek, eval/noise_floor.csv)",
    "skrip": "eval/evaluate_mesh.py",
}


def versi_paket():
    out = {}
    for p in PAKET:
        try:
            mod = __import__(p if p != "pillow" else "PIL")
            out[p] = getattr(mod, "__version__", "?")
        except ImportError:
            out[p] = "(tidak terpasang)"
    return out


def blok(judul, d):
    s = [f"### {judul}", "", "| Parameter | Nilai |", "|---|---|"]
    for k, v in d.items():
        s.append(f"| {k} | {v} |")
    s.append("")
    return "\n".join(s)


def main():
    from silhocad_core import parameter_beku

    v = versi_paket()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    md = [
        "# Manifes Reproduksibilitas",
        "",
        f"**Dibuat otomatis:** {now} oleh `eval/manifes_reproduksibilitas.py`",
        "",
        "> Jangan menyunting berkas ini dengan tangan. Ubah sumbernya, lalu jalankan ulang.",
        "",
        "Seluruh nilai di bawah **beku**. Mengubah salah satunya membatalkan",
        "perbandingan dengan baris lama di `eval/hasil_evaluasi.csv`. Kalau memang",
        "harus berubah, buat **nama kondisi baru** di CSV — jangan menyunting di tempat.",
        "",
        "---",
        "",
        "## 1. Praproses siluet deterministik",
        "",
        "Sumber tunggal: `eval/silhocad_core.py`",
        "",
        "| Parameter | Nilai |",
        "|---|---|",
    ]
    for k, val in parameter_beku().items():
        md.append(f"| {k} | {val} |")
    md += [
        "",
        "Pengganti rembg/u2net, yang gagal sistematis pada render CAD abu-di-putih:",
        "pada TripoSR menyatakan seluruh kanvas sebagai foreground; pada TRELLIS",
        "justru menghapus bagian tengah objek.",
        "",
        "---",
        "",
        "## 2. Konfigurasi view",
        "",
        blok("Render", RENDER),
        "---",
        "",
        "## 3. Generator yang diuji",
        "",
        blok("Parameter beku per model", GENERATOR),
        "---",
        "",
        "## 4. Evaluator",
        "",
        blok("Protokol", EVALUATOR),
        "---",
        "",
        "## 5. Lingkungan",
        "",
        f"- Python {platform.python_version()} · {platform.system()} {platform.machine()}",
        f"- {_versi_os()}",
        f"- {_versi_blender()}",
        "",
        "> Versi OS dan Blender dicatat sejak 13 Agu 2026. Sebelum itu manifes hanya",
        "> menulis `Darwin`, tanpa nomor versi — sehingga pergeseran lingkungan tidak",
        "> bisa ditelusuri dari manifes lama.",
        "",
        "| Paket | Versi |",
        "|---|---|",
    ]
    for k, val in v.items():
        md.append(f"| {k} | {val} |")
    md += [
        "",
        "> **`trimesh` WAJIB dikunci.** `chamfer` dan `normal_consistency`",
        "> reproduksibel bit-identik lintas sistem operasi dan lintas versi trimesh,",
        "> tetapi **`volume_iou` tidak** — voxelisasi berubah antar versi",
        "> (0.999163 vs 0.998983 pada GT-vs-GT obj01, diukur 2 Agu 2026).",
        "",
        "---",
        "",
        "## 6. Uji regresi sebelum menambah baris ke CSV",
        "",
        "Wajib dijalankan setelah jeda panjang atau update dependensi:",
        "",
        "```bash",
        "cd eval",
        "python evaluate_mesh.py \\",
        "  --gt ../dataset/obj01_dr500.stl \\",
        "  --pred ../results/hunyuan_mv_elev20/obj01.glb \\",
        "  --model CEK_DETERMINISME --csv /tmp/cek.csv",
        "```",
        "",
        "Harus persis: `chamfer = 0.036233` · `fscore@0.01 = 0.276840` · `volume_iou = 0.676786`",
        "",
        "Kalau hanya `volume_iou` yang meleset, penyebabnya hampir pasti versi trimesh.",
        "",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    with open(OUT_REQ, "w", encoding="utf-8") as f:
        f.write("# Versi terkunci — dibuat otomatis oleh eval/manifes_reproduksibilitas.py\n")
        f.write(f"# {now}\n")
        f.write("# trimesh WAJIB dikunci: volume_iou berbeda antar versi.\n")
        for k, val in v.items():
            if val not in ("?", "(tidak terpasang)"):
                f.write(f"{k}=={val}\n")

    print(f"Tersimpan:\n  {OUT_MD}\n  {OUT_REQ}\n")
    print("Versi terdeteksi:")
    for k, val in v.items():
        print(f"  {k:12} {val}")


if __name__ == "__main__":
    main()
