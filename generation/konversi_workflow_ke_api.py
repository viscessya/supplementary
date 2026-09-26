# -*- coding: utf-8 -*-
"""Converts the ComfyUI workflow graphs in workflow/ from the GUI format (nodes and links) to the API
format that the /prompt endpoint requires.

The GUI format stores widget values as an unnamed list, and some entries (e.g. control_after_generate of
KSampler) do not exist in the API format, so guessing names from positions silently shifts parameters.
Input names and their order are therefore read from the running ComfyUI via GET /object_info.
The GUI workflows are only read; new api_*.json files are written. The frozen parameters of the result are
checked by run_hunyuan_comfy_n30.py --cek.

Usage (ComfyUI running):
    python konversi_workflow_ke_api.py
    python run_hunyuan_comfy_n30.py --cek
"""
import json
import os
import sys
import urllib.request

COMFY = "http://127.0.0.1:8188"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASANGAN = [
    ("3d_hunyuan3d-v2mv_eksperimen_seed42_octree448.json", "api_hunyuan_mv.json"),
    ("3d_hunyuan3d-v2mv_eksperimen_singleview_seed42_octree448.json", "api_hunyuan_sv.json"),
]

RENAME_CHECKPOINT = {
    "hunyuan3d-dit-v2-mv.safetensors": "hunyuan3d-dit-v2-mv_fp16.safetensors",
}

TIPE_WIDGET = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
DILEWATI = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}


def ambil_object_info():
    try:
        with urllib.request.urlopen(COMFY + "/object_info", timeout=60) as r:
            return json.load(r)
    except Exception as e:
        sys.exit(f"BERHENTI: tidak bisa mengambil /object_info dari {COMFY}\n"
                 f"   Pastikan ComfyUI sedang jalan. Galat: {e}")


def nama_widget(spec_input):
    """Kembalikan daftar (nama, punya_control_after_generate) untuk input yang berupa widget."""
    hasil = []
    for bagian in ("required", "optional"):
        for nama, spec in (spec_input.get(bagian) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            tipe = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(tipe, list) or tipe in TIPE_WIDGET:
                hasil.append((nama, bool(opts.get("control_after_generate"))))
    return hasil


def konversi(path_gui, object_info):
    gui = json.load(open(path_gui, encoding="utf-8"))
    if "nodes" not in gui:
        sys.exit(f"PERHATIAN: {path_gui} sudah format API, bukan GUI.")

    peta_link = {}
    for l in gui.get("links", []):
        peta_link[l[0]] = (str(l[1]), l[2])

    api = {}
    catatan = []
    for n in gui["nodes"]:
        ct = n.get("type")
        if ct in DILEWATI or n.get("mode") in (2, 4):     # 2=muted, 4=bypassed
            continue
        if ct not in object_info:
            sys.exit(f"BERHENTI: node '{ct}' tidak dikenal ComfyUI ini. "
                     f"Custom node belum terpasang?")
        nid = str(n["id"])
        inputs = {}

        tersambung = set()
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None:
                asal = peta_link.get(inp["link"])
                if asal:
                    inputs[inp["name"]] = [asal[0], asal[1]]
                    tersambung.add(inp["name"])

        nilai = list(n.get("widgets_values") or [])
        if isinstance(nilai, dict):
            nilai = list(nilai.values())
        i = 0
        for nama, punya_control in nama_widget(object_info[ct].get("input", {})):
            if nama in tersambung:
                continue
            if i >= len(nilai):
                break
            inputs[nama] = nilai[i]
            i += 1
            if punya_control:
                i += 1        # lewati nilai control_after_generate milik GUI
                catatan.append(f"{ct}.{nama}: melewati control_after_generate")

        if ct == "ImageOnlyCheckpointLoader":
            lama = inputs.get("ckpt_name")
            if lama in RENAME_CHECKPOINT:
                inputs["ckpt_name"] = RENAME_CHECKPOINT[lama]
                catatan.append(
                    f"🔁 checkpoint dipetakan: {lama} → {RENAME_CHECKPOINT[lama]} "
                    f"(berkas identik, 4.928.151.562 byte — diverifikasi 18 Agu 2026)")

        req = object_info[ct].get("input", {}).get("required") or {}
        opt = object_info[ct].get("input", {}).get("optional") or {}
        for nama, v in inputs.items():
            if isinstance(v, list):      # sambungan antar node, bukan nilai
                continue
            spec = req.get(nama) or opt.get(nama)
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            t = spec[0]
            salah = None
            if t == "INT" and not isinstance(v, int):
                salah = "INT"
            elif t == "FLOAT" and not isinstance(v, (int, float)):
                salah = "FLOAT"
            elif t == "BOOLEAN" and not isinstance(v, bool):
                salah = "BOOLEAN"
            if salah:
                skema = {k: (s[0] if isinstance(s, (list, tuple)) and s else s)
                         for k, s in req.items()}
                sys.exit(
                    f"\nBERHENTI: nilai widget BERGESER di node '{ct}'.\n"
                    f"   {ct}.{nama} = {v!r}  ← seharusnya bertipe {salah}\n\n"
                    f"   nilai GUI      : {nilai}\n"
                    f"   hasil pemetaan : "
                    f"{ {k: x for k, x in inputs.items() if not isinstance(x, list)} }\n"
                    f"   skema ComfyUI  : {skema}\n\n"
                    f"   Ada tipe input yang belum dikenali skrip ini. Tambahkan\n"
                    f"   tipe itu ke TIPE_WIDGET di kepala berkas.\n"
                    f"   🚫 JANGAN abaikan — pergeseran nilai tidak memunculkan\n"
                    f"      galat apa pun saat generate.")

        sisa = [x for x in nilai[i:] if x not in ("", None)]
        if sisa:
            catatan.append(f"PERHATIAN: {ct}: {len(sisa)} nilai GUI tidak terpetakan: {sisa}")

        api[nid] = {"class_type": ct, "inputs": inputs}
    return api, catatan


def main():
    print("=" * 76)
    oi = ambil_object_info()
    print(f"  OK /object_info terbaca — {len(oi)} tipe node dikenal ComfyUI")
    print("=" * 76)

    for gui_nama, api_nama in PASANGAN:
        p_gui = os.path.join(BASE, "workflow", gui_nama)
        p_api = os.path.join(BASE, "workflow", api_nama)
        if not os.path.exists(p_gui):
            print(f"  lewati — tidak ada: {gui_nama}")
            continue

        if os.path.exists(p_api):
            os.remove(p_api)
            print(f"  🗑️  hasil konversi lama dihapus: {api_nama}")
        api, catatan = konversi(p_gui, oi)
        json.dump(api, open(p_api, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\n■ {gui_nama}\n  → {api_nama}  ({len(api)} node)")
        for c in sorted(set(catatan)):
            print(f"     · {c}")
        for nid, node in api.items():
            if node["class_type"] in ("KSampler", "VAEDecodeHunyuan3D", "VoxelToMesh",
                                      "EmptyLatentHunyuan3Dv2", "ModelSamplingAuraFlow"):
                wid = {k: v for k, v in node["inputs"].items() if not isinstance(v, list)}
                print(f"     {node['class_type']:26} {wid}")

    print("\n" + "=" * 76)
    print("LANGKAH BERIKUTNYA — WAJIB, jangan langsung menjalankan 90 run:")
    print("    python pc_scripts/run_hunyuan_comfy_n30.py --cek")
    print("Gerbang itu mencocokkan hasil konversi dengan 9 parameter beku.")
    print("Kalau ia berhenti, konversinya salah — jangan diakali, laporkan pesannya.")


if __name__ == "__main__":
    main()
