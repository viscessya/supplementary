#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assembles the driver-component gallery figure: reference CAD, 0-degree reconstruction, and
20-degree reconstruction (rows) for obj25, obj20, obj05, and obj24 (columns).

The panels are Blender clay renders made beforehand with the same isometric camera for every panel,
including the reference. This script only trims their white margins and lays them out; the Chamfer gains
in the labels are read from eval/seed_n30_per_objek.csv (column dCD_rerata, three-seed mean).

Usage: python eval/susun_galeri_pendorong.py   (writes figur/fig_galeri_pendorong.png and a copy for the manuscript)
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RENDER = os.path.join(BASE, "ijai_v2", "gallery_median")
CSV_SEED = os.path.join(BASE, "eval", "seed_n30_per_objek.csv")
KELUARAN = [os.path.join(BASE, "figur", "fig_galeri_pendorong.png"),
            os.path.join(BASE, "ijai_v2", "fig_galeri_pendorong.png")]

KOLOM = [
    ("obj25", "Valve block",
     "obj25_hunyuan_m130_elev0_seed123.png", "obj25_hunyuan_m130_elev20_seed123.png"),
    ("obj20", "DIN-rail\nenclosure",
     "obj20_hunyuan_m130_elev0.png", "obj20_hunyuan_m130_elev20_seed123.png"),
    ("obj05", "Worm-gear\nbrake unit",
     "obj05_hunyuan_m130_elev0_seed123.png", "obj05_hunyuan_m130_elev20_seed123.png"),
    ("obj24", "Hydraulic\nmanifold",
     "obj24_hunyuan_m130_elev0_seed777.png", "obj24_hunyuan_m130_elev20_seed123.png"),
]
LABEL_BARIS = ["Ground\ntruth", "$0^\\circ$", "$20^\\circ$"]
ABU_TEKS = "#2b2b2b"
MARGIN = 0.02  # margin relatif setelah potong latar


AMBANG = 240   # di bawah ini dianggap objek, bukan latar
MIN_PIKSEL = 4  # baris/kolom butuh sebanyak ini supaya tidak terkecoh derau render


def potong(path):
    """Buang latar putih, sisakan margin kecil. Mengembalikan array RGB.

    Latar render Cycles tidak putih bersih: ada piksel derau yang turun sampai
    241. Memakai ambang saja membuat kotak potong selalu seluas gambar, jadi
    baris/kolom baru dihitung sebagai isi kalau punya minimal MIN_PIKSEL piksel
    gelap.
    """
    a = np.asarray(Image.open(path).convert("RGB"))
    gelap = a.min(axis=2) < AMBANG
    baris = np.where(gelap.sum(axis=1) >= MIN_PIKSEL)[0]
    kolom = np.where(gelap.sum(axis=0) >= MIN_PIKSEL)[0]
    if baris.size == 0 or kolom.size == 0:
        sys.exit(f"BERHENTI: render kosong (semua putih): {path}")
    my = int(round((baris[-1] - baris[0] + 1) * MARGIN)) + 2
    mx = int(round((kolom[-1] - kolom[0] + 1) * MARGIN)) + 2
    y0, y1 = max(baris[0] - my, 0), min(baris[-1] + my + 1, a.shape[0])
    x0, x1 = max(kolom[0] - mx, 0), min(kolom[-1] + mx + 1, a.shape[1])
    return a[y0:y1, x0:x1]


def dcd_tiga_seed():
    if not os.path.exists(CSV_SEED):
        sys.exit(f"BERHENTI: tidak ada {CSV_SEED}")
    with open(CSV_SEED, newline="", encoding="utf-8") as f:
        return {r["objek"]: float(r["dCD_rerata"]) for r in csv.DictReader(f)}


def main():
    d = dcd_tiga_seed()
    gambar = []  # [baris][kolom]
    for baris in range(3):
        satu = []
        for oid, _, f0, f20 in KOLOM:
            nama = f"{oid}_reference.png" if baris == 0 else (f0 if baris == 1 else f20)
            p = os.path.join(DIR_RENDER, nama)
            if not os.path.exists(p):
                sys.exit("BERHENTI: render tidak ada: %s\n"
                         "Render clay galeri dibuat di Blender "
                         "dan tidak disertakan di repositori ini." % p)
            satu.append(potong(p))
        gambar.append(satu)

    rasio = [max(g[j].shape[1] / g[j].shape[0] for g in gambar) for j in range(4)]
    tinggi_baris = 1.95
    KIRI, KANAN = 0.092, 0.948   # sisa kiri untuk label baris, sisa kanan untuk judul kolom obj24
    lebar_gambar = sum(rasio) * tinggi_baris / (KANAN - KIRI)
    fig = plt.figure(figsize=(lebar_gambar, tinggi_baris * 3 + 0.82))
    gs = fig.add_gridspec(3, 4, width_ratios=rasio,
                          left=KIRI, right=KANAN, top=0.775, bottom=0.005,
                          wspace=0.015, hspace=0.02)
    for i in range(3):
        for j, (oid, nama, _, _) in enumerate(KOLOM):
            a = fig.add_subplot(gs[i, j])
            a.imshow(gambar[i][j])
            a.set_axis_off()
            if i == 0:
                a.set_title("%s\n%s\nGain = %.3f" % (oid, nama, d[oid]),
                            fontsize=17, color=ABU_TEKS, pad=7, linespacing=1.25)
            if j == 0:
                a.text(-0.045, 0.5, LABEL_BARIS[i], transform=a.transAxes,
                       fontsize=17.5, color=ABU_TEKS, ha="right", va="center",
                       linespacing=1.15)
    for p in KELUARAN:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=200, facecolor="white")
        print("ditulis:", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
