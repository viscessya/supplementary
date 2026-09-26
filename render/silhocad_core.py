# -*- coding: utf-8 -*-
"""Deterministic silhouette preprocessing, defined once and imported by every script that needs it.

Used by the deterministic-silhouette conditioning variant. The values here are frozen: changing them would
define a new experimental condition, not an edit of this one.
"""
import numpy as np


ALPHA_FULL_BELOW = 200

FOREGROUND_RATIO = 0.85

GRAY = 0.5

CANVAS_TRIPOSR = 512
CANVAS_TRELLIS = 1024   # TRELLIS memakai ukuran render asli; crop/scale oleh pipeline


def extract_alpha(rgb: np.ndarray) -> np.ndarray:
    """
    Alpha siluet deterministik dari render berlatar putih terkontrol.

    Ini adalah praproses siluet deterministik: pengganti rembg/u2net, yang gagal
    sistematis pada render CAD abu-di-putih (kontras rendah) — pada TripoSR
    rembg menyatakan seluruh kanvas sebagai foreground, pada TRELLIS rembg
    justru menghapus bagian tengah objek.

    Sifat yang penting: sepenuhnya deterministik, tanpa model terlatih, dan
    mempertahankan tepi anti-alias (alpha linier, bukan biner).

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3)
        Citra RGB. float atau uint8; dibaca sebagai float32.

    Returns
    -------
    np.ndarray, shape (H, W), float32 dalam [0, 1]
    """
    g = np.asarray(rgb).max(axis=2).astype(np.float32)
    return np.clip((255.0 - g) / (255.0 - ALPHA_FULL_BELOW), 0.0, 1.0)


def parameter_beku() -> dict:
    """Ringkasan parameter untuk dicetak ke log / manifes reproduksibilitas."""
    return {
        "alpha_full_below": ALPHA_FULL_BELOW,
        "foreground_ratio": FOREGROUND_RATIO,
        "gray_background": GRAY,
        "canvas_triposr": CANVAS_TRIPOSR,
        "canvas_trellis": CANVAS_TRELLIS,
    }
