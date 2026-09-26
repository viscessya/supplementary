# Supplementary Material S1: Generation and Evaluation Reproducibility

*Concentrated gains from camera elevation in diffusion-based engineering component reconstruction*

This document provides the checkpoint identifier, frozen generation parameters, ComfyUI workflow graph, evaluation protocol, and software environment referenced in Sections 3.3–3.5 of the manuscript, so that the reported results can be reproduced or audited independently. All values below are frozen; changing any of them would invalidate direct comparison with the results reported in Table 4, and any future variant should be recorded under a new condition name rather than overwriting these settings.

## 1. Input rendering (Section 3.1–3.3)

| Parameter | Value |
|---|---|
| Renderer | Blender Cycles, tested on Blender 3.6+ / 4.x |
| Samples | 64, GPU device (Metal on Apple M4 Pro in this study) |
| Resolution | 1024 x 1024 |
| Views | 4 orthographic — front / left / back / right, in this fixed order |
| Background | Pure white (255,255,255) |
| Object material (default, "clay") | base color 0.42, roughness 0.6, metallic 0.0 |
| Lighting | 3 area lights, 60 / 25 / 25 W |
| View transform | Standard (not Filmic or AgX, which compress contrast) |
| Object normalization | centered at origin, maximum dimension scaled to 1 unit |
| Camera | orthographic, ortho_scale = margin, distance 5.0 |
| Margin | 1.30 (a margin of 1.15 clips two pilot objects at elevated views) |
| Script | `render/render_ortho_views.py` (the root-level copy of the same filename is a stale duplicate and must not be used — see the file's own header) |

### Input-conditioning variants (Section 3.3)

All variants share the render configuration above except for the single changed parameter noted:

| Variant | Change from the unmodified render |
|---|---|
| Deterministic silhouette | Foreground computed directly from ground-truth CAD geometry (not a learned background-removal model), composited onto a neutral gray background (value 0.5) |
| Gray background | World background color changed from pure white to neutral gray (0.5); this also slightly reduces ambient light, since Blender Cycles draws ambient illumination from the world background node |
| Semi-metallic material | Base color RGB 0.55 / 0.56 / 0.58, roughness 0.35, metallic 0.4 (replaces the default clay material) |
| Diagonal azimuths | Camera azimuths rotated 45 degrees from the standard front/left/back/right positions, to 45 / 135 / 225 / 315 degrees; elevation, material, margin, and camera type unchanged |
| Perspective projection | Orthographic camera replaced with a perspective camera whose field of view is calibrated from the object's center distance, matching the orthographic baseline at the same 1.30 margin, widening only for the rare object/view combination that would otherwise clip |

All five variants were rendered at every tested elevation (0, 10, 20, 30, 40 degrees).

## 2. Hunyuan3D-2mv generation (Section 3.4)

### Checkpoint identifier

- File name: `hunyuan3d-dit-v2-mv_fp16.safetensors`
- Verified file size: 4,928,151,562 bytes
- Note: the workflow graph (frozen 2026-07-09) originally referenced this checkpoint under the name `hunyuan3d-dit-v2-mv.safetensors`. That file was renamed on disk after July 2026; the byte size above was used on 2026-08-18 to confirm the renamed file is the identical checkpoint (same weights), not a substituted one.

### Frozen sampling parameters

| ComfyUI node | Parameter | Value |
|---|---|---|
| ImageOnlyCheckpointLoader | ckpt_name | hunyuan3d-dit-v2-mv_fp16.safetensors |
| KSampler | seed | 42 (frozen in the graph). Seeds 123 and 777 are applied at run time by `generation/run_hunyuan_comfy_n30.py` through `SEED_OVERRIDE`, which patches only the runtime copy of the workflow; the graph on disk stays at 42 so the frozen-parameter gate keeps working. Arms using an override carry the seed in their name |
| KSampler | steps | 30 |
| KSampler | cfg | 5.0 |
| KSampler | sampler_name | euler |
| KSampler | scheduler | normal |
| KSampler | denoise | 1 |
| EmptyLatentHunyuan3Dv2 | resolution | 4096 |
| VAEDecodeHunyuan3D | num_chunks | 8000 |
| VAEDecodeHunyuan3D | octree_resolution | 448 |
| VoxelToMesh | algorithm | surface net |
| VoxelToMesh | threshold | 0.6 |
| ModelSamplingAuraFlow | shift | 1 |

The same fixed checkpoint and parameter set is used for every condition in Table 4; only the four rendered input views (Section 1 above) differ between conditions.

### Mesh extractor

All reported geometry comes from ComfyUI's `VoxelToMesh` node with the "surface net" algorithm, not from the marching-cubes extraction of the Hunyuan3D reference implementation. ComfyUI's surface net builds a quad for every 2x2 group of adjacent active cells, so its meshes carry thousands of non-manifold edges and none is watertight. To separate this from the generator, the 60 seed-42 grids of the primary comparison were regenerated (reproducing every original metric to six decimals) and also extracted with scikit-image marching cubes at the same threshold, via the custom node in `generation/comfy_node_mc/`. Result (`results/analisis_uji_mc_23sep.txt`): surface net 0/60 watertight, median 5,224 non-manifold edges; marching cubes 60/60 watertight, 0 non-manifold and 0 boundary edges; median Chamfer difference between the two extractors 0.00005. The 0-to-20-degree Chamfer gain stays significant under marching cubes (seed 42, Holm-adjusted p = 0.033 against 0.029).

### Workflow graph

The ComfyUI workflow graphs are included in this supplementary package in ComfyUI's native GUI export format (importable directly via ComfyUI's "Open" / "Load" menu):

- `generation/workflow/3d_hunyuan3d-v2mv_eksperimen_seed42_octree448.json` — multi-view (4-input) branch, used for all elevation and conditioning-variant runs
- `generation/workflow/3d_hunyuan3d-v2mv_eksperimen_singleview_seed42_octree448.json` — single-view (1-input) branch

To submit these programmatically via ComfyUI's `/prompt` API endpoint (as this study's batch-run script does), export each graph to ComfyUI's API format from a running ComfyUI instance (Workflow menu -> Export (API), or via ComfyUI's `/object_info` endpoint, since the GUI format stores widget values as unlabeled lists and the correct parameter names can only be recovered from a live ComfyUI instance). Before running any batch, verify the exported API graph reproduces every frozen parameter listed above; the batch-run script used in this study (`pc_scripts/run_hunyuan_comfy_n30.py`) performs this check automatically and refuses to run if a mismatch is detected.

### Generation hardware

RTX 4090, 24 GB VRAM.

### Known documentation gap

The exact ComfyUI build, PyTorch version, and CUDA version used on the generation machine were not recorded at the time of the runs and could not be reconstructed from the project files. Anyone attempting an exact reproduction should treat the checkpoint identifier and sampling parameters above as authoritative, and expect minor floating-point-level differences if the ComfyUI/PyTorch/CUDA stack differs from the original.

## 3. Evaluation protocol (Section 3.5)

| Step | Detail |
|---|---|
| Normalization | Each mesh is centered at its bounding-box centroid and scaled so its bounding-box diagonal equals 1 |
| Alignment | Multi-start ICP: 5 initializations (identity + 4 seeded random rotations, seed 42), up to 60 iterations each, convergence tolerance 1e-7 on RMS registration error; the lowest-residual initialization is kept |
| Chamfer distance | 100,000 deterministically sampled surface points per mesh (area-weighted, barycentric); computed as the sum of the two mean nearest-neighbor distances (`d_pred->gt.mean() + d_gt->pred.mean()`) — **this is an unsquared distance sum**. Equation (1) of the manuscript was corrected on 2026-09-20 to match this implementation; earlier drafts wrote it with squared distances. No reported number changed, since every value always came from this code |
| F-score | Precision/recall at tau = 0.01 and 0.02 of the normalized bounding-box diagonal; F-score is their harmonic mean |
| Normal consistency | Bidirectional mean absolute cosine between matched-point normals (unoriented: a flipped normal still counts as consistent) |
| Volumetric IoU | 128^3 voxel grid (pitch = 1/128 of the normalized diagonal); occupancy from surface voxelization with flood-fill interior detection, not generalized winding number |
| Metric RNG | Separated from the ICP RNG (metric seed = ICP seed + 12345 = 12387), so the number of draws ICP uses never changes the metric values |
| Script | `evaluation/evaluate_mesh.py` |

### Regression / determinism check

Before adding any new row to the results CSV after a long gap or a dependency update, this exact check should reproduce the following values bit-for-bit (aside from `volume_iou`, which is sensitive to the installed `trimesh` version — see below):

```
python evaluate_mesh.py --gt dataset/obj01_dr500.stl --pred results/hunyuan_mv_elev20/obj01.glb --model CEK_DETERMINISME --csv /tmp/cek.csv
```

Expected: `chamfer = 0.036233`, `fscore@0.01 = 0.276840`, `volume_iou = 0.676786`.

## 4. Software environment (evaluation)

- Python 3.14.5, Darwin arm64 (Apple Silicon)

| Package | Version |
|---|---|
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| trimesh | 4.12.2 |
| pillow | 12.3.0 |
| rtree | 1.4.1 |
| matplotlib | 3.11.1 |

`trimesh` must be pinned to this exact version: `chamfer` and `normal_consistency` are bit-reproducible across operating systems and trimesh versions, but `volume_iou` is not — its voxelization changed measurably between trimesh versions during this study's development (0.999163 vs. 0.998983 on a GT-vs-itself sanity check for obj01, observed 2026-08-02).

## 5. Files included in this repository

| File | Purpose |
|---|---|
| `generation/workflow/3d_hunyuan3d-v2mv_eksperimen_seed42_octree448.json` | Hunyuan3D-2mv multi-view ComfyUI workflow graph |
| `generation/workflow/3d_hunyuan3d-v2mv_eksperimen_singleview_seed42_octree448.json` | Hunyuan3D-2mv single-view ComfyUI workflow graph |
| `generation/run_hunyuan_comfy_n30.py`, `generation/konversi_workflow_ke_api.py` | Batch generation runner and GUI-to-API workflow converter |
| `render/*.py` | Blender scripts for the unmodified render and all five conditioning variants |
| `evaluation/evaluate_mesh.py` | Evaluation script (Chamfer, F-score, normal consistency, volumetric IoU, mesh-quality indicators) |
| `evaluation/jalankan_evaluasi_n30.py`, `evaluation/gerbang_mesh_n30.py`, `evaluation/verifikasi_kontribusi_n30.py`, `evaluation/manifes_reproduksibilitas.py` | Batch evaluation, mesh-quality gating, contribution-claim verification, and environment-manifest generation |
| `dataset_manifest_supplementary.csv` | Per-object manifest: manufacturer, part number, source URL, dimensions, triangle count, watertightness, non-manifold edge count |
| `analysis/evaluasi_seed_n30.py`, `analysis/analisis_seed_n30.py` | Three-seed replication of the primary comparison (Table 3): evaluation and the pre-specified analysis with its gates, tests, and prediction checks |
| `analysis/evaluasi_seed_tabel3.py`, `analysis/analisis_seed_tabel3.py` | Second-seed replication of the one positive result in Table 4 (perspective projection at 30 and 40 degrees) and of the single gray-background difference at 40 degrees |
| `analysis/hitung_visibilitas.py`, `analysis/plot_peta_histogram_visibilitas.py` | Surface-visibility measure computed from the reference meshes alone, and the figures and tests built on it (the two surface-visibility figures) |
| `analysis/susun_galeri_pendorong.py` | Assembles the driver-component gallery figure from the Blender renders and the three-seed per-object table |
| `generation/comfy_node_mc/__init__.py`, `generation/uji_marching_cubes.py` | Custom ComfyUI marching-cubes node and the runner for the extractor comparison |
| `analysis/evaluasi_uji_mc.py`, `analysis/uji_sintetis_surfnet_vs_mc.py` | Evaluation and pre-specified analysis of the extractor comparison; synthetic surface-net vs marching-cubes pre-test |
| `results/hasil_uji_mc_23sep.csv`, `results/analisis_uji_mc_23sep.txt`, `results/uji_mc_runlog.csv` | Per-mesh metrics for both extractors, the analysis log with gates and prediction checks, and the generation run log |
| `results/hasil_evaluasi_n30.csv` | Per-object metrics for every elevation-by-condition arm at seed 42 (source of Table 4) |
| `results/hasil_seed_n30.csv`, `results/seed_n30_per_objek.csv`, `results/analisis_seed_n30.txt` | Per-object metrics for seeds 123 and 777 at 0 and 20 degrees, the per-object three-seed summary, and the analysis log (source of Table 3) |
| `results/hasil_seed_tabel3.csv`, `results/analisis_seed_tabel3.txt` | Per-object metrics and analysis log for the second-seed replication |
| `results/visibilitas_per_objek.csv`, `results/ringkasan_uji_visibilitas.txt` | Per-object visibility table behind the two surface-visibility figures, including the percentage-point values quoted in Section 4.2, and the log of every test tried, including the ones that failed |
| `analysis/analisis_fitur_objek.py` | Computes `results/fitur_objek.csv` from the reference meshes |
| `GLOSSARY.md` | English translations of the Indonesian file, column, and log terms |
| `results/fitur_objek.csv` | Per-object reference-mesh features: dimensions, triangle count, watertightness, non-manifold edges, and visible surface share at each elevation |
| `requirements-pinned.txt` | Exact evaluation-environment package versions |
| `photographs/potong_dan_hapus_latar.py`, `photographs/segmentasi_birefnet_lite.py` | Smartphone-photograph demonstration (Section 4.4): common square crop and BiRefNet-general-lite background removal |
| `photographs/inputs/` | The four 1024 x 1024 photograph crops with the real background and with the background removed (the Hunyuan3D-2mv input) |
| `photographs/render_pompa_clay.py`, `photographs/render_pompa_clay_transparan.py`, `photographs/susun_figure_pompa_v2.py` | Clay renders of the pump reconstruction and the builder of the photograph figure |

See `README.md` for the full repository layout. The manufacturer-published CAD files themselves are not included, since catalogue terms prohibit redistribution; the manifest above records each component's manufacturer and part number for independent sourcing, consistent with the manuscript's Data Availability statement.
