# Reproducibility Repository

Code, workflow graphs, and per-object dataset metadata supporting:

*Concentrated gains from camera elevation in diffusion-based engineering component reconstruction*

Camera elevation on the unmodified render is the paper's primary, confirmatory result (Section 4.1); the five input-conditioning variants below, including the deterministic-silhouette preprocessing in `render/silhocad_core.py`, are exploratory comparisons against that same unmodified render (Section 3.3/4.3), not the study's main finding.

The primary comparison is averaged over three generation seeds (42, 123, 777). The elevation sweep and the conditioning variants use seed 42, except for the rows replicated under seed 123 that are described in Section 3.5 of the manuscript. A third study re-extracted the 60 seed-42 grids of the primary comparison with marching cubes to separate the generator from ComfyUI's surface-net mesh extractor (Sections 3.5 and 4.3). A qualitative demonstration on smartphone photographs of a physical water pump (Section 4.4) is documented in `photographs/`. All three studies were planned in writing before their new meshes were generated. The plans themselves are internal working notes and are not included; the analysis scripts carry the fixed gates and tests and print each prediction next to its outcome.

This repository does not include the manufacturer-published CAD source files (catalogue terms prohibit redistribution) or the Hunyuan3D-2mv checkpoint itself (a third-party model, not redistributed here). `dataset_manifest_supplementary.csv` records each component's manufacturer and part number for independent sourcing. See `SUPPLEMENTARY_MATERIAL.md` for the full checkpoint identifier, frozen parameters, and evaluation protocol.

Scripts, logs, and some table columns keep their original Indonesian names; `GLOSSARY.md` translates them. Scripts and logs written before the manuscript's results tables were renumbered call Table 4 "Table 3" and vice versa; file names containing `tabel3` refer to what is now Table 4.

## Pipeline overview

```
render/          Blender scripts that turn a CAD mesh into the unmodified render and the five conditioning variants
generation/      ComfyUI workflow graphs, the batch runner that drives Hunyuan3D-2mv, and the marching-cubes node for the extractor comparison
evaluation/      Metric computation, mesh-quality gating, and result verification
analysis/        Seed-replication studies, the extractor comparison, the surface-visibility measure, and the figure builders
results/         The per-object tables and analysis logs behind every number in the manuscript
photographs/     Smartphone-photograph demonstration: cropping, background removal, inputs, renders, and the figure builder
```

### 1. `render/` — input image generation (run in Blender)

| Script | Produces |
|---|---|
| `render_ortho_views.py` | The unmodified render: 4 orthographic views (front/left/back/right), any elevation |
| `render_ortho_views_backdrop_v1.py` | Gray-background variant |
| `render_ortho_views_material_v1.py` | Semi-metallic-material variant |
| `render_ortho_views_diagonal_v1.py` | Diagonal-azimuth variant (45/135/225/315 degrees) |
| `render_persp_views_v4.py` | Perspective-projection variant |
| `silhocad_core.py` | Deterministic-silhouette preprocessing, shared by the silhouette conditioning variant |

All five conditioning variants share the base parameters in `SUPPLEMENTARY_MATERIAL.md` Section 1 except for the one change each script name describes.

### 2. `generation/` — Hunyuan3D-2mv shape generation (run against a live ComfyUI instance)

| File | Purpose |
|---|---|
| `workflow/3d_hunyuan3d-v2mv_eksperimen_seed42_octree448.json` | Multi-view (4-input) workflow graph — GUI export format, load via ComfyUI's Open/Load menu |
| `workflow/3d_hunyuan3d-v2mv_eksperimen_singleview_seed42_octree448.json` | Single-view (1-input) workflow graph |
| `konversi_workflow_ke_api.py` | Converts a workflow graph from GUI format to the API format ComfyUI's `/prompt` endpoint requires (queries a running ComfyUI's `/object_info` for correct widget names, rather than guessing from position) |
| `comfy_node_mc/__init__.py` | Custom ComfyUI node `SimpanMeshMarchingCubes`: extracts a mesh with scikit-image marching cubes from the same decoded voxel grid that `VoxelToMesh` receives, at the same threshold (0.6), in the same coordinate frame. Copy the folder into `ComfyUI/custom_nodes/` |
| `uji_marching_cubes.py` | Regenerates the 60 seed-42 reconstructions of the primary comparison and saves, from one prompt each, the surface-net mesh (`VoxelToMesh`) and the marching-cubes mesh of the same grid |
| `run_hunyuan_comfy_n30.py` | Batch-runs the converted API workflow over all objects/elevations/conditions via ComfyUI's API, tracking every run (object, elevation, arm, prompt_id) and refusing to run if the exported workflow's parameters do not match the frozen values in `SUPPLEMENTARY_MATERIAL.md` |

### 3. `evaluation/` — metrics and verification

| Script | Purpose |
|---|---|
| `evaluate_mesh.py` | Computes Chamfer distance, F-score@{0.01,0.02}, normal consistency, volumetric IoU, and mesh-quality indicators (watertightness, non-manifold/boundary edges, connected components) for one predicted/reference mesh pair |
| `jalankan_evaluasi_n30.py` | Batch-runs `evaluate_mesh.py` over the full 30-object x elevation x condition grid |
| `gerbang_mesh_n30.py` | Pre-evaluation mesh-quality gate (catches degenerate or empty predictions before they are scored) |
| `verifikasi_kontribusi_n30.py` | Recomputes every contribution claim reported in the manuscript directly from the results CSV, with explicit guardrails distinguishing the one pre-specified confirmatory comparison (0-degree vs. 20-degree, unmodified render) from the other exploratory comparisons in Table 4 |
| `manifes_reproduksibilitas.py` | Regenerates `requirements-pinned.txt` and the parameter manifest from the live environment |

### 4. `analysis/` — replication studies and the visibility measure

| Script | Purpose |
|---|---|
| `evaluasi_seed_n30.py` / `analisis_seed_n30.py` | Three-seed replication of the primary 0-vs-20-degree comparison (Table 3). The analysis script carries its own gates, pre-specified tests, and automatic prediction checks; it refuses to report anything if an arm is incomplete |
| `evaluasi_seed_tabel3.py` / `analisis_seed_tabel3.py` | Second-seed replication of the only positive result in Table 4 (perspective projection at 30 and 40 degrees) and of the single gray-background difference at 40 degrees |
| `evaluasi_uji_mc.py` | Evaluates both extractors' meshes with `evaluation/evaluate_mesh.py` and runs the pre-specified extractor analysis (`--analisis`): determinism gate, topology per extractor, geometry difference, elevation effect under each extractor, and automatic prediction checks |
| `uji_sintetis_surfnet_vs_mc.py` | Synthetic pre-test: ComfyUI's surface net vs marching cubes on clean analytic shapes. ComfyUI (GPL-3.0) is not redistributed here; the script reads `voxel_to_mesh_surfnet` from your ComfyUI install or from a pinned upstream commit at run time. This is the one script adapted after it was run, to avoid bundling GPL code; its output is unchanged |
| `analisis_fitur_objek.py` | Per-object reference-mesh features in `results/fitur_objek.csv` (dimensions, topology, facing and visible surface shares), computed from the reference meshes alone |
| `hitung_visibilitas.py` | Surface-visibility measure: 200,000 area-weighted sample points per reference mesh, orthographic depth buffer, best viewing angle per point. Prints a consistency check against `results/fitur_objek.csv` and stops if it disagrees |
| `plot_peta_histogram_visibilitas.py` | The two surface-visibility figures (visibility map; visibility gain vs Chamfer improvement) plus `results/ringkasan_uji_visibilitas.txt`, which logs every test tried, including the ones that failed |
| `susun_galeri_pendorong.py` | Assembles the driver-component gallery figure from the Blender renders and the three-seed per-object table |

### 5. `results/` — the tables behind the numbers

Every per-object table and analysis log the manuscript relies on, so that any reported value can be recomputed without rerunning generation. `hasil_uji_mc_23sep.csv`, `analisis_uji_mc_23sep.txt`, and `uji_mc_runlog.csv` hold the extractor comparison. `visibilitas_per_objek.csv` holds the raw visibility percentages that Section 4.2 quotes to one decimal place but the visibility-map figure can only show as integers.

### 6. `photographs/` — smartphone-photograph demonstration (Section 4.4)

The four 1024 x 1024 inputs with and without background, the cropping and BiRefNet background-removal scripts, the Blender clay renders, and the figure builder. See `photographs/README.md` for the photographs used, the model and its checksum, package versions, and a note on provenance.

### Paths

The code of every script here is the code that produced the results. For publication, only module headers, comments, and printed messages were rewritten (internal notes and local paths removed, English headers added); a check of each file's syntax tree against the original confirms that code, numeric constants, and control flow are unchanged. Each script resolves its inputs relative to the working project layout it ran in, where `eval/` holds the scripts and result tables, `results/` holds the generated meshes, `dataset/` holds the reference CAD, and `figur/` holds the figures. In this repository those tables live under `results/` and the scripts are split across `evaluation/` and `analysis/`, so the paths at the top of each script need adjusting before it will run. We chose not to rewrite these paths, so that the code published is exactly the code that was run. The one exception is `photographs/potong_dan_hapus_latar.py`, written after the steps were first run interactively and checked to reproduce the committed inputs pixel for pixel; it takes its paths as arguments.

## Requirements

See `requirements-pinned.txt` for the exact evaluation-environment package versions. `trimesh` must be pinned to the listed version: `volume_iou` is not bit-reproducible across trimesh versions (see `SUPPLEMENTARY_MATERIAL.md` Section 4).

## What is intentionally not included

Earlier, superseded iterations of some scripts (e.g., three prior perspective-projection field-of-view formulas, and a rejected lighting-calibration attempt for the material variant) are not included here; only the version actually used to produce the results in the manuscript is. The full development history remains in the authors' private working notes.
