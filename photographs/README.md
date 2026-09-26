# Smartphone-photograph demonstration (manuscript Section 4.4)

A qualitative check of the protocol on a physical part outside the 30-component dataset: a used Shimizu PS-128 BIT
water pump, photographed with a tripod-mounted smartphone about 0.5 m away from the four rig directions
(front = fan cover, left, back = brass impeller cover, right) at about 20 degrees elevation. There is no reference
geometry for this unit, so no metric is computed; the demonstration does not test the elevation effect.

| File | Purpose |
|---|---|
| `potong_dan_hapus_latar.py` | Photograph -> common square crop (one scale for all four views) -> BiRefNet background removal -> white-background four-view input. Header documents every step, the memory note, and the pixel-for-pixel reproduction check |
| `segmentasi_birefnet_lite.py` | Runs BiRefNet-general-lite (ONNX) on one image and writes the soft mask (ImageNet normalization, 1024 x 1024 input, CPU) |
| `inputs/latar_asli/{front,left,back,right}.png` | The 1024 x 1024 crops with the real background (Figure 8a uses `left.png`) |
| `inputs/latar_putih/{front,left,back,right}.png` | The background-removed crops fed to Hunyuan3D-2mv (Figure 8b-c) |
| `render_pompa_clay.py` | Blender clay render of a reconstruction, same material, lights and orthographic camera as the dataset renders |
| `render_pompa_clay_transparan.py` | Same, with a transparent film so the object can be cropped tightly |
| `susun_figure_pompa_v2.py` | Assembles Figure 8 (photograph -> background removed -> four-view input -> reconstruction from four sides) |

Photographs chosen: IMG_4013 (front), IMG_4011 (left), IMG_4012 (back), IMG_4010 (right), converted from HEIC to PNG
with pillow-heif (EXIF orientation applied). The full-resolution originals are not included because smartphone
photographs carry location metadata; the eight crops above are the exact inputs and contain no metadata.

Generation used the unchanged ComfyUI workflow of the main experiment (`generation/workflow/`), seed 123 (shown in
Figure 8) and seed 42 (similar shape). The reconstructed meshes are not redistributed, as for the main experiment.

A first run on unprocessed photographs (an earlier set taken at about 0 degrees, fed to the same workflow without
background removal) reconstructed the ledge under the pump as a platform; this is the observation Section 4.4 reports.
The platform came from the uncleaned background, not from the elevation: the same 0-degree photographs gave no
platform once BiRefNet had removed the background. The 20-degree set was chosen as the demonstration before it was
generated, and the 0-degree results were kept as an archive.

## Model and versions

- BiRefNet-general-lite, ONNX export distributed with rembg:
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx
  (SHA-256 5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333). Not redistributed here.
  BiRefNet and rembg are MIT-licensed; cite P. Zheng et al., CAAI Artif. Intell. Res., vol. 3, art. 9150038, 2024.
- onnxruntime 1.25.0 (CPU), numpy 1.26.4, scipy 1.17.1, pillow 12.2.0, pillow-heif 1.8.0, Blender Python module bpy 5.0.1.

Unlike the rest of the repository, `potong_dan_hapus_latar.py` is not the file that first produced the crops: the
steps were first run interactively and then written into this script, which reproduces the committed inputs pixel
for pixel (see its header). Paths are given as command-line arguments.
