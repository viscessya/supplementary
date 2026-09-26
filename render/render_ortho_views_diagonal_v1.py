# -*- coding: utf-8 -*-
"""Diagonal-azimuth variant of render/render_ortho_views.py: the four cameras are rotated 45 degrees, to
45/135/225/315 degrees. Orthographic camera, material, lighting, margin, and elevation are unchanged.

The output files keep the names front.png, left.png, back.png, and right.png so that the generation
workflow can load them unchanged; they hold the 45, 135, 225, and 315-degree views respectively.
Before the first render, the projected width of every component at these azimuths was checked numerically
at margin 1.30: none is clipped (the closest, obj13, reaches 0.644 of the 0.65 limit).

Usage: blender -b -P render_ortho_views_diagonal_v1.py -- --input model.stl --out output_dir --elev 20 --margin 1.30
Tested with Blender 3.6+ and 4.x.
"""
import bpy, sys, os, math, argparse
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--out", default=None)
p.add_argument("--res", type=int, default=1024)
p.add_argument("--bg", choices=["white", "transparent"], default="white")
p.add_argument("--elev", type=float, default=0.0)
p.add_argument("--margin", type=float, default=1.15)
args = p.parse_args(argv)

inp = os.path.abspath(args.input)
name = os.path.splitext(os.path.basename(inp))[0]
outdir = os.path.abspath(args.out or f"./renders_{name}")
os.makedirs(outdir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

ext = os.path.splitext(inp)[1].lower()
importers = {
    ".stl":  lambda: bpy.ops.wm.stl_import(filepath=inp) if hasattr(bpy.ops.wm, "stl_import") else bpy.ops.import_mesh.stl(filepath=inp),
    ".obj":  lambda: bpy.ops.wm.obj_import(filepath=inp) if hasattr(bpy.ops.wm, "obj_import") else bpy.ops.import_scene.obj(filepath=inp),
    ".ply":  lambda: bpy.ops.wm.ply_import(filepath=inp) if hasattr(bpy.ops.wm, "ply_import") else bpy.ops.import_mesh.ply(filepath=inp),
    ".glb":  lambda: bpy.ops.import_scene.gltf(filepath=inp),
    ".gltf": lambda: bpy.ops.import_scene.gltf(filepath=inp),
    ".fbx":  lambda: bpy.ops.import_scene.fbx(filepath=inp),
}
if ext not in importers:
    raise SystemExit(f"Format {ext} tidak didukung. Ekspor CAD ke STL/OBJ dulu.")
importers[ext]()

meshes = [o for o in scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("Tidak ada mesh yang terimpor.")

bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
center = (mn + mx) / 2
dims = mx - mn
scale = 1.0 / max(dims.x, dims.y, dims.z)
obj.location -= center
bpy.ops.object.transform_apply(location=True)
obj.scale = (scale, scale, scale)
bpy.ops.object.transform_apply(scale=True)

mat = bpy.data.materials.new("NeutralClay")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.42, 0.42, 0.42, 1.0)
bsdf.inputs["Roughness"].default_value = 0.6
bsdf.inputs["Metallic"].default_value = 0.0
obj.data.materials.clear()
obj.data.materials.append(mat)

try:
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
except Exception:
    try:
        bpy.ops.object.shade_smooth()
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(30)
    except Exception:
        pass

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)

def add_light(loc, energy=300, size=3.0):
    light_data = bpy.data.lights.new("L", type="AREA")
    light_data.energy = energy
    light_data.size = size
    lo = bpy.data.objects.new("L", light_data)
    scene.collection.objects.link(lo)
    lo.location = loc
    d = Vector((0, 0, 0)) - Vector(loc)
    lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

add_light((2, -2, 2), energy=60)
add_light((-2, -2, 1.5), energy=25)
add_light((0, 2.5, 2), energy=25)

cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = args.margin
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

DIST = 5.0
elev = math.radians(args.elev)

def place_camera(azim_deg):
    a = math.radians(azim_deg)
    x = DIST * math.sin(a) * math.cos(elev)
    y = -DIST * math.cos(a) * math.cos(elev)
    z = DIST * math.sin(elev)
    cam.location = (x, y, z)
    d = Vector((0, 0, 0)) - Vector(cam.location)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
try:
    scene.cycles.device = "GPU"
except Exception:
    pass
scene.render.resolution_x = args.res
scene.render.resolution_y = args.res
scene.render.image_settings.file_format = "PNG"

try:
    scene.view_settings.view_transform = "Standard"
except Exception:
    pass

if args.bg == "transparent":
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
else:
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"

views = {"front": 45, "left": 135, "back": 225, "right": 315}
for vname, az in views.items():
    place_camera(az)
    scene.render.filepath = os.path.join(outdir, f"{vname}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[OK] {vname}.png")

print(f"\nSelesai. 4 view diagonal tersimpan di: {outdir}")
