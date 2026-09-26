"""Same as render_pompa_clay.py, except that the film is transparent (sc.render.film_transparent = True),
so that susun_figure_pompa_v2.py can crop each render tightly by its alpha channel.

Usage (Blender or the bpy module):
    python render_pompa_clay_transparan.py -- --in mesh.glb --out-dir render/ --views 45:25 135:25 --res 900 --samples 96
"""
import bpy, sys, os, math, argparse
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--views", nargs="+", default=["225:25", "315:25", "45:25", "135:25"], help="azimut:elevasi (derajat)")
p.add_argument("--res", type=int, default=1024)
p.add_argument("--margin", type=float, default=1.15)
p.add_argument("--samples", type=int, default=64)
a = p.parse_args(argv)
os.makedirs(a.out_dir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(a.inp)[1].lower()
if ext in (".glb", ".gltf"): bpy.ops.import_scene.gltf(filepath=a.inp)
else: bpy.ops.wm.stl_import(filepath=a.inp)
sc = bpy.context.scene
meshes = [o for o in sc.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
ob.location -= (mn + mx) / 2; bpy.ops.object.transform_apply(location=True)
s = 1.0 / max(mx - mn); ob.scale = (s, s, s); bpy.ops.object.transform_apply(scale=True)
diag = ((mx - mn) * s).length
mat = bpy.data.materials.new("NeutralClay"); mat.use_nodes = True
b = mat.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.42, 0.42, 0.42, 1.0); b.inputs["Roughness"].default_value = 0.6; b.inputs["Metallic"].default_value = 0.0
ob.data.materials.clear(); ob.data.materials.append(mat)
try: bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
except Exception: pass
w = bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
w.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
def light(loc, e):
    L = bpy.data.lights.new("L", "AREA"); L.energy = e; L.size = 3.0
    lo = bpy.data.objects.new("L", L); sc.collection.objects.link(lo); lo.location = loc
    lo.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
light((2, -2, 2), 60); light((-2, -2, 1.5), 25); light((0, 2.5, 2), 25)
cd = bpy.data.cameras.new("Cam"); cd.type = "ORTHO"; cd.ortho_scale = diag * a.margin
cam = bpy.data.objects.new("Cam", cd); sc.collection.objects.link(cam); sc.camera = cam
sc.render.engine = "CYCLES"; sc.cycles.samples = a.samples; sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.view_settings.view_transform = "Standard"; sc.render.film_transparent = True
for v in a.views:
    az, el = [math.radians(float(x)) for x in v.split(":")]
    D = 5.0
    cam.location = (D * math.sin(az) * math.cos(el), -D * math.cos(az) * math.cos(el), D * math.sin(el))
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(a.out_dir, f"pompa_az{v.split(':')[0]}_el{v.split(':')[1]}.png")
    bpy.ops.render.render(write_still=True); print("OK", sc.render.filepath)
