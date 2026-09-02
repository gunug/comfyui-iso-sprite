"""Standalone bpy worker: import a rigged FBX/GLB, apply a random bone pose, render a PNG.

Runs in the ComfyUI-UniRig isolated environment (the one that owns `bpy`), NOT in
ComfyUI's python. Arguments arrive as a single JSON file path.
"""
import bpy, sys, json, math, random
from mathutils import Vector, Euler

cfg = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))

bpy.ops.wm.read_factory_settings(use_empty=True)
path = cfg["mesh_path"]
low = path.lower()
if low.endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=path)
elif low.endswith((".glb", ".gltf")):
    bpy.ops.import_scene.gltf(filepath=path)
elif low.endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=path)
else:
    raise RuntimeError("unsupported mesh format: " + path)

sc = bpy.context.scene
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]

# --- pose -----------------------------------------------------------------
# degrees, per bone: (x_range, y_range, z_range) in bone-local space
PLAN = {
    "Spine":        ((-6, 10), (-8, 8), (-8, 8)),
    "Spine1":       ((-6, 10), (-8, 8), (-8, 8)),
    "Neck":         ((-8, 8), (-10, 10), (-8, 8)),
    "Head":         ((-10, 10), (-18, 18), (-10, 10)),
    "LeftShoulder": ((-8, 8), (-8, 8), (-10, 10)),
    "RightShoulder": ((-8, 8), (-8, 8), (-10, 10)),
    "LeftArm":      ((-25, 25), (-18, 18), (-35, 15)),
    "RightArm":     ((-25, 25), (-18, 18), (-15, 35)),
    "LeftForeArm":  ((-5, 5), (-10, 10), (-45, -10)),
    "RightForeArm": ((-5, 5), (-10, 10), (10, 45)),
    "LeftUpLeg":    ((-25, 20), (-8, 8), (-12, 12)),
    "RightUpLeg":   ((-25, 20), (-8, 8), (-12, 12)),
    "LeftLeg":      ((0, 40), (-4, 4), (-4, 4)),
    "RightLeg":     ((0, 40), (-4, 4), (-4, 4)),
}
posed = []
if arms and cfg["pose_strength"] > 0.0:
    arm = arms[0]
    if arm.animation_data:
        arm.animation_data.action = None
    rng = random.Random(cfg["seed"])
    k = cfg["pose_strength"]
    by_suffix = {}
    for pb in arm.pose.bones:
        by_suffix.setdefault(pb.name.split(":")[-1], pb)
    for name, ranges in PLAN.items():
        pb = by_suffix.get(name)
        if pb is None:
            continue
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = Euler(
            tuple(math.radians(rng.uniform(lo, hi) * k) for lo, hi in ranges), "XYZ"
        )
        posed.append(pb.name)
    bpy.context.view_layer.update()

# --- frame the subject -----------------------------------------------------
dg = bpy.context.evaluated_depsgraph_get()
mins = Vector((1e9, 1e9, 1e9))
maxs = Vector((-1e9, -1e9, -1e9))
for ob in sc.objects:
    if ob.type != "MESH":
        continue
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    for v in me.vertices:
        w = ev.matrix_world @ v.co
        for i in range(3):
            mins[i] = min(mins[i], w[i])
            maxs[i] = max(maxs[i], w[i])
    ev.to_mesh_clear()
if mins.x > maxs.x:
    raise RuntimeError("no mesh geometry found in " + path)
center = (mins + maxs) / 2.0
size = max(maxs[i] - mins[i] for i in range(3))

az = math.radians(cfg["azimuth"])
el = math.radians(cfg["elevation"])
d = size * 3.0
offset = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * d
cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
cd.type = "ORTHO"
cd.ortho_scale = size * cfg["zoom"]
cam.location = center + offset
cam.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()

for loc, energy in (((2, -2, 2), 3.0), ((-2, -2, 1), 1.5), ((0, 2, 1), 1.2)):
    ld = bpy.data.lights.new("l", "SUN")
    ld.energy = energy
    lo = bpy.data.objects.new("l", ld)
    sc.collection.objects.link(lo)
    lo.location = center + Vector(loc) * size
    lo.rotation_euler = (center - lo.location).normalized().to_track_quat("-Z", "Y").to_euler()

world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
bg = [float(c) / 255.0 for c in cfg["bg_color"].split(",")][:3]
world.node_tree.nodes["Background"].inputs[0].default_value = (bg + [1.0])
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = cfg["samples"]
sc.cycles.use_denoising = False          # OIDN is unavailable in the bundled bpy
sc.render.resolution_x = cfg["width"]
sc.render.resolution_y = cfg["height"]
sc.render.film_transparent = cfg["transparent"]
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA" if cfg["transparent"] else "RGB"
sc.render.filepath = cfg["out_path"]
bpy.ops.render.render(write_still=True)

print("POSE_RENDER_OK " + json.dumps({"posed_bones": posed, "size": size}), flush=True)
