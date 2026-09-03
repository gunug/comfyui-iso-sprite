"""Standalone bpy worker: import an ANIMATED FBX/GLB and render it as an
isometric 8-direction sprite sequence (one PNG per direction per frame).

Runs in the ComfyUI-UniRig isolated environment (the one that owns `bpy`), NOT in
ComfyUI's python. Arguments arrive as a single JSON file path.
"""
import bpy, sys, json, math, os
from mathutils import Vector

cfg = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))

bpy.ops.wm.read_factory_settings(use_empty=True)
path = cfg["mesh_path"]
low = path.lower()
if low.endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=path)
elif low.endswith((".glb", ".gltf")):
    bpy.ops.import_scene.gltf(filepath=path)
else:
    raise RuntimeError("unsupported mesh format: " + path)

sc = bpy.context.scene

# --- animation range -------------------------------------------------------
# Prefer the imported action's own range; fall back to the scene range.
starts, ends = [], []
for ob in bpy.data.objects:
    ad = ob.animation_data
    if ad and ad.action:
        r = ad.action.frame_range
        starts.append(r[0])
        ends.append(r[1])
for act in bpy.data.actions:
    r = act.frame_range
    starts.append(r[0])
    ends.append(r[1])

if starts and max(ends) > min(starts):
    f_start, f_end = int(round(min(starts))), int(round(max(ends)))
else:
    f_start, f_end = sc.frame_start, sc.frame_end

if cfg.get("frame_start", -1) >= 0:
    f_start = int(cfg["frame_start"])
if cfg.get("frame_end", -1) >= 0:
    f_end = int(cfg["frame_end"])
if f_end < f_start:
    f_end = f_start

n_frames = max(1, int(cfg["frames"]))
if n_frames == 1 or f_end == f_start:
    frame_ids = [f_start] * n_frames
else:
    if cfg["include_last"] or n_frames == 1:
        denom = max(1, n_frames - 1)
        frame_ids = [f_start + int(round(i * (f_end - f_start) / float(denom)))
                     for i in range(n_frames)]
    else:
        # Loop-safe: stop one step short of the end so frame[0] and the frame
        # after the last do not duplicate the same pose in a cycling clip.
        span = (f_end - f_start) + 1
        frame_ids = [f_start + int(round(i * span / float(n_frames))) for i in range(n_frames)]
    frame_ids = [min(max(f, f_start), f_end) for f in frame_ids]


def world_bounds():
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
    return mins, maxs


# --- framing: union bbox over EVERY sampled frame --------------------------
# One fixed camera per direction, sized to the whole clip, so the character does
# not swim or rescale between frames of the sheet.
gmins = Vector((1e9, 1e9, 1e9))
gmaxs = Vector((-1e9, -1e9, -1e9))
for f in sorted(set(frame_ids)):
    sc.frame_set(f)
    bpy.context.view_layer.update()
    mins, maxs = world_bounds()
    if mins.x > maxs.x:
        continue
    for i in range(3):
        gmins[i] = min(gmins[i], mins[i])
        gmaxs[i] = max(gmaxs[i], maxs[i])
if gmins.x > gmaxs.x:
    raise RuntimeError("no mesh geometry found in " + path)

center = (gmins + gmaxs) / 2.0
size = max(gmaxs[i] - gmins[i] for i in range(3))
ground_z = gmins.z

# --- world / lights / render settings --------------------------------------
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

engine = cfg.get("engine", "CYCLES")
try:
    sc.render.engine = engine
except TypeError:
    sc.render.engine = "CYCLES"
    engine = "CYCLES"
if sc.render.engine == "CYCLES":
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(cfg["samples"])
    sc.cycles.use_denoising = False      # OIDN is unavailable in the bundled bpy

sc.render.resolution_x = int(cfg["cell_width"])
sc.render.resolution_y = int(cfg["cell_height"])
sc.render.resolution_percentage = 100
sc.render.film_transparent = bool(cfg["transparent"])
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA" if cfg["transparent"] else "RGB"

cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
cd.type = "ORTHO"
cd.ortho_scale = size * float(cfg["zoom"])

el = math.radians(float(cfg["elevation"]))
d = size * 3.0
out_dir = cfg["out_dir"]
os.makedirs(out_dir, exist_ok=True)

dirs = int(cfg["directions"])
step = 360.0 / dirs


def render_all(prefix):
    """Render every direction x frame cell, named <prefix>dNN_fNNN.png."""
    out = []
    for di in range(dirs):
        az = math.radians(float(cfg["start_azimuth"]) + di * step * (-1.0 if cfg["clockwise"] else 1.0))
        offset = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * d
        cam.location = center + offset
        cam.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()
        for fi, f in enumerate(frame_ids):
            sc.frame_set(f)
            bpy.context.view_layer.update()
            fp = os.path.join(out_dir, "%sd%02d_f%03d.png" % (prefix, di, fi))
            sc.render.filepath = fp
            bpy.ops.render.render(write_still=True)
            out.append(fp)
    return out


def make_view_normal_material():
    """Emission shader that outputs the CAMERA-space shading normal, encoded
    n * 0.5 + 0.5 (OpenGL convention: R=+X right, G=+Y up, B=+Z toward viewer).
    Applied as a view-layer material override, so it is exact geometry, not an
    image-space estimate."""
    m = bpy.data.materials.new("iso_view_normal")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    vt = nt.nodes.new("ShaderNodeVectorTransform")
    vt.vector_type = "NORMAL"
    vt.convert_from = "WORLD"
    vt.convert_to = "CAMERA"
    mul = nt.nodes.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    # Blender's camera space looks down -Z, so a surface facing the camera has
    # z = -1 there. Negate Z on the way in so the encoded blue is 1.0 for
    # camera-facing geometry, which is what OpenGL consumers expect.
    mul.inputs[1].default_value = (0.5, 0.5, -0.5)
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    emis.inputs["Strength"].default_value = 1.0
    nt.links.new(geo.outputs["Normal"], vt.inputs[0])
    nt.links.new(vt.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], add.inputs[0])
    nt.links.new(add.outputs[0], emis.inputs["Color"])
    nt.links.new(emis.outputs[0], out.inputs["Surface"])
    return m


written = render_all("")

normals_written = []
if cfg.get("render_normals", False):
    vl = bpy.context.view_layer
    vl.material_override = make_view_normal_material()
    # The encoded normal must survive untouched: no view transform, no film
    # curve, always straight alpha so the sheet keeps the same cutout.
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    # "Raw" writes the linear value straight to the file. "Standard" would apply
    # the sRGB curve and silently gamma-corrupt the encoded normal.
    for tf in ("Raw", "Standard"):
        try:
            sc.view_settings.view_transform = tf
            break
        except TypeError:
            continue
    try:
        sc.view_settings.look = "None"
        sc.view_settings.exposure = 0.0
        sc.view_settings.gamma = 1.0
        sc.render.dither_intensity = 0.0
    except Exception:
        pass
    if sc.render.engine == "CYCLES":
        # Pure emission converges instantly; samples only buy edge AA.
        sc.cycles.samples = max(4, min(int(cfg["samples"]), 16))
    normals_written = render_all("n_")
    vl.material_override = None

print("ISO_SPRITE_OK " + json.dumps({
    "size": size, "ground_z": ground_z, "engine": engine,
    "frame_start": f_start, "frame_end": f_end, "frame_ids": frame_ids,
    "directions": dirs, "count": len(written),
    "normal_count": len(normals_written),
}), flush=True)
