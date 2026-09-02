"""Standalone bpy worker: read an animated FBX/GLB's clip range and frame rate.

No render — this only imports the file and reports its timing so the node can
size a sprite sheet to the actual animation length.
"""
import bpy, sys, json

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

starts, ends, names = [], [], []
for act in bpy.data.actions:
    r = act.frame_range
    starts.append(r[0])
    ends.append(r[1])
    names.append(act.name)

if starts and max(ends) > min(starts):
    f_start, f_end = int(round(min(starts))), int(round(max(ends)))
else:
    f_start, f_end = sc.frame_start, sc.frame_end

src_fps = float(sc.render.fps) / float(sc.render.fps_base or 1.0)

print("CLIP_INFO_OK " + json.dumps({
    "frame_start": f_start,
    "frame_end": f_end,
    "length": (f_end - f_start) + 1,
    "scene_fps": src_fps,
    "actions": names,
}), flush=True)
