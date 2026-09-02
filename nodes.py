"""Render a rigged character (FBX/GLB) in a random pose to an IMAGE.

The heavy lifting happens in render_worker.py, which is executed by the python
that owns `bpy` — the ComfyUI-UniRig isolated environment. This node only
marshals arguments, runs that subprocess, and converts the PNG to a tensor.
"""
import json
import os
import subprocess
import tempfile
import glob

import numpy as np
import torch
from PIL import Image

import folder_paths

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "render_worker.py")


def _candidate_pythons():
    """Every place a bpy-capable python is likely to live, best guess first."""
    env_override = os.environ.get("POSE_RENDER_PYTHON")
    if env_override:
        yield env_override
    roots = [
        os.environ.get("COMFY_ENV_ROOT"),
        r"D:\comfy-env",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "comfy-env"),
        os.path.expanduser("~/.local/share/comfy-env"),
    ]
    for root in roots:
        if not root:
            continue
        for pat in ("envs/*/.pixi/envs/default/python.exe",
                    "envs/*/.pixi/envs/default/bin/python"):
            for hit in sorted(glob.glob(os.path.join(root, pat))):
                yield hit


def _find_python():
    for p in _candidate_pythons():
        if p and os.path.isfile(p):
            return p
    raise RuntimeError(
        "No bpy-capable python found. Install ComfyUI-UniRig (it provisions one), "
        "or set the POSE_RENDER_PYTHON environment variable to a python that has bpy."
    )


def _resolve_mesh(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in (folder_paths.get_output_directory(), folder_paths.get_input_directory()):
        for cand in (os.path.join(base, path), os.path.join(base, "3d", path)):
            if os.path.exists(cand):
                return cand
    raise RuntimeError("Mesh file not found: %s" % path)


class PoseRandomRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Rigged FBX/GLB path (wire MIA: Auto Rig's fbx_output_path here)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                 "control_after_generate": True}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                            "tooltip": "0 = rest pose, 1 = normal random pose."}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "samples": ("INT", {"default": 96, "min": 8, "max": 1024,
                                    "tooltip": "Cycles CPU samples. 96 is a good speed/noise trade."}),
                "azimuth": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "elevation": ("FLOAT", {"default": 0.0, "min": -89.0, "max": 89.0, "step": 1.0}),
                "zoom": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 4.0, "step": 0.05,
                                   "tooltip": "Ortho frame size relative to the subject."}),
                "bg_color": ("STRING", {"default": "230, 230, 230"}),
                "transparent": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "render"
    CATEGORY = "UniRig/Render"
    DESCRIPTION = ("Import a rigged FBX/GLB, apply a seeded random humanoid pose, and render it "
                   "with Blender (Cycles CPU) to an IMAGE you can save with SaveImage.")

    def render(self, mesh_path, seed, pose_strength, width, height, samples,
               azimuth, elevation, zoom, bg_color, transparent):
        mesh = _resolve_mesh(mesh_path.strip())
        python = _find_python()
        tmpdir = tempfile.mkdtemp(prefix="pose_render_")
        out_png = os.path.join(tmpdir, "render.png")
        cfg_path = os.path.join(tmpdir, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "mesh_path": mesh, "out_path": out_png, "seed": int(seed),
                "pose_strength": float(pose_strength), "width": int(width),
                "height": int(height), "samples": int(samples),
                "azimuth": float(azimuth), "elevation": float(elevation),
                "zoom": float(zoom), "bg_color": bg_color,
                "transparent": bool(transparent),
            }, f)

        proc = subprocess.run([python, WORKER, "--", cfg_path],
                              capture_output=True, text=True, cwd=tmpdir)
        produced = out_png if os.path.exists(out_png) else None
        if produced is None:
            # Blender appends the frame number when the path has no extension quirk
            hits = sorted(glob.glob(os.path.join(tmpdir, "render*.png")))
            produced = hits[0] if hits else None
        if produced is None:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            raise RuntimeError("Blender render produced no image (exit %s):\n%s"
                               % (proc.returncode, "\n".join(tail)))

        img = Image.open(produced)
        img.load()
        has_alpha = img.mode in ("RGBA", "LA")
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        image = torch.from_numpy(rgba[:, :, :3])[None, ...]
        alpha = rgba[:, :, 3] if has_alpha else np.ones(rgba.shape[:2], dtype=np.float32)
        mask = torch.from_numpy(alpha)[None, ...]
        return (image, mask)


NODE_CLASS_MAPPINGS = {"PoseRandomRender": PoseRandomRender}
NODE_DISPLAY_NAME_MAPPINGS = {"PoseRandomRender": "Pose Render (random pose -> IMAGE)"}
