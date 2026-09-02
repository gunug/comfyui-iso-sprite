"""Isometric 8-direction sprite-atlas renderer for an animated rigged FBX/GLB.

Same execution model as PoseRandomRender: the bpy work happens in
iso_sprite_worker.py, run by the python that owns `bpy` (the ComfyUI-UniRig
isolated environment). This node marshals arguments, runs that subprocess, and
packs the PNGs into an IMAGE batch plus a single atlas sheet.
"""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image

import folder_paths

from .nodes import _find_python, _resolve_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "iso_sprite_worker.py")


def _to_rgba(png_path):
    img = Image.open(png_path)
    img.load()
    return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0


class IsoSpriteSheetRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Animated rigged FBX/GLB path (wire UniRig: Apply Animation's animated_fbx_path here)."}),
                "directions": ("INT", {"default": 8, "min": 1, "max": 32,
                                       "tooltip": "8 = classic isometric 8-way. Cameras are spread evenly over 360 degrees."}),
                "frames": ("INT", {"default": 8, "min": 1, "max": 64,
                                   "tooltip": "Animation frames sampled evenly across the clip. Total renders = directions x frames."}),
                "cell_width": ("INT", {"default": 256, "min": 32, "max": 1024, "step": 8}),
                "cell_height": ("INT", {"default": 256, "min": 32, "max": 1024, "step": 8}),
                "elevation": ("FLOAT", {"default": 30.0, "min": -89.0, "max": 89.0, "step": 0.5,
                                        "tooltip": "Camera pitch. 30 = game isometric, 35.264 = true dimetric/isometric."}),
                "start_azimuth": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                                            "tooltip": "Yaw of direction 0. 0 = character faces the camera (south)."}),
                "clockwise": ("BOOLEAN", {"default": True,
                                          "tooltip": "Direction order S, SW, W, NW, N, NE, E, SE when on."}),
                "zoom": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 4.0, "step": 0.05,
                                   "tooltip": "Ortho frame size relative to the subject. One camera size for the whole clip."}),
                "engine": (["CYCLES", "BLENDER_EEVEE_NEXT"], {"default": "CYCLES"}),
                "samples": ("INT", {"default": 24, "min": 4, "max": 512,
                                    "tooltip": "Cycles CPU samples per cell. Keep low, this runs directions x frames times."}),
                "transparent": ("BOOLEAN", {"default": True}),
                "bg_color": ("STRING", {"default": "230, 230, 230"}),
                "layout": (["rows=direction", "rows=frame"], {"default": "rows=direction"}),
                "filename_prefix": ("STRING", {"default": "iso_sprite_atlas"}),
                "save_atlas": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "frame_start": ("INT", {"default": -1, "min": -1, "max": 100000,
                                        "tooltip": "-1 = use the clip's own range."}),
                "frame_end": ("INT", {"default": -1, "min": -1, "max": 100000,
                                      "tooltip": "-1 = use the clip's own range."}),
                "include_last": ("BOOLEAN", {"default": False,
                                             "tooltip": "Off = drop the duplicate final frame so a looping cycle tiles cleanly."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "masks", "atlas", "atlas_path", "report")
    FUNCTION = "render"
    CATEGORY = "UniRig/Render"
    DESCRIPTION = ("Render an animated rigged FBX/GLB from N evenly spaced isometric camera angles "
                   "(8 by default) across the whole animation, and pack the result into a sprite "
                   "atlas. images is the flat batch in direction-major order; atlas is the sheet.")

    def render(self, mesh_path, directions, frames, cell_width, cell_height, elevation,
               start_azimuth, clockwise, zoom, engine, samples, transparent, bg_color,
               layout, filename_prefix, save_atlas,
               frame_start=-1, frame_end=-1, include_last=False):
        mesh = _resolve_mesh(mesh_path.strip())
        python = _find_python()
        tmpdir = tempfile.mkdtemp(prefix="iso_sprite_")
        try:
            out_dir = os.path.join(tmpdir, "cells")
            cfg_path = os.path.join(tmpdir, "cfg.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "mesh_path": mesh, "out_dir": out_dir,
                    "directions": int(directions), "frames": int(frames),
                    "cell_width": int(cell_width), "cell_height": int(cell_height),
                    "elevation": float(elevation), "start_azimuth": float(start_azimuth),
                    "clockwise": bool(clockwise), "zoom": float(zoom),
                    "engine": engine, "samples": int(samples),
                    "transparent": bool(transparent), "bg_color": bg_color,
                    "frame_start": int(frame_start), "frame_end": int(frame_end),
                    "include_last": bool(include_last),
                }, f)

            proc = subprocess.run([python, WORKER, "--", cfg_path],
                                  capture_output=True, text=True, cwd=tmpdir)

            cells = []
            for di in range(int(directions)):
                row = []
                for fi in range(int(frames)):
                    p = os.path.join(out_dir, "d%02d_f%03d.png" % (di, fi))
                    if not os.path.exists(p):
                        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
                        raise RuntimeError(
                            "Blender produced no sprite cell d%02d_f%03d (exit %s):\n%s"
                            % (di, fi, proc.returncode, "\n".join(tail)))
                    row.append(_to_rgba(p))
                cells.append(row)

            info = {}
            for line in (proc.stdout or "").splitlines():
                if line.startswith("ISO_SPRITE_OK "):
                    info = json.loads(line[len("ISO_SPRITE_OK "):])

            flat = [c for row in cells for c in row]
            images = torch.from_numpy(np.stack([c[:, :, :3] for c in flat], axis=0))
            masks = torch.from_numpy(np.stack([c[:, :, 3] for c in flat], axis=0))

            h, w = flat[0].shape[:2]
            if layout == "rows=direction":
                rows, cols = int(directions), int(frames)
                grid = cells
            else:
                rows, cols = int(frames), int(directions)
                grid = [[cells[di][fi] for di in range(int(directions))]
                        for fi in range(int(frames))]
            sheet = np.zeros((rows * h, cols * w, 4), dtype=np.float32)
            for r, row in enumerate(grid):
                for c, cell in enumerate(row):
                    sheet[r * h:(r + 1) * h, c * w:(c + 1) * w, :] = cell
            atlas = torch.from_numpy(sheet[:, :, :3])[None, ...]

            atlas_path = ""
            if save_atlas:
                full, base, counter, subfolder, _ = folder_paths.get_save_image_path(
                    filename_prefix, folder_paths.get_output_directory(),
                    cols * w, rows * h)
                atlas_path = os.path.join(full, "%s_%05d_.png" % (base, counter))
                mode = "RGBA" if transparent else "RGB"
                arr = (np.clip(sheet, 0.0, 1.0) * 255.0).astype(np.uint8)
                Image.fromarray(arr, "RGBA").convert(mode).save(atlas_path)

            report = json.dumps({
                "cells": len(flat), "directions": int(directions), "frames": int(frames),
                "cell": [w, h], "sheet": [cols * w, rows * h], "layout": layout,
                "elevation": float(elevation), "start_azimuth": float(start_azimuth),
                "clockwise": bool(clockwise), "engine": info.get("engine", engine),
                "clip_range": [info.get("frame_start"), info.get("frame_end")],
                "sampled_frames": info.get("frame_ids"),
                "atlas_path": atlas_path,
            }, ensure_ascii=False, indent=2)
            return (images, masks, atlas, atlas_path, report)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {"IsoSpriteSheetRender": IsoSpriteSheetRender}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IsoSpriteSheetRender": "Iso Sprite Atlas Render (animation -> 8-dir atlas)"
}
