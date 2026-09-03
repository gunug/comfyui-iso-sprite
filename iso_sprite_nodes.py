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


def _to_rgba(png_path, size=None, is_normal=False):
    img = Image.open(png_path)
    img.load()
    a = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    if size is not None and (a.shape[1], a.shape[0]) != tuple(size):
        a = _downsample(a, size, is_normal)
    return a


def _downsample(a, size, is_normal):
    """Resolve a supersampled cell down to its final size.

    Pillow's RGBA resize is already alpha-weighted, so a STRAIGHT-alpha cell goes
    in and comes back out straight, with the transparent black outside the
    silhouette correctly ignored. Premultiplying by hand here would apply that
    weighting twice and leave a bright halo along the edge.
    """
    im = Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGBA")
    # BOX is a plain area average, which is exactly what supersampling wants at an
    # integer ratio; Lanczos only earns its ringing on a fractional one.
    exact = (a.shape[1] % size[0] == 0) and (a.shape[0] % size[1] == 0)
    out = np.asarray(im.resize(tuple(size), Image.BOX if exact else Image.LANCZOS),
                     dtype=np.float32) / 255.0
    if is_normal:
        # Averaging unit normals shortens them, so renormalise before re-encoding
        # or the downsampled sheet reads as a slightly flattened surface.
        n = out[:, :, :3] * 2.0 - 1.0
        n = n / np.maximum(np.sqrt((n * n).sum(axis=2, keepdims=True)), 1e-6)
        out[:, :, :3] = np.clip(n * 0.5 + 0.5, 0.0, 1.0)
    return out


def _parse_rgb(text, fallback=(0.0, 0.0, 0.0)):
    try:
        vals = [float(c) / 255.0 for c in text.split(",")][:3]
        return vals if len(vals) == 3 else list(fallback)
    except ValueError:
        return list(fallback)


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
                "engine": (["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
                           {"default": "CYCLES",
                            "tooltip": "BLENDER_EEVEE_NEXT is kept only so older workflows still load; it is an alias of "
                                       "BLENDER_EEVEE (Blender 4.2+ renamed it back). Cycles is the reference look."}),
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
                "render_normals": ("BOOLEAN", {"default": False,
                                               "tooltip": "Also render a matching VIEW-SPACE normal-map set (same directions/frames/layout) "
                                                          "via a Blender material override. True geometry normals, not an image estimate. "
                                                          "OpenGL convention: R=+X right, G=+Y up, B=+Z toward camera. Roughly doubles render time."}),
                "supersample": ("INT", {"default": 2, "min": 1, "max": 4,
                                        "tooltip": "Render each cell N times larger and Lanczos it back down. 2 is the single "
                                                   "biggest quality win on small sprites (clean edges and clean alpha) and costs "
                                                   "roughly 3.5x the render time. 1 = off."}),
                "view_transform": (["Standard", "AgX", "Khronos PBR Neutral", "Filmic", "Raw"],
                                   {"default": "Standard",
                                    "tooltip": "Blender's tone mapping. Blender defaults to AgX, which desaturates and rolls off "
                                               "highlights on purpose - on character art it reads muddy. Standard keeps the "
                                               "texture's own colour. Does not affect the normal pass, which is always Raw."}),
                "ambient_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 5.0, "step": 0.05,
                                               "tooltip": "Environment light. This used to be locked to bg_color at full strength, so "
                                                          "a light backdrop flattened the shading. Higher = softer and flatter."}),
                "ambient_color": ("STRING", {"default": "190, 200, 215",
                                             "tooltip": "Colour of the environment light. A cool ambient against the warm key is what "
                                                        "gives the form its depth."}),
                "key_strength": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 20.0, "step": 0.1,
                                           "tooltip": "Main sun, warm, from the front-left. Casts the shadow that reads the form."}),
                "fill_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 20.0, "step": 0.1,
                                            "tooltip": "Opposite side, cool. Lifts the shadow side without erasing it."}),
                "rim_strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1,
                                           "tooltip": "Backlight. This is what pops the silhouette off the background - the single "
                                                      "most game-sprite-looking light in the rig."}),
                "sun_softness": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 45.0, "step": 0.5,
                                           "tooltip": "Angular size of the suns, in degrees. 0 = razor-hard shadow edges, "
                                                      "6 = soft daylight."}),
                "ground_shadow": ("BOOLEAN", {"default": False,
                                              "tooltip": "Add a shadow-catcher floor so the character casts a contact shadow into the "
                                                         "cell's alpha instead of floating. Cycles only; excluded from the normal pass."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "masks", "atlas", "atlas_path", "report",
                    "normals", "normal_atlas", "normal_atlas_path")
    FUNCTION = "render"
    CATEGORY = "UniRig/Render"
    DESCRIPTION = ("Render an animated rigged FBX/GLB from N evenly spaced isometric camera angles "
                   "(8 by default) across the whole animation, and pack the result into a sprite "
                   "atlas. images is the flat batch in direction-major order; atlas is the sheet. "
                   "Enable render_normals for a matching view-space normal-map set and sheet.")

    # --- helpers -----------------------------------------------------------
    def _load_cells(self, out_dir, prefix, directions, frames, proc, size=None):
        cells = []
        for di in range(directions):
            row = []
            for fi in range(frames):
                p = os.path.join(out_dir, "%sd%02d_f%03d.png" % (prefix, di, fi))
                if not os.path.exists(p):
                    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
                    raise RuntimeError(
                        "Blender produced no %ssprite cell d%02d_f%03d (exit %s):\n%s"
                        % ("normal " if prefix else "", di, fi, proc.returncode, "\n".join(tail)))
                row.append(_to_rgba(p, size=size, is_normal=bool(prefix)))
            cells.append(row)
        return cells

    def _build_sheet(self, cells, layout, directions, frames):
        flat = [c for row in cells for c in row]
        h, w = flat[0].shape[:2]
        if layout == "rows=direction":
            rows, cols = directions, frames
            grid = cells
        else:
            rows, cols = frames, directions
            grid = [[cells[di][fi] for di in range(directions)] for fi in range(frames)]
        sheet = np.zeros((rows * h, cols * w, 4), dtype=np.float32)
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                sheet[r * h:(r + 1) * h, c * w:(c + 1) * w, :] = cell
        return flat, sheet, rows, cols, h, w

    def _save_sheet(self, sheet, prefix, sheet_w, sheet_h, transparent):
        full, base, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), sheet_w, sheet_h)
        path = os.path.join(full, "%s_%05d_.png" % (base, counter))
        mode = "RGBA" if transparent else "RGB"
        arr = (np.clip(sheet, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(arr, "RGBA").convert(mode).save(path)
        return path

    # --- main --------------------------------------------------------------
    def render(self, mesh_path, directions, frames, cell_width, cell_height, elevation,
               start_azimuth, clockwise, zoom, engine, samples, transparent, bg_color,
               layout, filename_prefix, save_atlas,
               frame_start=-1, frame_end=-1, include_last=False, render_normals=False,
               supersample=2, view_transform="Standard", ambient_strength=0.65,
               ambient_color="190, 200, 215", key_strength=5.5, fill_strength=1.2,
               rim_strength=5.0, sun_softness=6.0, ground_shadow=False):
        directions = int(directions)
        frames = int(frames)
        mesh = _resolve_mesh(mesh_path.strip())
        python = _find_python()
        tmpdir = tempfile.mkdtemp(prefix="iso_sprite_")
        try:
            out_dir = os.path.join(tmpdir, "cells")
            cfg_path = os.path.join(tmpdir, "cfg.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "mesh_path": mesh, "out_dir": out_dir,
                    "directions": directions, "frames": frames,
                    "cell_width": int(cell_width), "cell_height": int(cell_height),
                    "elevation": float(elevation), "start_azimuth": float(start_azimuth),
                    "clockwise": bool(clockwise), "zoom": float(zoom),
                    "engine": engine, "samples": int(samples),
                    "transparent": bool(transparent), "bg_color": bg_color,
                    "frame_start": int(frame_start), "frame_end": int(frame_end),
                    "include_last": bool(include_last),
                    "render_normals": bool(render_normals),
                    "supersample": int(supersample),
                    "view_transform": view_transform,
                    "ambient_strength": float(ambient_strength),
                    "ambient_color": ambient_color,
                    "key_strength": float(key_strength),
                    "fill_strength": float(fill_strength),
                    "rim_strength": float(rim_strength),
                    "sun_softness": float(sun_softness),
                    "ground_shadow": bool(ground_shadow),
                }, f)

            proc = subprocess.run([python, WORKER, "--", cfg_path],
                                  capture_output=True, text=True, cwd=tmpdir)

            cell_size = (int(cell_width), int(cell_height))
            cells = self._load_cells(out_dir, "", directions, frames, proc, size=cell_size)

            if not transparent:
                # Blender always renders straight alpha now, so the backdrop is
                # composited here. That keeps bg_color out of the lighting.
                back = np.array(_parse_rgb(bg_color), dtype=np.float32)
                for row in cells:
                    for cell in row:
                        a = cell[:, :, 3:4]
                        cell[:, :, :3] = cell[:, :, :3] * a + back * (1.0 - a)

            info = {}
            for line in (proc.stdout or "").splitlines():
                if line.startswith("ISO_SPRITE_OK "):
                    info = json.loads(line[len("ISO_SPRITE_OK "):])

            flat, sheet, rows, cols, h, w = self._build_sheet(cells, layout, directions, frames)
            images = torch.from_numpy(np.stack([c[:, :, :3] for c in flat], axis=0))
            masks = torch.from_numpy(np.stack([c[:, :, 3] for c in flat], axis=0))
            atlas = torch.from_numpy(sheet[:, :, :3])[None, ...]

            atlas_path = ""
            if save_atlas:
                atlas_path = self._save_sheet(sheet, filename_prefix, cols * w, rows * h, transparent)

            # --- matching view-space normal pass ---------------------------
            normal_atlas_path = ""
            if render_normals:
                n_cells = self._load_cells(out_dir, "n_", directions, frames, proc,
                                           size=cell_size)
                n_flat, n_sheet, _, _, _, _ = self._build_sheet(n_cells, layout, directions, frames)
                normals = torch.from_numpy(np.stack([c[:, :, :3] for c in n_flat], axis=0))
                normal_atlas = torch.from_numpy(n_sheet[:, :, :3])[None, ...]
                if save_atlas:
                    normal_atlas_path = self._save_sheet(
                        n_sheet, filename_prefix + "_normal", cols * w, rows * h, True)
            else:
                # Flat "facing the camera" placeholder so downstream nodes still
                # receive a valid IMAGE when the pass is off.
                blank = torch.tensor([0.5, 0.5, 1.0]).view(1, 1, 1, 3).repeat(1, h, w, 1)
                normals = blank
                normal_atlas = blank.clone()

            report = json.dumps({
                "cells": len(flat), "directions": directions, "frames": frames,
                "cell": [w, h], "sheet": [cols * w, rows * h], "layout": layout,
                "elevation": float(elevation), "start_azimuth": float(start_azimuth),
                "clockwise": bool(clockwise), "engine": info.get("engine", engine),
                "clip_range": [info.get("frame_start"), info.get("frame_end")],
                "sampled_frames": info.get("frame_ids"),
                "atlas_path": atlas_path,
                "render_normals": bool(render_normals),
                "normal_cells": info.get("normal_count", 0),
                "normal_space": "view/camera space, OpenGL (R=+X right, G=+Y up, B=+Z toward camera)",
                "normal_atlas_path": normal_atlas_path,
                "supersample": info.get("supersample", int(supersample)),
                "render_resolution": info.get("render_resolution"),
                "view_transform": info.get("view_transform", view_transform),
                "ground_shadow": info.get("ground_shadow", bool(ground_shadow)),
                "lights": {"key": float(key_strength), "fill": float(fill_strength),
                           "rim": float(rim_strength), "sun_softness_deg": float(sun_softness),
                           "ambient": float(ambient_strength), "ambient_color": ambient_color,
                           "rig": "world-fixed (does not follow the camera)"},
            }, ensure_ascii=False, indent=2)
            return (images, masks, atlas, atlas_path, report,
                    normals, normal_atlas, normal_atlas_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {"IsoSpriteSheetRender": IsoSpriteSheetRender}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IsoSpriteSheetRender": "Iso Sprite Atlas Render (animation -> 8-dir atlas)"
}
