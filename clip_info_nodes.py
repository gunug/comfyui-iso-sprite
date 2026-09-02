"""AnimClipFrameCount — size a sprite sheet to the animation it is rendering.

Reads the clip's real frame range and frame rate out of the FBX/GLB, then works
out how many sprite frames to bake and at what playback fps those frames run at
the clip's ORIGINAL speed. Wire `frames` into Iso Sprite Atlas Render and
`playback_fps` into SaveAnimatedWEBP and the sheet plays back at real time
instead of racing.
"""
import json
import os
import shutil
import subprocess
import tempfile

from .nodes import _find_python, _resolve_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "clip_info_worker.py")


class AnimClipFrameCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Animated rigged FBX/GLB path (wire UniRig: Apply Animation's animated_fbx_path here)."}),
                "mode": (["target_fps", "every_nth_frame", "all_frames"], {"default": "target_fps",
                          "tooltip": "target_fps: bake this many sprite frames per second of clip. every_nth_frame: keep 1 of every N source frames. all_frames: one sprite per source frame."}),
                "target_fps": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 60.0, "step": 0.5,
                                         "tooltip": "mode=target_fps only. 10-15 is the usual game-sprite range."}),
                "every_nth": ("INT", {"default": 3, "min": 1, "max": 60,
                                      "tooltip": "mode=every_nth_frame only."}),
                "min_frames": ("INT", {"default": 4, "min": 1, "max": 512}),
                "max_frames": ("INT", {"default": 24, "min": 1, "max": 512,
                                       "tooltip": "Hard cap. Renders cost directions x frames, so this is your render-time budget."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                                    "tooltip": "Playback speed multiplier. 1.0 = the clip's original speed."}),
                "source_fps_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 1.0,
                                                  "tooltip": "0 = read the frame rate from the file. Set 30 for Mixamo clips that import at the wrong rate."}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "playback_fps", "frame_start", "frame_end", "duration_sec", "report")
    FUNCTION = "measure"
    CATEGORY = "UniRig/Render"
    DESCRIPTION = ("Read an animated FBX/GLB's clip length and frame rate, and return how many "
                   "sprite frames to bake plus the playback fps that keeps the sheet at the "
                   "clip's original speed. Wire frames -> Iso Sprite Atlas Render, "
                   "playback_fps -> SaveAnimatedWEBP.")

    def measure(self, mesh_path, mode, target_fps, every_nth, min_frames, max_frames,
                speed, source_fps_override):
        mesh = _resolve_mesh(mesh_path.strip())
        python = _find_python()
        tmpdir = tempfile.mkdtemp(prefix="clip_info_")
        try:
            cfg_path = os.path.join(tmpdir, "cfg.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"mesh_path": mesh}, f)
            proc = subprocess.run([python, WORKER, "--", cfg_path],
                                  capture_output=True, text=True, cwd=tmpdir)

            info = None
            for line in (proc.stdout or "").splitlines():
                if line.startswith("CLIP_INFO_OK "):
                    info = json.loads(line[len("CLIP_INFO_OK "):])
            if info is None:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
                raise RuntimeError("Could not read the clip's timing (exit %s):\n%s"
                                   % (proc.returncode, "\n".join(tail)))

            f_start = int(info["frame_start"])
            f_end = int(info["frame_end"])
            length = max(1, int(info["length"]))
            src_fps = float(source_fps_override) if source_fps_override > 0 else float(info["scene_fps"])
            if src_fps <= 0:
                src_fps = 24.0
            duration = length / src_fps

            if mode == "all_frames":
                wanted = length
            elif mode == "every_nth_frame":
                wanted = int(round(length / float(max(1, every_nth))))
            else:
                wanted = int(round(duration * float(target_fps)))

            lo = min(int(min_frames), int(max_frames))
            hi = max(int(min_frames), int(max_frames))
            frames = max(lo, min(hi, max(1, wanted)))

            # The sheet holds `frames` samples spread over the whole clip, so
            # showing them at frames/duration reproduces the original speed.
            playback_fps = (frames / duration) * float(speed) if duration > 0 else 12.0
            playback_fps = max(0.1, min(240.0, playback_fps))

            report = json.dumps({
                "clip_frames": length, "frame_range": [f_start, f_end],
                "source_fps": round(src_fps, 3), "duration_sec": round(duration, 3),
                "mode": mode, "wanted_frames": wanted, "capped_to": [lo, hi],
                "frames": frames, "playback_fps": round(playback_fps, 3),
                "speed": float(speed), "actions": info.get("actions", []),
            }, ensure_ascii=False, indent=2)
            return (frames, float(playback_fps), f_start, f_end, float(duration), report)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {"AnimClipFrameCount": AnimClipFrameCount}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimClipFrameCount": "Anim Clip Frame Count (clip -> frames + fps)"
}
