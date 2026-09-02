# comfyui-iso-sprite

Turn a rigged, animated 3D character into an **isometric 8-direction sprite atlas**
inside ComfyUI. Blender (`bpy`) does the rendering in a subprocess; ComfyUI gets
back an IMAGE batch, a packed atlas sheet, and the numbers it needs to play the
result back at the clip's original speed.

Built to sit at the end of a [ComfyUI-UniRig](https://github.com/Comfy-Org/ComfyUI-UniRig)
pipeline: mesh → auto-rig → apply animation → **sprite atlas**.

![8 directions x 39 frames, rendered from one animated FBX](docs/example_atlas.webp)

*8 directions × 39 frames off a single 1.6 s clip. Rows are directions.*

<img src="docs/example_direction_loop.webp" width="256" alt="one direction, played back at the clip's original speed">

## Nodes

### `Iso Sprite Atlas Render` (`IsoSpriteSheetRender`)

Renders an animated FBX/GLB from N evenly spaced camera angles across the whole
animation and packs the cells into one sheet.

| input | default | notes |
| --- | --- | --- |
| `mesh_path` | — | animated rigged FBX/GLB. Wire `UniRig: Apply Animation`'s `animated_fbx_path` here. |
| `directions` | 8 | cameras spread evenly over 360°. 8 = classic isometric 8-way. |
| `frames` | 8 | animation frames sampled evenly across the clip. Total renders = `directions × frames`. |
| `cell_width` / `cell_height` | 256 | per-sprite size. |
| `elevation` | 30 | camera pitch. 30 = game isometric, 35.264 = true dimetric. |
| `start_azimuth` | 0 | yaw of direction 0. 0 = character faces the camera. |
| `clockwise` | true | direction order S, SW, W, NW, N, NE, E, SE. |
| `zoom` | 1.3 | ortho frame size relative to the subject. |
| `engine` | CYCLES | `BLENDER_EEVEE_NEXT` falls back to Cycles when the bundled `bpy` has no EEVEE. |
| `samples` | 24 | Cycles CPU samples per cell. |
| `transparent` | true | alpha background for the atlas PNG. |
| `layout` | rows=direction | atlas orientation. |
| `frame_start` / `frame_end` | -1 | -1 = use the clip's own range. |
| `include_last` | false | off = drop the duplicate final frame so a cycle tiles cleanly. |

Outputs: `images` (flat batch, direction-major), `masks`, `atlas` (the sheet),
`atlas_path`, `report` (JSON: sampled frames, clip range, sheet size).

**One camera per direction, sized to the union bounding box of every sampled
frame** — the character does not swim or rescale between cells.

### `Anim Clip Frame Count` (`AnimClipFrameCount`)

Reads the clip's real frame range and frame rate, then answers "how many sprite
frames, played at what fps". Without it, an 8-frame sheet of a 1.6 s clip played
at a guessed 12 fps runs ~2.4× too fast.

| input | default | notes |
| --- | --- | --- |
| `mesh_path` | — | same animated FBX/GLB. |
| `mode` | target_fps | `target_fps` / `every_nth_frame` / `all_frames`. |
| `target_fps` | 12 | sprite frames per second of clip. 10–15 is the usual game range. |
| `min_frames` / `max_frames` | 4 / 24 | `max_frames` is your render-time budget. |
| `speed` | 1.0 | playback multiplier; 1.0 = original speed. |
| `source_fps_override` | 0 | 0 = read from the file. Set 30 for Mixamo clips that import at the wrong rate. |

Outputs: `frames`, `playback_fps`, `frame_start`, `frame_end`, `duration_sec`, `report`.

Wire `frames` → `Iso Sprite Atlas Render.frames` and `playback_fps` →
`SaveAnimatedWEBP.fps`.

### `Pose Render (random pose -> IMAGE)` (`PoseRandomRender`)

Single still: import a rigged FBX/GLB, apply a seeded random humanoid pose,
render it. Useful for dataset generation and for eyeballing a rig.

## Requirements

A python that owns `bpy`. Installing
[ComfyUI-UniRig](https://github.com/Comfy-Org/ComfyUI-UniRig) provisions one and
these nodes find it automatically; otherwise point `POSE_RENDER_PYTHON` at a
python with `bpy` installed:

```bash
export POSE_RENDER_PYTHON=/path/to/python-with-bpy
```

Nothing runs in ComfyUI's own interpreter except argument marshalling and image
packing, so `bpy` never has to be compatible with ComfyUI's torch.

## Typical graph

```
Load3D → Hy3DLoadMesh → MIA: Auto Rig → UniRig: Apply Animation
                                              ├→ Anim Clip Frame Count ─┐
                                              └→ Iso Sprite Atlas Render ┤
                                                     ├ atlas  → PreviewImage / SaveImage
                                                     └ images → Get Image from Batch → SaveAnimatedWEBP
```

`Get Image from Batch` with `batch_index = direction × frames` and
`length = frames` pulls a single direction out for a looping preview.

## Notes

- `SaveAnimatedWEBP` drops alpha, so a `transparent` render previews on black.
  Turn `transparent` off and set `bg_color` if you want a filled background.
- Render cost is `directions × frames` cells. On a small character at 256 px /
  24 samples this is roughly 0.15 s per cell on CPU.
- The atlas PNG is written to ComfyUI's output directory via `filename_prefix`;
  `atlas_path` returns where it landed.

## License

MIT
