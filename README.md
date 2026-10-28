# path-sam

Segment Anything along a drawn path, in 3D. You trace a rough PATH along an axon,
a vessel or a fibre in the orkestrator 3D scene, run **Segment Anything Along Path**
from the shape's card, and get back a label crop (and, if you ask, a surface).

The model is [micro-sam](https://github.com/computational-cell-analytics/micro-sam),
Segment Anything fine-tuned for microscopy. The default is `vit_b_lm` for light
microscopy; `vit_l_lm`, `vit_b_em_organelles` and plain `vit_b` can be chosen too.

## How it works in 3D

SAM only ever sees 2D images. micro-sam's 3D mode takes a mask on one slice of a
stack and carries it to the next one. In z-planes, an axon running sideways shows
up as a long smear rather than an object, so the stack here is cut along the path:

1. The path is snapped onto the intensity ridge.
2. The volume is sampled in planes perpendicular to the path, one every
   `frame_step_um`. In each plane the structure is a compact blob at the centre,
   whatever its course.
3. Each plane is normalised to the structure's local brightness, so a fading stain
   stays visible. A plane too dim to prompt is also shown dim, never stretched
   into noise.
4. micro-sam is prompted every `prompt_every_um`: the centre as the object, and
   `n_negative` points on the surround as background. It carries the mask through
   the planes in between.
5. Every voxel within `radius_um` of the path reads the plane masks where it was
   sampled. The result is a label crop placed on the image.

The geometry is in `src/path_sam/geometry.py`; micro-sam is wrapped in
`src/path_sam/sam.py`.

## Tuning by re-running

The agent keeps three things between runs:
- the voxels it read;
- the smoothed crop;
- the SAM embeddings of the last few stacks.

The embeddings are nearly all of a run's time: one encoder pass per plane, about
10 planes/s on an RTX 4070. So a re-run that changes only the prompts takes well
under a second: `prompt_every_um`, `n_negative`, `min_contrast`, `extend_ends`,
`iou_threshold` and `projection`.

Changing the radius, the field, the smoothing, the ridge snap or the plane spacing
changes what SAM sees, so those re-embed.

## Flavours

| flavour | hardware | torch |
| --- | --- | --- |
| `cuda` | NVIDIA GPU with at least 6 GB of VRAM; host driver ≥ 580 | PyPI's CUDA 13 build (`--extra cu130`) |
| `vanilla` | any amd64 CPU; slow (no arm64: micro-sam's `bioimage-cpp` has no aarch64 wheel) | PyTorch's CPU index (`--extra cpu`) |

Both images bake `vit_b_lm` into `/opt/micro_sam`, so a pod needs no network for it.

## Development

```bash
uv sync --extra cu130          # or --extra cpu
uv run pytest                  # offline: geometry, the action against a stub mikro, and
                               # real micro-sam if its weights are downloaded and a GPU is there
uv run pytest -m hub           # against a konstruktor hub (Docker, about a minute to start)
arkitekt run dev
```

Part of the code is copied from the testing scripts, which are not installable:
- `straighten_path_app.py`: placement and voxel size (`placement.py`);
- `path_segment_app.py`: the voxel cache (`cache.py`);
- `path_segment_core.py`: the ridge snap and profile (`geometry.py`).
