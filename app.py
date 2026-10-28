"""Segment Anything along a drawn PATH, in 3D: micro-sam guided by a rough tracing.

You draw a rough PATH through a volume in the orkestrator 3D scene, along an axon,
a vessel or a fibre. Then you run **Segment Anything Along Path** from the shape's
card. The action:

- snaps the path onto the intensity ridge;
- cuts the volume into cross-sections perpendicular to it, so the structure is a
  compact blob at the centre of every one, whatever its course through the volume;
- prompts micro-sam with the centre (and the surround as background) every
  `prompt_every_um`, and lets it carry the mask through the frames between;
- maps the frames back onto the voxels and returns a label crop, which the scene
  adds as a LABEL layer. A surface comes too, with `with_mesh`.

Segment Anything sees 2D images only. Here the stack it carries a mask through is
made along the path, not along z, which is what makes it work in 3D: in z-planes,
an axon running sideways is a smear, not an object. The geometry is in
``src/path_sam/geometry.py``, micro-sam in ``src/path_sam/sam.py``.

**Re-running is the adjustment loop.** The agent keeps the voxels it read (bricks,
per lens), the smoothed crop, and the image embeddings of the last few stacks.
The embeddings are nearly all of a run's time (one pass of SAM's encoder per
frame), so a re-run that changes only the prompts (`prompt_every_um`,
`n_negative`, `min_contrast`, `extend_ends`, `iou_threshold`, `projection`) takes
well under a second.

Placement is the one ``testing/path_segment_app.py`` proved: vertices go from the
path's drawing space onto the lens' voxels along the coordinate graph. The label
crop is a CATEGORIZED translation of the lens with no pyramid.

Run:  arkitekt run dev
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Optional, Tuple

import numpy as np
import xarray as xr
from arkitekt import App, Description, Task, run
from mikro import Mikro, mikro_service
from mikro.api.schema import (
    AnnotationDerivedFromInput,
    AxisType,
    MeshLens,
    ValueRelation,
)
from mikro.arkitekt.specs import LabelMask, PathAnnotation, SingleChannelVolume
from mikro.arkitekt.widgets import reachable_from

from path_sam import geometry
from path_sam.cache import Lru, VoxelCache
from path_sam.placement import _ON_A_FACE, dataset_name_of, lens_volume, path_frame, spacing_um_of
from path_sam.sam import MODEL_TYPES, MicroSam, load_predictor

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.2.0"


@dataclass
class SamCache:
    """The app context: what every task of this agent shares."""

    voxels: VoxelCache = field(
        default_factory=lambda: VoxelCache(int(float(os.environ.get("PATH_SAM_CACHE_GB", "4")) * 1024**3))
    )
    #: A stack's embeddings are 4 MiB per frame (256 x 64 x 64 float32), so only the
    #: last few stacks are kept: three of a long path at the default step are gigabytes.
    embeddings: Lru = field(default_factory=lambda: Lru(int(os.environ.get("PATH_SAM_EMBEDDINGS", "3"))))


# The app is a declaration: who it is, what it offers, and the services it uses.
# `arkitekt run` finds it in this module (as `app`) and runs it; so does
# `python app.py`, below.
app = App(
    "path-sam",
    __version__,
    description="Segment Anything (micro-sam) along a drawn path, in 3D",
    author="jhnnsrs",
    services=[mikro_service],
    app_context=SamCache,
)

COLLECTIONS = ["segment"]


# Plain Enums, not (str, Enum): a str member as a default turns the port into free text.
class SamModel(Enum):
    """The micro-sam model to segment with."""

    VIT_B_LM = "vit_b_lm"
    VIT_L_LM = "vit_l_lm"
    VIT_B_EM_ORGANELLES = "vit_b_em_organelles"
    VIT_B = "vit_b"


class Projection(Enum):
    """What micro-sam carries from one cross-section to the next."""

    MASK = "mask"
    BOX = "box"
    POINTS = "points"
    POINTS_AND_MASK = "points_and_mask"
    SINGLE_POINT = "single_point"


assert {m.value for m in SamModel} == set(MODEL_TYPES)


@dataclass
class Outputs:
    label: Any
    mesh: Any = None


def segment_path(
    mikro: Mikro,
    task: Task,
    cache: SamCache,
    path: Any,
    lens: Any,
    params: geometry.SamParams,
    segmenter: geometry.Segmenter,
    *,
    model_key: str,
    with_mesh: bool,
) -> Outputs:
    """Segment along `path` in what `lens` shows, and register what was found."""
    started = time.perf_counter()
    voxels = cache.voxels
    space_axes = tuple(lens.axes_of_type(AxisType.SPACE))
    if len(space_axes) != 3:
        raise ValueError(f"Segmenting along a path needs three space axes, the image has {list(space_axes)}.")
    vertices, frame = path_frame(mikro, path, lens, space_axes)
    if frame.warning:
        task.log(frame.warning)

    # A path pinned to a timepoint is segmented at that timepoint. Everything else
    # beside the space axes has to be a single position (a pinned channel, say).
    lens_axes = tuple(lens.axis_names)
    pinned = {
        str(c.name): int(c.value)
        for c in path.coordinates or ()
        if c.name in lens_axes and c.name not in space_axes and c.name not in lens.axes_of_type(AxisType.CHANNEL)
    }
    key, open_ = lens_volume(lens, space_axes, pinned)

    data = voxels.handle(key, open_)
    shape = np.array([data.sizes[axis] for axis in space_axes])
    if (vertices < -0.5 - _ON_A_FACE).any() or (vertices > shape - 0.5 + _ON_A_FACE).any():
        raise ValueError(
            f"Annotation '{path.name}' leaves the image: its vertices span "
            f"{vertices.min(axis=0).round(1).tolist()} to {vertices.max(axis=0).round(1).tolist()} "
            f"over {list(space_axes)}, the image is {shape.tolist()}."
        )

    spacing_um = voxels.remember(str(lens.id), "spacing", lambda: spacing_um_of(mikro, lens))
    calibrated = all(axis in spacing_um for axis in space_axes)
    spacing = np.array([spacing_um[axis] for axis in space_axes]) if calibrated else np.ones(3)
    if not calibrated:
        task.log("The image is not calibrated: every length below is in voxels, not micrometres.")
    unit = "µm" if calibrated else "px"

    low, high = geometry.required_box(vertices, spacing, shape, params)
    fetched_before = voxels.fetched
    task.progress(10, f"Reading {(high - low).tolist()} voxels around '{path.name}'")
    crop = voxels.read(key, open_, low, high)
    warming = voxels.warm(key, data)
    read_s = time.perf_counter() - started

    smoothed = voxels.field(
        (key, tuple(low), tuple(high), params.sigma_um, tuple(spacing)),
        lambda: geometry.smooth(crop, spacing, params.sigma_um),
    )
    embedded_before = cache.embeddings.misses

    def embed(images: np.ndarray) -> Any:
        # Keyed by what the encoder sees, so every knob that changes the frames
        # (path, radius, field, smoothing, ridge snap, frame step) is a new key,
        # and every knob that only changes the prompts is not.
        digest = hashlib.blake2b(np.ascontiguousarray(images).tobytes(), digest_size=16).hexdigest()
        task.progress(30, f"Embedding {len(images)} cross-sections")
        return cache.embeddings.get((model_key, images.shape, digest), lambda: segmenter.embed(images))

    result = geometry.segment(crop, vertices - low, spacing, params, segmenter, field=smoothed, embed=embed)
    if result.empty:
        raise ValueError(
            f"Nothing along '{path.name}' stands out of its surroundings: no cross-section has "
            f"{params.min_contrast:g} of the peak contrast, or SAM found nothing at the prompts. "
            "Lower min_contrast, widen the radius, or check that the path runs along the stain."
        )
    summary = result.summary
    task.log(
        f"{summary['volume_um3']:.2f} {unit}³ over {summary['segmented_length_um']:.1f} {unit} "
        f"({int(summary['frames_segmented'])} of {int(summary['frames'])} cross-sections, "
        f"{len(result.keyframes)} prompted), mean diameter {summary['mean_diameter_um']:.2f} {unit}"
    )

    name = path.name or f"path {path.id}"
    source = dataset_name_of(mikro, lens)
    task.progress(70, "Uploading the label crop")
    # The crop keeps every axis of the lens, the non-space ones as single positions, so
    # its edge back to the lens is a plain translation with one entry per lens axis.
    origin_space = dict(zip(space_axes, (low + result.origin).astype(float)))
    origin = [origin_space.get(axis, float(pinned.get(axis, 0))) for axis in lens_axes]
    label_array = (
        xr.DataArray(result.mask.astype(np.uint8), dims=list(space_axes))
        .expand_dims([axis for axis in lens_axes if axis not in space_axes])
        .transpose(*lens_axes)
    )
    label_ds = mikro.create_array_dataset(
        data=label_array,
        scales=[],  # a mean of label ids is another id, and a subsample pyramid drifts
        name=f"{source} · SAM along {name} (r {params.radius_um:g} {unit})",
        axes=lens.carried_axes(list(lens_axes)),
        derived_from=[
            lens.derive_translation(origin, value_relation=ValueRelation.CATEGORIZED),
            AnnotationDerivedFromInput(annotation=path.id, value_relation=ValueRelation.TRANSFORMED),
        ],
    )
    outputs = Outputs(label=label_ds.lens())

    if with_mesh:
        task.progress(85, "Meshing")
        outputs.mesh = upload_mesh(mikro, lens, space_axes, result, low, name, source)

    total = time.perf_counter() - started
    stages = ", ".join(f"{k} {v * 1000:.0f}" for k, v in result.timings.items())
    reused = cache.embeddings.misses == embedded_before
    task.log(
        f"{total:.2f} s: read {read_s:.2f} s ({voxels.fetched - fetched_before} bricks fetched, "
        f"{voxels.bytes / 1024**2:.0f} MiB cached{', warming the rest' if warming else ''}); "
        f"{'embeddings reused; ' if reused else ''}segment ms: {stages}"
    )
    task.progress(100, f"Segmented '{name}'")
    return outputs


def upload_mesh(
    mikro: Mikro,
    lens: Any,
    space_axes: tuple[str, ...],
    result: geometry.Segmentation,
    low: np.ndarray,
    name: str,
    source: str,
) -> MeshLens:
    """The surface, in the lens' voxels with xyz components, as a one-object mesh collection."""
    from fabriks import build_collection

    verts, faces = geometry.surface(result.mask, low + result.origin)
    # Components reversed, faces left alone: the reversal is a reflection and turns the
    # marching-cubes faces outward by itself. The octree wants non-negative cells, and a
    # cap on a volume face sits at -0.5, so it is clipped rather than moved.
    xyz_axes = list(reversed(space_axes))
    verts_xyz = np.clip(verts[:, ::-1], 0.0, None)
    # `axes` names the vertex slots. Left out, the viewer guesses them as the
    # collection's last three axes reversed, which transposes this (x, y, z) one.
    built = build_collection({1: (verts_xyz, faces)}, levels=2, axes=tuple(xyz_axes))
    collection = mikro.create_mesh_collection(
        name=f"{source} · SAM surface along {name}",
        store=built,
        axes=xyz_axes,
        # A projection: BY_DIMENSION naming x, y and z, by axis name -- the lens may
        # carry axes the surface has not (a channel), which a MAP_AXIS (a pure
        # permutation) is refused for, and an axis no one names is not claimed.
        derived_from=[lens.derive_projection(xyz_axes, value_relation=ValueRelation.TRANSFORMED)],
    )
    return collection.lens()


# --------------------------------------------------------------------------- #
# The actions
# --------------------------------------------------------------------------- #


@app.action(name="Segment Anything Along Path", collections=COLLECTIONS, concurrency="serial")
def segment_anything_along_path(
    path: Annotated[PathAnnotation, Description("The PATH to segment along: a rough tracing is enough")],
    lens: Annotated[SingleChannelVolume, reachable_from("path"), Description("The channel the structure is stained in")],
    radius_um: Annotated[float, Description("How far from the path the structure may reach")] = 1.5,
    model: Annotated[SamModel, Description("The micro-sam model: light microscopy, EM organelles or plain SAM")] = SamModel.VIT_B_LM,
    field_um: Annotated[
        Optional[float], Description("Half the width of a cross-section SAM sees. Default 3x radius")
    ] = None,
    frame_step_um: Annotated[
        Optional[float], Description("Distance between cross-sections. Default a third of the radius")
    ] = None,
    prompt_every_um: Annotated[float, Description("Distance between prompted cross-sections")] = 2.0,
    n_negative: Annotated[int, Description("Background points per prompt, around the structure")] = 8,
    min_contrast: Annotated[
        float, Description("Cross-sections below this fraction of the peak contrast are not prompted")
    ] = 0.15,
    sigma_um: Annotated[float, Description("Smoothing before anything reads the intensity")] = 0.3,
    refine_path: Annotated[bool, Description("Snap the path onto the intensity ridge first")] = True,
    extend_ends: Annotated[bool, Description("Let SAM follow the structure past the first and last prompt")] = False,
    iou_threshold: Annotated[
        float, Description("When extending, stop where a cross-section overlaps the previous less than this")
    ] = 0.8,
    projection: Annotated[
        Projection, Description("What is carried from one cross-section to the next")
    ] = Projection.MASK,
    with_mesh: Annotated[bool, Description("Also upload a surface (slower)")] = False,
    *,
    mikro: Mikro,
    task: Task,
    cache: SamCache,
) -> Tuple[
    Annotated[LabelMask, Description("The segmentation, as a label crop")],
    Annotated[Optional[MeshLens], Description("The surface, if asked for")],
]:
    """Segment Anything Along Path

    Cuts the volume into cross-sections along a rough path, prompts micro-sam at
    the path every few micrometres and lets it follow the structure between, in
    3D. Fast to re-run with other prompts: the embeddings stay cached
    """
    params = geometry.SamParams(
        radius_um=radius_um,
        field_um=field_um,
        sigma_um=sigma_um,
        refine=refine_path,
        frame_step_um=frame_step_um,
        prompt_every_um=prompt_every_um,
        n_negative=n_negative,
        min_contrast=min_contrast,
        extend_ends=extend_ends,
        iou_threshold=iou_threshold,
        projection=projection.value,
    )
    out = segment_path(
        mikro, task, cache, path, lens, params, MicroSam(model.value),
        model_key=model.value, with_mesh=with_mesh,
    )
    return out.label, out.mesh


@app.action(name="Warm SAM Model", collections=COLLECTIONS)
def warm_sam_model(
    model: Annotated[SamModel, Description("The micro-sam model to load")] = SamModel.VIT_B_LM,
    *,
    task: Task,
) -> str:
    """Warm SAM Model

    Loads a micro-sam model onto the GPU before the first segmentation, so that
    one is as fast as the rest
    """
    task.progress(10, f"Loading {model.value}")
    started = time.perf_counter()
    predictor = load_predictor(model.value)
    return f"{model.value} ({MODEL_TYPES[model.value]}) is loaded on {predictor.device} in {time.perf_counter() - started:.1f} s"


if __name__ == "__main__":
    run(app, context=SamCache())
