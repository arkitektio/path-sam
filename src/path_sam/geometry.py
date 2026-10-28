"""The geometry around SAM: a path's volume as a stack of cross-sections, and back.

Segment Anything sees 2D images only, and micro-sam's 3D mode carries a mask from
one slice of a stack to the next. A linear structure crosses z-planes at whatever
angle it runs, often as a long smear, so the stack is not the volume's own planes:

    rough path     ->  arclength-uniform centreline, parallel-transported frames
    ridge snap     ->  each sample moves to the intensity centroid of its cross-section
    straighten     ->  frame i is the plane perpendicular to the path at sample i,
                       so the structure is a compact blob at the centre of every frame
    normalise      ->  N = (I - bg_i) / (fg_i - bg_i) per frame, clipped to a fixed range
    SAM            ->  (sam.py) prompt the centre of keyframes, propagate through the rest
    back           ->  each voxel near the path reads the mask stack where it was sampled

All geometry is in the volume's own axis order and, given a voxel size, in
micrometres. With ``spacing = (1, 1, 1)`` every "_um" reads as voxels.

The path helpers come from ``testing/straighten_path_app.py`` and the ridge snap,
profile, corridor and finish from ``testing/path_segment_core.py``, copied, since
neither is installable. A frame's ``(v, u)`` runs along ``(binormal, normal)``, as
``cross_sections`` has it; ``to_voxels`` is the exact inverse of ``straighten``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    gaussian_filter1d,
    label,
    map_coordinates,
)
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

Projection = Literal["mask", "points", "box", "points_and_mask", "single_point"]

#: A straightened frame is at least this many pixels across, finer than the voxels
#: when the field is small: SAM resizes every frame to 1024 anyway.
_MIN_FRAME_PX = 96


@dataclass(frozen=True)
class SamParams:
    """Every knob of one run. Physical units when the volume is calibrated, voxels otherwise."""

    #: How far from the centreline a voxel may be and still belong to the structure.
    radius_um: float = 1.5
    #: Half the width of a straightened frame. `None` means three times the radius.
    field_um: float | None = None
    #: Gaussian smoothing applied before anything reads the intensity.
    sigma_um: float = 0.3
    #: Snap the centreline onto the intensity ridge before straightening.
    refine: bool = True
    #: How far a centreline sample may move in one ridge-snap iteration.
    search_um: float = 1.0
    #: Ridge-snap iterations.
    iterations: int = 2
    #: Width of the background ring outside `radius_um`.
    shell_um: float = 1.0
    #: Smoothing of shifts and profiles along the path.
    along_um: float = 1.0
    #: Distance between frames along the path. `None` means a third of the radius (and
    #: never finer than a voxel). Every frame is a pass of SAM's encoder, the slow part.
    frame_step_um: float | None = None
    #: Distance between prompted frames.
    prompt_every_um: float = 2.0
    #: Negative points per prompted frame, on a ring just outside the corridor.
    n_negative: int = 8
    #: A frame whose contrast is below this fraction of the peak gets no prompt.
    min_contrast: float = 0.15
    #: Let SAM carry the mask past the first and last prompted frames.
    extend_ends: bool = False
    #: When extending, stop where a frame's mask overlaps the previous one less than this.
    iou_threshold: float = 0.8
    #: What micro-sam carries from one frame to the next.
    projection: Projection = "mask"
    #: The normalised intensity range a frame is shown to SAM in (0 = surround, 1 = the core).
    clip: tuple[float, float] = (-0.5, 1.5)

    @property
    def field(self) -> float:
        wanted = self.field_um if self.field_um is not None else 3.0 * self.radius_um
        # The negative prompts sit on the background ring, which has to be in the frame.
        return max(wanted, self.radius_um + self.shell_um)

    def frame_step(self, finest: float) -> float:
        if self.frame_step_um is not None:
            return max(float(self.frame_step_um), 1e-3)
        return max(finest, self.radius_um / 3.0)

    @property
    def margin_um(self) -> float:
        """How far past the path's vertices anything is read."""
        return (
            max(self.field, self.radius_um + self.shell_um)
            + (self.iterations * self.search_um if self.refine else 0.0)
            + 3.0 * self.sigma_um
        )


@dataclass
class Stack:
    """The straightened cross-sections of one path, and how they sit in the crop."""

    line: "Centreline"
    frames: np.ndarray  # (m, d, d) float32, the smoothed field
    pixel: float  # in-plane pixel size, physical
    half: int  # frames are 2 * half + 1 pixels across; the path is at (half, half)
    fg: np.ndarray  # (m,)
    bg: np.ndarray  # (m,)

    @property
    def contrast(self) -> np.ndarray:
        return np.clip(self.fg - self.bg, 0.0, None)


@dataclass
class Segmentation:
    """What one run found. Index coordinates are crop-local (the crop the volume was)."""

    mask: np.ndarray  # bool, the bounding box of the segmentation
    origin: np.ndarray  # (3,) int: crop-local index of mask[0, 0, 0]
    centreline: np.ndarray  # (m, 3) crop-local index coordinates
    arclength: np.ndarray  # (m,)
    keyframes: np.ndarray  # frame indices that were prompted
    frame_masks: np.ndarray  # (m, d, d) bool, what SAM returned per frame
    summary: dict[str, float]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.mask.any()


# --------------------------------------------------------------------------- #
# Centreline, fields and the stages before SAM (copied)
# --------------------------------------------------------------------------- #


def resample_path(
    vertices: np.ndarray, spacing: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit tangents along the path, uniform in arc length.

    `vertices` is ``(n, 3)`` in index units, `spacing` the physical size of one
    index step per axis and `step` the physical distance between samples. A
    cubic spline is fitted through the vertices in *physical* space, so a step
    along a coarse axis counts for the distance it really covers. Both results
    are physical, ``(m, 3)``.
    """
    points = np.asarray(vertices, dtype=np.float64) * spacing
    # A repeated vertex has no direction and makes the spline fit singular.
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
    points = points[keep]
    if len(points) < 2:
        raise ValueError("A path needs at least two distinct vertices to be straightened.")

    chord = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    dense = int(max(2000, 8 * chord / step))
    u_dense = np.linspace(0.0, 1.0, dense)
    try:
        tck, _ = splprep(list(points.T), s=0, k=min(3, len(points) - 1))
        curve = np.column_stack(splev(u_dense, tck))
    except Exception:
        # The polyline itself, parametrised by chord length.
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        curve = np.column_stack(
            [np.interp(u_dense * along[-1], along, points[:, i]) for i in range(3)]
        )

    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    n_samples = max(2, int(round(length[-1] / step)) + 1)
    targets = np.linspace(0.0, length[-1], n_samples)
    centres = np.column_stack([np.interp(targets, length, curve[:, i]) for i in range(3)])

    tangents = np.gradient(curve, axis=0)
    tangents = np.column_stack([np.interp(targets, length, tangents[:, i]) for i in range(3)])
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    return centres, tangents


def transport_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A normal and binormal per tangent, carried along the path without twist.

    The first normal is arbitrary; each next one is the previous projected onto
    the new perpendicular plane. Rebuilding the frame per point from a fixed
    reference instead makes the cross-section jump wherever the tangent passes
    that reference.
    """
    first = tangents[0]
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(first)))] = 1.0
    normal = np.cross(first, reference)
    normal /= np.linalg.norm(normal)

    normals = np.empty_like(tangents)
    for i, tangent in enumerate(tangents):
        normal = normal - np.dot(normal, tangent) * tangent
        normal /= np.linalg.norm(normal)
        normals[i] = normal
    return normals, np.cross(tangents, normals)


class _Clock:
    """Per-stage wall time, so a slow run says where it was slow."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = self.timings.get(stage, 0.0) + now - self._last
        self._last = now


def smooth(volume: np.ndarray, spacing: np.ndarray, sigma_um: float) -> np.ndarray:
    """The volume as float32, blurred physically round (anisotropic voxels taken into account)."""
    values = np.asarray(volume, dtype=np.float32)
    if sigma_um <= 0:
        return values
    return gaussian_filter(values, sigma=sigma_um / np.asarray(spacing), mode="nearest", truncate=3.0)


def required_box(
    vertices: np.ndarray, spacing: np.ndarray, shape: np.ndarray, params: SamParams
) -> tuple[np.ndarray, np.ndarray]:
    """The ``[low, high)`` index box a run on these vertices reads, clipped to the volume."""
    margin = np.ceil(params.margin_um / np.asarray(spacing)).astype(int) + 2
    low = np.floor(vertices.min(axis=0)).astype(int) - margin
    high = np.ceil(vertices.max(axis=0)).astype(int) + margin + 1
    return np.clip(low, 0, shape), np.clip(high, 0, shape)


def disk_offsets(inner: float, outer: float, step: float) -> np.ndarray:
    """``(k, 2)`` offsets ``(v, u)`` on a grid of `step` in the annulus ``inner <= r <= outer``."""
    n = int(np.ceil(outer / step))
    v, u = np.meshgrid(np.arange(-n, n + 1) * step, np.arange(-n, n + 1) * step, indexing="ij")
    r = np.hypot(v, u)
    keep = (r >= inner) & (r <= outer + 1e-9)
    offsets = np.column_stack([v[keep], u[keep]])
    return offsets if len(offsets) else np.zeros((1, 2))


def cross_sections(field: np.ndarray, line: Centreline, offsets: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """``(m, k)``: the field at each offset of each sample's perpendicular plane, linearly."""
    points = (
        line.points[:, None, :]
        + offsets[None, :, 0, None] * line.binormals[:, None, :]
        + offsets[None, :, 1, None] * line.normals[:, None, :]
    )
    indices = (points / spacing).reshape(-1, 3).T
    return map_coordinates(field, indices, order=1, mode="nearest").reshape(points.shape[:2])


@dataclass
class Centreline:
    """Samples along a path, uniform in arclength, physical, in crop-local coordinates."""

    points: np.ndarray  # (m, 3)
    tangents: np.ndarray  # (m, 3)
    normals: np.ndarray  # (m, 3)
    binormals: np.ndarray  # (m, 3)
    step: float

    @property
    def arclength(self) -> np.ndarray:
        segments = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(segments)])


def centreline_of(vertices: np.ndarray, spacing: np.ndarray, step: float) -> Centreline:
    """A centreline through `vertices` (index coordinates), sampled every `step`."""
    points, tangents = resample_path(vertices, spacing, step)
    normals, binormals = transport_frames(tangents)
    return Centreline(points, tangents, normals, binormals, step)


def refine_centreline(
    field: np.ndarray, line: Centreline, spacing: np.ndarray, params: SamParams
) -> Centreline:
    """Move each sample to the intensity centroid of its cross-section, smoothly.

    Only what stands out of its own cross-section pulls: the median is subtracted
    first. Without that, a sample in a dim stretch is dragged towards whatever
    happens to be brightest within `search_um`. Shifts are clamped, then smoothed
    along the path, so one noisy plane cannot kink the line.
    """
    offsets = disk_offsets(0.0, params.search_um, line.step)
    for _ in range(max(0, params.iterations)):
        values = cross_sections(field, line, offsets, spacing)
        weights = np.clip(values - np.median(values, axis=1, keepdims=True), 0.0, None)
        total = weights.sum(axis=1, keepdims=True)
        shift_vu = np.divide(weights @ offsets, total, out=np.zeros((len(values), 2)), where=total > 0)
        shift = shift_vu[:, :1] * line.binormals + shift_vu[:, 1:] * line.normals
        norm = np.linalg.norm(shift, axis=1, keepdims=True)
        shift *= np.minimum(1.0, params.search_um / np.maximum(norm, 1e-12))
        shift = gaussian_filter1d(shift, sigma=params.along_um / line.step, axis=0, mode="nearest")
        line = centreline_of((line.points + shift) / spacing, spacing, line.step)
    return line


def profile(
    field: np.ndarray, line: Centreline, spacing: np.ndarray, params: SamParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(fg, bg)`` per sample: the core mean and the median of a ring around the corridor."""
    core = max(line.step, 0.5 * params.radius_um)
    fg = cross_sections(field, line, disk_offsets(0.0, core, line.step), spacing).mean(axis=1)
    ring = disk_offsets(params.radius_um, params.radius_um + params.shell_um, line.step)
    bg = np.median(cross_sections(field, line, ring, spacing), axis=1)
    sigma = params.along_um / line.step
    return (
        gaussian_filter1d(fg, sigma, mode="nearest"),
        gaussian_filter1d(bg, sigma, mode="nearest"),
    )


def corridor(
    shape: tuple[int, ...], line: Centreline, spacing: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(coords, distance, nearest)`` of every voxel within `radius` of the centreline.

    `coords` is ``(K, 3)`` in C order (so its raveled indices are sorted), `nearest`
    the index of the closest sample. Candidates are painted as boxes around each
    sample and then measured exactly. Memory scales with the corridor, not the
    crop, which matters for a long diagonal path whose bounding box is mostly empty.
    """
    candidate = np.zeros(shape, dtype=bool)
    half = np.ceil(radius / spacing).astype(int) + 1
    centres = np.rint(line.points / spacing).astype(int)
    for centre in centres[:: max(1, int(half.min() // 2) or 1)]:
        low = np.clip(centre - half, 0, shape)
        high = np.clip(centre + half + 1, 0, shape)
        candidate[low[0] : high[0], low[1] : high[1], low[2] : high[2]] = True
    # The stride above can skip the last sample's box.
    low = np.clip(centres[-1] - half, 0, shape)
    high = np.clip(centres[-1] + half + 1, 0, shape)
    candidate[low[0] : high[0], low[1] : high[1], low[2] : high[2]] = True

    coords = np.argwhere(candidate)
    distance, nearest = cKDTree(line.points).query(coords * spacing, distance_upper_bound=radius)
    inside = np.isfinite(distance)
    return coords[inside], distance[inside], nearest[inside]


def _finish(coords: np.ndarray, reached: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Close one voxel and fill holes, inside the corridor, on the tight box of the result."""
    if not reached.any():
        return np.zeros((0, 0, 0), dtype=bool), np.zeros(3, dtype=int)
    hits = coords[reached]
    low = np.maximum(hits.min(axis=0) - 1, 0)
    high = np.minimum(hits.max(axis=0) + 2, shape)
    inside = coords[np.all((coords >= low) & (coords < high), axis=1)] - low
    allowed = np.zeros(high - low, dtype=bool)
    allowed[tuple(inside.T)] = True
    mask = np.zeros(high - low, dtype=bool)
    mask[tuple((hits - low).T)] = True
    mask = binary_fill_holes(binary_closing(mask, structure=np.ones((3, 3, 3)), border_value=0) | mask) & allowed
    # Tight again: the margin above was only for the closing.
    found = np.argwhere(mask)
    tight_low, tight_high = found.min(axis=0), found.max(axis=0) + 1
    mask = mask[tuple(slice(int(a), int(b)) for a, b in zip(tight_low, tight_high))]
    return mask, low + tight_low


def surface(mask: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes over `mask`: ``(verts, faces)``, verts in index coordinates of the crop.

    The pad is what lets the surface close where the mask touches its own box. It
    moves the origin by one voxel, so the offset is ``origin - 1``. The vertices are
    in array axis order. Reverse the components (``verts[:, ::-1]``) for xyz, and do
    not also flip the faces: the reversal is a reflection and already turns them
    outward. ``step_size`` stays 1, because coarser steps shift the surface.
    """
    padded = np.pad(mask, 1).astype(np.uint8)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, step_size=1)
    return verts + (np.asarray(origin, dtype=np.float64) - 1.0), faces


# --------------------------------------------------------------------------- #
# Straightening, and back
# --------------------------------------------------------------------------- #


#: Frames sampled at once, so a long path does not hold every sample position in memory.
_FRAME_BATCH = 64


def straighten(
    field: np.ndarray, line: Centreline, spacing: np.ndarray, params: SamParams, fine: Centreline | None = None
) -> Stack:
    """The cross-section of `field` at every centreline sample, plus its core and surround.

    The core and surround are read along `fine` (the same path, sampled every finest
    voxel) when it is given, and carried onto the frames by arclength: `profile`
    samples its disks at the line's own step, which is too coarse at the frame step.
    """
    if fine is None:
        fg, bg = profile(field, line, spacing, params)
    else:
        fine_fg, fine_bg = profile(field, fine, spacing, params)
        fine_at = fine.arclength * (line.arclength[-1] / max(fine.arclength[-1], 1e-12))
        fg = np.interp(line.arclength, fine_at, fine_fg)
        bg = np.interp(line.arclength, fine_at, fine_bg)
    pixel = min(float(np.min(spacing)), 2.0 * params.field / _MIN_FRAME_PX)
    half = int(np.ceil(params.field / pixel))
    offsets = np.arange(-half, half + 1) * pixel
    v, u = np.meshgrid(offsets, offsets, indexing="ij")
    frames = np.empty((len(line.points), 2 * half + 1, 2 * half + 1), dtype=np.float32)
    for start in range(0, len(line.points), _FRAME_BATCH):
        part = slice(start, start + _FRAME_BATCH)
        points = (
            line.points[part, None, None, :]
            + v[None, :, :, None] * line.binormals[part, None, None, :]
            + u[None, :, :, None] * line.normals[part, None, None, :]
        )
        indices = (points / spacing).reshape(-1, 3).T
        frames[part] = map_coordinates(field, indices, order=1, mode="nearest").reshape(points.shape[:3])
    return Stack(line=line, frames=frames, pixel=pixel, half=half, fg=fg, bg=bg)


def to_voxels(
    frame_masks: np.ndarray, stack: Stack, spacing: np.ndarray, shape: tuple[int, ...], radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(coords, inside)``: the corridor voxels and which of them the frame masks cover.

    The inverse of `straighten`. A voxel near the path reads the mask stack where it
    was sampled: at its nearest sample's frame (shifted along the tangent by its
    offset, so it falls between two frames), at its ``(v, u)`` in that frame. The
    read is linear over the 0/1 masks and kept at one half. Only voxels within
    `radius` of the centreline are asked, so SAM cannot grow a result past it.
    """
    line = stack.line
    coords, _, nearest = corridor(shape, line, spacing, radius)
    offset = coords * spacing - line.points[nearest]
    along = nearest + np.einsum("ij,ij->i", offset, line.tangents[nearest]) / line.step
    v = np.einsum("ij,ij->i", offset, line.binormals[nearest]) / stack.pixel + stack.half
    u = np.einsum("ij,ij->i", offset, line.normals[nearest]) / stack.pixel + stack.half
    values = map_coordinates(
        frame_masks.astype(np.float32), np.vstack([along, v, u]), order=1, mode="constant", cval=0.0
    )
    return coords, values >= 0.5


def shown_to_sam(stack: Stack, params: SamParams) -> np.ndarray:
    """The frames as SAM sees them: locally normalised, every frame on the same scale.

    micro-sam stretches each frame from its own minimum to its own maximum. A frame
    with nothing in it would be stretched into noise, and SAM finds objects in
    noise. So the frames are normalised to the path's local core and surround
    (1 = as bright as the structure here, 0 = its surroundings), clipped, and the two
    far corners of each frame are pinned to the ends of the clip. The corners lie
    outside the corridor, so nothing ever reads them back.

    A fading stain stays at 1 as it fades, which is the point. Dividing by a
    contrast of next to nothing would amplify noise just the same, though, so the
    divisor never falls below `min_contrast` of the path's peak: a frame too dim to
    be prompted is also shown dim.
    """
    low, high = params.clip
    floor = max(params.min_contrast * float(stack.contrast.max()), 1e-6)
    span = np.maximum(stack.fg - stack.bg, floor)
    images = (stack.frames - stack.bg[:, None, None]) / span[:, None, None]
    images = np.clip(images, low, high).astype(np.float32)
    images[:, 0, 0] = low
    images[:, -1, -1] = high
    return images


def keyframes_of(stack: Stack, params: SamParams) -> np.ndarray:
    """The frames to prompt: about every `prompt_every_um` where the structure stands out.

    The first and last such frames are always among them, so nothing between the
    ends is left to propagation from one side only.
    """
    contrast = stack.contrast
    peak = float(contrast.max()) if len(contrast) else 0.0
    if peak <= 0:
        return np.zeros(0, dtype=int)
    valid = np.flatnonzero(contrast >= params.min_contrast * peak)
    every = max(1, int(round(params.prompt_every_um / stack.line.step)))
    picked = [int(valid[0])]
    for index in valid[1:]:
        if index - picked[-1] >= every:
            picked.append(int(index))
    if picked[-1] != valid[-1]:
        picked.append(int(valid[-1]))
    return np.asarray(picked, dtype=int)


def prompts_for(stack: Stack, params: SamParams) -> tuple[np.ndarray, np.ndarray]:
    """``(points, labels)`` in a frame's ``(row, column)``: the path, and the surround around it."""
    centre = float(stack.half)
    points = [(centre, centre)]
    ring = (params.radius_um + 0.5 * params.shell_um) / stack.pixel
    for k in range(max(0, params.n_negative)):
        angle = 2.0 * np.pi * (k + 0.5) / params.n_negative
        points.append((centre + ring * np.sin(angle), centre + ring * np.cos(angle)))
    labels = np.array([1] + [0] * max(0, params.n_negative))
    return np.asarray(points), labels


def keep_centre(mask: np.ndarray, half: int, radius_px: float) -> np.ndarray:
    """The one connected part of a frame's mask that is the structure on the path.

    The part under the frame's centre, or else the one with the most pixels within
    the corridor. A neighbour SAM picked up beside it is dropped.
    """
    mask = np.asarray(mask, dtype=bool).reshape(mask.shape[-2:])
    parts, count = label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return mask
    if parts[half, half]:
        return parts == parts[half, half]
    rows, cols = np.ogrid[: mask.shape[0], : mask.shape[1]]
    near = (rows - half) ** 2 + (cols - half) ** 2 <= radius_px**2
    counts = np.bincount(parts[near], minlength=count + 1)
    counts[0] = 0
    if counts.max() == 0:
        return np.zeros_like(mask)
    return parts == int(np.argmax(counts))


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


class Segmenter:
    """What `segment` needs from SAM. ``sam.MicroSam`` is the real one; tests pass a fake.

    `embed` takes the ``(m, d, d)`` frames `shown_to_sam` makes; `from_points`
    answers a prompt on frame `i` with a ``(d, d)`` mask; `propagate` fills the
    frames between (and with `extend_ends`, beyond) the masked keyframes.
    """

    def embed(self, images: np.ndarray) -> object:
        raise NotImplementedError

    def from_points(self, embeddings: object, i: int, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def propagate(
        self,
        segmentation: np.ndarray,
        embeddings: object,
        keyframes: np.ndarray,
        *,
        extend_ends: bool,
        iou_threshold: float,
        projection: Projection,
    ) -> np.ndarray:
        raise NotImplementedError


def segment(
    volume: np.ndarray,
    vertices: np.ndarray,
    spacing: np.ndarray,
    params: SamParams,
    segmenter: Segmenter,
    *,
    field: np.ndarray | None = None,
    embed=None,
) -> Segmentation:
    """Segment the structure along `vertices` (crop-local index coordinates) in `volume`.

    `field` is ``smooth(volume, spacing, params.sigma_um)``, and `embed` a stand-in
    for ``segmenter.embed``: pass them when they are cached. A re-run that only
    changes the prompts then skips both the blur and the encoder.
    """
    clock = _Clock()
    spacing = np.asarray(spacing, dtype=np.float64)
    shape = tuple(int(n) for n in volume.shape)
    if field is None:
        field = smooth(volume, spacing, params.sigma_um)
    clock.lap("smooth")

    # The ridge snap and the profile read cross-sections at the line's own step, so
    # they run on a line sampled every finest voxel. The frames are then cut every
    # frame step along that same (snapped) line.
    finest = float(spacing.min())
    fine = centreline_of(vertices, spacing, finest)
    if params.refine:
        fine = refine_centreline(field, fine, spacing, params)
    step = params.frame_step(finest)
    line = fine if step <= finest else centreline_of(fine.points / spacing, spacing, step)
    clock.lap("refine")

    stack = straighten(field, line, spacing, params, fine=None if line is fine else fine)
    keyframes = keyframes_of(stack, params)
    images = shown_to_sam(stack, params)
    clock.lap("straighten")

    radius_px = params.radius_um / stack.pixel
    frame_masks = np.zeros(images.shape, dtype=np.uint8)
    prompted = []
    if len(keyframes):
        embeddings = (embed or segmenter.embed)(images)
        clock.lap("embed")
        points, labels = prompts_for(stack, params)
        for i in keyframes:
            found = keep_centre(segmenter.from_points(embeddings, int(i), points, labels), stack.half, radius_px)
            if found.any():
                frame_masks[i] = found
                prompted.append(int(i))
        clock.lap("prompt")
        if prompted:
            frame_masks = np.asarray(
                segmenter.propagate(
                    frame_masks,
                    embeddings,
                    np.asarray(prompted),
                    extend_ends=params.extend_ends,
                    iou_threshold=params.iou_threshold,
                    projection=params.projection,
                )
            ).reshape(images.shape)
            frame_masks = np.stack([keep_centre(f, stack.half, radius_px) for f in frame_masks]).astype(np.uint8)
        clock.lap("propagate")

    coords, inside = to_voxels(frame_masks, stack, spacing, shape, params.radius_um)
    mask, origin = _finish(coords, inside, shape)
    clock.lap("finish")

    return Segmentation(
        mask=mask,
        origin=origin,
        centreline=line.points / spacing,
        arclength=line.arclength,
        keyframes=np.asarray(prompted, dtype=int),
        frame_masks=frame_masks.astype(bool),
        summary=summarise(mask, frame_masks, stack, spacing, radius_px),
        timings=clock.timings,
    )


def summarise(
    mask: np.ndarray, frame_masks: np.ndarray, stack: Stack, spacing: np.ndarray, radius_px: float
) -> dict[str, float]:
    """Volume, segmented length and mean diameter, the last from the cross-sections themselves."""
    rows, cols = np.ogrid[: frame_masks.shape[1], : frame_masks.shape[2]]
    near = (rows - stack.half) ** 2 + (cols - stack.half) ** 2 <= radius_px**2
    area = (frame_masks.astype(bool) & near).sum(axis=(1, 2)) * stack.pixel**2
    filled = np.flatnonzero(area > 0)
    arclength = stack.line.arclength
    return {
        "path_length_um": float(arclength[-1]),
        "segmented_length_um": float(arclength[filled[-1]] - arclength[filled[0]]) if len(filled) else 0.0,
        "volume_um3": float(mask.sum()) * float(np.prod(spacing)),
        "n_voxels": float(mask.sum()),
        "mean_diameter_um": float(2.0 * np.sqrt(area[filled] / np.pi).mean()) if len(filled) else 0.0,
        "frames": float(len(frame_masks)),
        "frames_segmented": float(len(filled)),
    }


# --------------------------------------------------------------------------- #
# A synthetic AIS, for the tests (copied from path_segment_core)
# --------------------------------------------------------------------------- #


@dataclass
class Synthetic:
    volume: np.ndarray
    spacing: np.ndarray
    truth: np.ndarray  # (m, 3) index coordinates of the true centreline
    truth_arclength: np.ndarray
    truth_mask: np.ndarray  # bool, the AIS tube between `ais` bounds
    rough: np.ndarray  # (n, 3) index coordinates of a jittered hand-drawn path
    ais: tuple[float, float]  # true start/end, µm along the true centreline
    radius_um: float


def synthetic_ais(seed: int = 0) -> Synthetic:
    """A curved axon: bright AIS from 3 to 11 µm fading to 40 % distally, dim elsewhere.

    The volume is (z, y, x) at 0.3 x 0.1 x 0.1 µm, the tube 0.5 µm in radius. The
    rough path is six vertices on the true line, each knocked up to 0.3 µm sideways,
    and it runs 3 µm past the AIS on both ends, as a quick click-through would.
    """
    rng = np.random.default_rng(seed)
    spacing = np.array([0.3, 0.1, 0.1])
    shape = (40, 170, 170)
    radius, ais = 0.5, (3.0, 11.0)

    t = np.linspace(0.0, 1.0, 4000)
    line = np.column_stack(
        [6.0 + 2.0 * np.sin(np.pi * t), 3.0 + 11.0 * t, 3.0 + 9.0 * t + 2.0 * np.sin(2 * np.pi * t)]
    )
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])

    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing="ij"), -1)
    distance, nearest = cKDTree(line).query(grid.reshape(-1, 3))
    distance, along = distance.reshape(shape), length[nearest].reshape(shape)

    def strength(s: np.ndarray) -> np.ndarray:
        plateau = np.where(s < 6.0, 1.0, 1.0 - 0.6 * (s - 6.0) / (ais[1] - 6.0))
        edge = 1 / (1 + np.exp(-(s - ais[0]) / 0.1)) * 1 / (1 + np.exp((s - ais[1]) / 0.1))
        return plateau * edge

    tube = 1 / (1 + np.exp((distance - radius) / 0.05))
    volume = 100.0 + 1000.0 * strength(along) * tube + 80.0 * tube  # a faint axon all along
    volume = gaussian_filter(volume, sigma=0.12 / spacing) + rng.normal(0, 20.0, shape)
    truth_mask = (distance <= radius) & (along >= ais[0]) & (along <= ais[1])

    picks = np.interp(np.linspace(0.0, length[-1], 6), length, np.arange(len(line))).astype(int)
    rough = line[picks].copy()
    jitter = rng.normal(0, 1, rough.shape)
    rough += 0.3 * jitter / np.linalg.norm(jitter, axis=1, keepdims=True)
    return Synthetic(
        volume=np.clip(volume, 0, 65535).astype(np.uint16),
        spacing=spacing,
        truth=line / spacing,
        truth_arclength=length,
        truth_mask=truth_mask,
        rough=rough / spacing,
        ais=ais,
        radius_um=radius,
    )


def paste(mask: np.ndarray, origin: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A box mask placed back into a full volume of `shape`."""
    full = np.zeros(shape, dtype=bool)
    if mask.size:
        full[tuple(slice(int(o), int(o) + n) for o, n in zip(origin, mask.shape))] = mask
    return full


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def distance_to(points: np.ndarray, line: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Physical distance from each of `points` to the polyline `line` (both index coordinates)."""
    return cKDTree(line * spacing).query(points * spacing)[0]
