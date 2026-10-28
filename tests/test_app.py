"""The app for itself, with no server.

The geometry is tested as the plain numpy it is, on a synthetic axon initial
segment: a curved tube, bright for 8 µm and fading, drawn over by a rough path.
The action runs against a stub mikro and a thresholding stand-in for SAM, so the
placement, the upload and the caches are checked without a GPU. Real micro-sam
runs last, and only where its weights are already downloaded.

Run these with `uv run pytest`; the release workflow does too.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from arkitekt import Task
from arkitekt.runtime import Runtime
from mikro.api.schema import ValueRelation

import app as path_sam_app
from app import Projection, SamCache, SamModel, app, segment_anything_along_path
from path_sam import geometry
from path_sam.cache import VoxelCache
from path_sam.placement import path_frame

from stubs import StubLens, StubMikro, ThresholdSam, drawn_path, graph


@pytest.fixture(scope="module")
def synth() -> geometry.Synthetic:
    return geometry.synthetic_ais()


@pytest.fixture
def fake_sam(monkeypatch: pytest.MonkeyPatch) -> ThresholdSam:
    """The action's `MicroSam` replaced by one thresholding stand-in for every call."""
    sam = ThresholdSam()
    monkeypatch.setattr(path_sam_app, "MicroSam", lambda model_type: sam)
    return sam


PARAMS = dict(radius_um=1.0, sigma_um=0.15)


def full_mask(result: geometry.Segmentation, shape: tuple[int, ...]) -> np.ndarray:
    return geometry.paste(result.mask, result.origin, shape)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_cross_sections_map_back_onto_the_voxels_they_came_from(synth):
    """`to_voxels` is the inverse of `straighten`, with the structure off the frames' centre.

    The path is moved 0.6-0.9 µm off the axon, so the axon sits off-centre in every
    frame. A mask found in the frames has to land back on the axon, and every wrong
    reading of a frame (v and u swapped, either mirrored, the frames reversed)
    lands it somewhere else. A centred structure would hide all four.
    """
    params = geometry.SamParams(radius_um=1.5, sigma_um=0.15, refine=False)
    field = geometry.smooth(synth.volume, synth.spacing, params.sigma_um)
    line = geometry.centreline_of(synth.rough + [2.0, 0.0, 6.0], synth.spacing, float(synth.spacing.min()))
    assert geometry.distance_to(line.points / synth.spacing, synth.truth, synth.spacing).min() > 0.5
    stack = geometry.straighten(field, line, synth.spacing, params)
    bright = stack.frames > 600
    target = synth.truth_mask & (field > 600)

    def placed(frame_masks: np.ndarray) -> float:
        coords, inside = geometry.to_voxels(frame_masks, stack, synth.spacing, synth.volume.shape, params.radius_um)
        full = np.zeros(synth.volume.shape, dtype=bool)
        full[tuple(coords[inside].T)] = True
        return geometry.dice(full, target)

    assert placed(bright) > 0.9
    for wrong in (np.swapaxes(bright, 1, 2), bright[:, ::-1], bright[:, :, ::-1], bright[::-1]):
        assert placed(wrong) < 0.3


def test_frames_are_shown_to_sam_on_one_scale(synth):
    """The structure shows as about 1 where it is bright, and a frame too dim to prompt stays dim.

    Past the AIS only the faint axon is left, at under `min_contrast` of the peak.
    Normalised by its own contrast it would be as bright as the AIS (and the noise
    around it as loud); the floor keeps it below.
    """
    params = geometry.SamParams(**PARAMS)
    field = geometry.smooth(synth.volume, synth.spacing, params.sigma_um)
    line = geometry.centreline_of(synth.rough, synth.spacing, 0.3)
    stack = geometry.straighten(field, line, synth.spacing, params)
    images = geometry.shown_to_sam(stack, params)

    assert images.dtype == np.float32
    assert np.all(images[:, 0, 0] == params.clip[0]) and np.all(images[:, -1, -1] == params.clip[1])
    peak = int(np.argmax(stack.contrast))
    centre = slice(stack.half - 2, stack.half + 3)
    assert images[peak, centre, centre].mean() > 0.8
    dim = int(np.argmin(stack.contrast))
    assert stack.contrast[dim] < params.min_contrast * stack.contrast.max()
    assert images[dim, centre, centre].mean() < 0.8
    assert images[dim, centre, centre].mean() < images[peak, centre, centre].mean() / 2


def test_keyframes_cover_the_structure_end_to_end(synth):
    params = geometry.SamParams(**PARAMS, prompt_every_um=2.0)
    field = geometry.smooth(synth.volume, synth.spacing, params.sigma_um)
    line = geometry.centreline_of(synth.rough, synth.spacing, 0.25)
    stack = geometry.straighten(field, line, synth.spacing, params)
    keyframes = geometry.keyframes_of(stack, params)
    valid = np.flatnonzero(stack.contrast >= params.min_contrast * stack.contrast.max())

    assert keyframes[0] == valid[0] and keyframes[-1] == valid[-1]
    assert np.all(np.diff(keyframes)[:-1] * line.step >= params.prompt_every_um - 1e-9)


def test_only_the_part_on_the_path_is_kept():
    mask = np.zeros((21, 21), dtype=bool)
    mask[8:13, 8:13] = True  # under the centre
    mask[0:3, 0:3] = True  # a neighbour SAM picked up too
    kept = geometry.keep_centre(mask, half=10, radius_px=5)
    assert kept[10, 10] and not kept[1, 1]

    beside = np.zeros((21, 21), dtype=bool)
    beside[10:13, 13:16] = True  # near the centre but not under it
    assert geometry.keep_centre(beside, half=10, radius_px=5).sum() == beside.sum()


def test_the_pipeline_finds_the_axon(synth):
    sam = ThresholdSam()
    result = geometry.segment(synth.volume, synth.rough, synth.spacing, geometry.SamParams(**PARAMS), sam)

    assert geometry.dice(full_mask(result, synth.volume.shape), synth.truth_mask) > 0.75
    assert sam.embedded == 1 and sam.prompted == len(result.keyframes) >= 3
    # The segmentation follows the AIS (3-11 µm along the truth), not the faint axon beyond it.
    assert 6.0 < result.summary["segmented_length_um"] < 12.0


def test_nothing_to_prompt_is_empty_not_an_error():
    flat = np.full((20, 40, 40), 100, dtype=np.uint16)
    path = np.array([[10.0, 5.0, 5.0], [10.0, 35.0, 35.0]])
    result = geometry.segment(flat, path, np.ones(3), geometry.SamParams(radius_um=3.0), ThresholdSam())
    assert result.empty and len(result.keyframes) == 0


# --------------------------------------------------------------------------- #
# The action, against a stub mikro
# --------------------------------------------------------------------------- #


def test_a_path_drawn_on_a_scene_is_placed_on_the_voxels(synth):
    mikro = StubMikro(graph(synth.spacing))
    path = drawn_path(synth.rough, synth.spacing)
    vertices, frame = path_frame(mikro, path, StubLens(synth.volume), ("z", "y", "x"))

    assert np.allclose(vertices, synth.rough)
    assert np.allclose(frame.to_drawn(vertices), np.asarray(path.vectors))


def test_the_action_uploads_a_label_crop_placed_on_the_axon(synth, fake_sam):
    mikro = StubMikro(graph(synth.spacing))
    lens = StubLens(synth.volume)
    label, mesh = segment_anything_along_path(
        drawn_path(synth.rough, synth.spacing), lens, **PARAMS, with_mesh=True,
        mikro=mikro, task=Task.local(), cache=SamCache(VoxelCache(512 * 1024**2)),
    )

    created = mikro.datasets[-1]
    array = created["data"]
    assert label.id == "label-1"
    assert tuple(array.dims) == ("c", "z", "y", "x") and array.shape[0] == 1
    translation, parent = created["derived_from"]
    assert translation.value_relation == ValueRelation.CATEGORIZED
    assert str(parent.annotation) == "path-1"
    assert created["scales"] == []

    full = np.zeros(synth.volume.shape, dtype=bool)
    box = tuple(slice(int(o), int(o) + n) for o, n in zip(translation.offset[1:], array.shape[1:]))
    full[box] = array.values[0].astype(bool)
    assert geometry.dice(full, synth.truth_mask) > 0.75

    surface = mikro.meshes[0]
    assert mesh is not None and surface["axes"] == ["x", "y", "z"]
    assert surface["store"].manifest.axes == ("x", "y", "z")
    assert surface["derived_from"][0].kind == "BY_DIMENSION"


def test_a_re_run_with_other_prompts_reuses_the_embeddings_and_the_voxels(synth, fake_sam):
    mikro = StubMikro(graph(synth.spacing))
    lens = StubLens(synth.volume)
    cache = SamCache(VoxelCache(512 * 1024**2))
    path = drawn_path(synth.rough, synth.spacing)
    segment_anything_along_path(path, lens, **PARAMS, mikro=mikro, task=Task.local(), cache=cache)
    cache.voxels.settle()  # the first run pulls the whole small lens in the background
    fetched, opened = cache.voxels.fetched, lens.opened

    segment_anything_along_path(
        path, lens, **PARAMS, n_negative=2, prompt_every_um=1.0, projection=Projection.BOX,
        mikro=mikro, task=Task.local(), cache=cache,
    )
    assert fake_sam.embedded == 1, "the encoder ran again for a change of prompts"
    assert cache.voxels.fetched == fetched and lens.opened == opened

    # A different radius changes what the frames show: that is a new stack.
    segment_anything_along_path(path, lens, radius_um=1.2, sigma_um=0.15, mikro=mikro, task=Task.local(), cache=cache)
    assert fake_sam.embedded == 2


def test_nothing_to_segment_is_said_not_returned_empty(synth, fake_sam):
    flat = np.full_like(synth.volume, 100)
    with pytest.raises(ValueError, match="stands out"):
        segment_anything_along_path(
            drawn_path(synth.rough, synth.spacing), StubLens(flat), **PARAMS,
            mikro=StubMikro(graph(synth.spacing)), task=Task.local(), cache=SamCache(VoxelCache(512 * 1024**2)),
        )


def test_the_app_starts_and_offers_its_actions(local_runtime: Runtime[SamCache]):
    assert local_runtime.app is app
    assert {"segment_anything_along_path", "warm_sam_model"} <= set(app.registry.implementations)


# --------------------------------------------------------------------------- #
# Real micro-sam, where its weights are on this machine
# --------------------------------------------------------------------------- #


def _weights_present(model: str) -> bool:
    from micro_sam.util import get_cache_directory

    return (Path(get_cache_directory()) / "models" / model).exists()


@pytest.mark.skipif(not _weights_present(SamModel.VIT_B_LM.value), reason="micro-sam vit_b_lm is not downloaded")
def test_micro_sam_follows_the_axon(synth):
    from path_sam.sam import MicroSam, device

    if device() != "cuda" and not os.environ.get("PATH_SAM_TEST_CPU"):
        pytest.skip("no GPU (set PATH_SAM_TEST_CPU=1 to run it on the CPU, slowly)")

    started = time.perf_counter()
    result = geometry.segment(synth.volume, synth.rough, synth.spacing, geometry.SamParams(**PARAMS), MicroSam())
    seconds = time.perf_counter() - started
    score = geometry.dice(full_mask(result, synth.volume.shape), synth.truth_mask)
    print(f"micro-sam dice {score:.3f} in {seconds:.1f} s: {result.timings}")

    assert score > 0.6
    assert 6.0 < result.summary["segmented_length_um"] < 12.0
