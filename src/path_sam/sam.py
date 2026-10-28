"""micro-sam behind the `Segmenter` the geometry asks for.

torch and micro-sam are imported only when a model is first loaded. Declaring the
app (which `arkitekt plugin release` does inside the image, on a runner with no
GPU) never waits for them.

Every call into the predictor holds one lock. A SAM predictor is stateful: it
keeps the features of the frame it was last set to, and two tasks setting it
at once would answer one prompt with the other's frame. The GPU is one device
anyway.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

import numpy as np

from .geometry import Projection, Segmenter

logger = logging.getLogger(__name__)

#: The fine-tuned micro-sam models worth offering, and what each was trained on.
MODEL_TYPES = {
    "vit_b_lm": "light microscopy (ViT-B)",
    "vit_l_lm": "light microscopy (ViT-L, slower, more precise)",
    "vit_b_em_organelles": "electron microscopy organelles (ViT-B)",
    "vit_b": "the original Segment Anything (ViT-B)",
}

DEFAULT_MODEL = "vit_b_lm"

_PREDICTOR_LOCK = threading.Lock()


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_predictor(model_type: str = DEFAULT_MODEL) -> Any:
    """The micro-sam predictor for `model_type`, loaded once per process.

    Weights come from ``MICROSAM_CACHEDIR`` (baked into the image) and are
    downloaded there only when they are missing.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model '{model_type}'. Choose one of {sorted(MODEL_TYPES)}.")
    from micro_sam.util import get_sam_model

    where = device()
    logger.info("Loading micro-sam %s on %s (cache %s)", model_type, where, os.environ.get("MICROSAM_CACHEDIR", "default"))
    return get_sam_model(model_type=model_type, device=where)


class MicroSam(Segmenter):
    """A `Segmenter` on one micro-sam model."""

    def __init__(self, model_type: str = DEFAULT_MODEL):
        self.model_type = model_type

    @property
    def predictor(self) -> Any:
        return load_predictor(self.model_type)

    def embed(self, images: np.ndarray) -> dict:
        import torch
        from micro_sam.util import precompute_image_embeddings

        # The encoder is nearly all of a run (about 7 frames/s in fp32 on an RTX 4070,
        # 10 in bf16; batching does not help, it only runs out of memory). bf16 keeps
        # fp32's range, so it cannot overflow where fp16 could.
        precision = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device() == "cuda")
        with _PREDICTOR_LOCK, torch.inference_mode(), precision:
            return precompute_image_embeddings(self.predictor, images, ndim=3, verbose=False)

    def from_points(self, embeddings: dict, i: int, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        import torch
        from micro_sam.prompt_based_segmentation import segment_from_points

        with _PREDICTOR_LOCK, torch.inference_mode():
            mask = segment_from_points(self.predictor, points, labels, image_embeddings=embeddings, i=i)
        return np.asarray(mask).reshape(np.asarray(mask).shape[-2:])

    def propagate(
        self,
        segmentation: np.ndarray,
        embeddings: dict,
        keyframes: np.ndarray,
        *,
        extend_ends: bool,
        iou_threshold: float,
        projection: Projection,
    ) -> np.ndarray:
        import torch
        from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume

        with _PREDICTOR_LOCK, torch.inference_mode():
            filled, _ = segment_mask_in_volume(
                segmentation.copy(),
                self.predictor,
                embeddings,
                np.asarray(keyframes),
                stop_lower=not extend_ends,
                stop_upper=not extend_ends,
                iou_threshold=iou_threshold,
                projection=projection,
            )
        return filled
