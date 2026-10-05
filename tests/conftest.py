"""Shared fixtures.

Upstream `albumentations` is a hard dependency of the parity tests; if it is
not importable the suite fails loudly rather than silently skipping, because a
green run without upstream means nothing.

The compiled shared library is equally load-bearing: every assertion in this
suite runs against `dist/libmojo-albumentations.so`. There is no pure-Python
fallback in the wrapper, so a missing library is an import error rather than a
quiet pass - `test_library_is_loaded` pins that down explicitly.
"""

from __future__ import annotations

import albumentations.augmentations.blur.functional as BF  # noqa: F401
import albumentations.augmentations.geometric.functional as FG  # noqa: F401
import albumentations.augmentations.pixel.functional as PF  # noqa: F401
import cv2  # noqa: F401
import numpy as np
import pytest

from mojo_albumentations import _lib

RNG_SEED = 20260901


def pytest_report_header() -> str:
    return f"mojo-albumentations kernels: {_lib.loaded_from()}"


def test_library_is_loaded() -> None:
    """Fail loudly, not quietly, if the shared library is absent."""
    assert _lib._LIB_PATH.exists(), (
        f"{_lib._LIB_PATH} is missing; run `pixi run build`"
    )
    # One exported symbol resolved by name, to prove the .so is the source.
    assert hasattr(_lib.lib, "alb_apply_lut_u8")


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def img_u8(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def gray_u8(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, (64, 96), dtype=np.uint8)


@pytest.fixture(scope="session")
def img_f32(rng: np.random.Generator) -> np.ndarray:
    return rng.random((64, 96, 3), dtype=np.float32)


@pytest.fixture(scope="session")
def struct_kernel() -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))