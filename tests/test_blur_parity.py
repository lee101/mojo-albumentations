"""Parity tests: `mojo_albumentations` vs upstream `albumentations.blur.functional`."""

from __future__ import annotations

import random

import albumentations.augmentations.blur.functional as BF
import numpy as np
import pytest

import mojo_albumentations as M


def assert_exact(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype, f"{got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("ksize", [2, 3, 5, 9])
def test_box_blur_u8_is_bit_exact(img_u8: np.ndarray, ksize: int) -> None:
    assert_exact(M.box_blur(img_u8, ksize), BF.box_blur(img_u8, ksize))


def test_box_blur_gray_is_bit_exact(gray_u8: np.ndarray) -> None:
    assert_exact(M.box_blur(gray_u8, 3), BF.box_blur(gray_u8, 3))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.5, 7.0])
def test_create_gaussian_kernel_1d(sigma: float) -> None:
    assert_exact(
        M.create_gaussian_kernel_1d(sigma, 0),
        BF.create_gaussian_kernel_1d(sigma, 0),
    )


@pytest.mark.parametrize("sigma", [1.0, 3.0])
@pytest.mark.parametrize("ksize", [0, 5, 11])
def test_create_gaussian_kernel(sigma: float, ksize: int) -> None:
    assert_exact(
        M.create_gaussian_kernel(sigma, ksize), BF.create_gaussian_kernel(sigma, ksize)
    )


@pytest.mark.parametrize("size", [1, 3, 99, 100, 151, 256])
def test_create_gaussian_kernel_input_array(size: int) -> None:
    assert_exact(
        M.create_gaussian_kernel_input_array(size),
        BF.create_gaussian_kernel_input_array(size),
    )


@pytest.mark.parametrize("angle", [0.0, 17.0, 45.0, 90.0, 133.7])
@pytest.mark.parametrize("direction", [-1.0, -0.4, 0.0, 0.6, 1.0])
@pytest.mark.parametrize("allow_shifted", [False, True])
def test_create_motion_kernel(
    angle: float, direction: float, allow_shifted: bool
) -> None:
    got = M.create_motion_kernel(9, angle, direction, allow_shifted, random.Random(1234))
    want = BF.create_motion_kernel(
        9, angle, direction, allow_shifted, random.Random(1234)
    )
    assert_exact(got, want)


def test_gaussian_blur_float32_matches_opencv(img_f32: np.ndarray) -> None:
    import cv2

    from mojo_albumentations.augmentations.blur.functional import (
        create_gaussian_kernel_1d,
        gaussian_blur_float32,
    )

    for sigma, ksize in ((1.0, 5), (2.0, 9)):
        got = gaussian_blur_float32(img_f32, create_gaussian_kernel_1d(sigma, ksize))
        want = cv2.GaussianBlur(img_f32, (ksize, ksize), sigmaX=sigma)
        np.testing.assert_allclose(got, want, atol=2e-4, rtol=0)