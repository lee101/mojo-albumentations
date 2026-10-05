"""Parity tests: `mojo_albumentations` vs upstream `albumentations.geometric.functional`."""

from __future__ import annotations

import albumentations.augmentations.geometric.functional as FG
import cv2
import numpy as np
import pytest

import mojo_albumentations as M


def assert_exact(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype, f"{got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("member", ["e", "r90", "r180", "r270", "v", "hvt", "h", "t"])
def test_d4_is_bit_exact(img_u8: np.ndarray, member: str) -> None:
    assert_exact(M.d4(img_u8, member), FG.d4(img_u8, member))


def test_d4_rejects_unknown_member(img_u8: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Invalid group member"):
        M.d4(img_u8, "nope")


def test_transpose_is_bit_exact(img_u8: np.ndarray) -> None:
    assert_exact(M.transpose(img_u8), FG.transpose(img_u8))


@pytest.mark.parametrize("factor", [0, 1, 2, 3])
def test_rot90_is_bit_exact(img_u8: np.ndarray, factor: int) -> None:
    assert_exact(M.rot90(img_u8, factor), FG.rot90(img_u8, factor))


def test_vflip_hflip_are_bit_exact(img_u8: np.ndarray) -> None:
    assert_exact(M.vflip(img_u8), FG.vflip(img_u8))
    assert_exact(M.hflip(img_u8), FG.hflip(img_u8))


def test_extend_value() -> None:
    assert list(M.extend_value(0.5, 3)) == list(FG.extend_value(0.5, 3))
    assert list(M.extend_value((0.1, 0.2), 3)) == list(FG.extend_value((0.1, 0.2), 3))


@pytest.mark.parametrize("border_mode", [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE])
@pytest.mark.parametrize("pad", [(1, 2, 3, 4), (0, 0, 5, 0), (7, 0, 0, 7)])
def test_copy_make_border_is_bit_exact(
    img_u8: np.ndarray, pad: tuple[int, int, int, int], border_mode: int
) -> None:
    top, bottom, left, right = pad
    assert_exact(
        M.copy_make_border_with_value_extension(
            img_u8, top, bottom, left, right, border_mode, 0.0
        ),
        FG.copy_make_border_with_value_extension(
            img_u8, top, bottom, left, right, border_mode, 0.0
        ),
    )


def test_pad_is_bit_exact(img_u8: np.ndarray) -> None:
    assert_exact(
        M.pad(img_u8, 100, 130, cv2.BORDER_CONSTANT, 0.0),
        FG.pad(img_u8, 100, 130, cv2.BORDER_CONSTANT, 0.0),
    )


def test_remap_nearest_is_bit_exact(img_u8: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    map_x = rng.uniform(-20, 115, img_u8.shape[:2]).astype(np.float32)
    map_y = rng.uniform(-20, 85, img_u8.shape[:2]).astype(np.float32)
    assert_exact(
        M.remap(img_u8, map_x, map_y, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0),
        FG.remap(img_u8, map_x, map_y, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0),
    )


def test_remap_identity_is_bit_exact(img_u8: np.ndarray) -> None:
    height, width = img_u8.shape[:2]
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    assert_exact(
        M.remap(img_u8, map_x, map_y, cv2.INTER_NEAREST, cv2.BORDER_REPLICATE),
        FG.remap(img_u8, map_x, map_y, cv2.INTER_NEAREST, cv2.BORDER_REPLICATE),
    )


def test_remap_linear_is_bit_exact(img_u8: np.ndarray) -> None:
    """`cv2.remap(INTER_LINEAR)` - upstream's default interpolation."""
    rng = np.random.default_rng(4)
    map_x = rng.uniform(-20, 115, img_u8.shape[:2]).astype(np.float32)
    map_y = rng.uniform(-20, 85, img_u8.shape[:2]).astype(np.float32)
    assert_exact(
        M.remap(img_u8, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0),
        cv2.remap(img_u8, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
    )


def test_remap_rejects_unsupported_interpolation(img_u8: np.ndarray) -> None:
    """Only INTER_NEAREST and INTER_LINEAR are ported; the rest raise."""
    h, w = img_u8.shape[:2]
    zeros = np.zeros((h, w), dtype=np.float32)
    for interpolation in (cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA):
        with pytest.raises(NotImplementedError):
            M.remap(img_u8, zeros, zeros, interpolation, cv2.BORDER_CONSTANT, 0)


def test_remap_rejects_non_constant_border_value(img_u8: np.ndarray) -> None:
    h, w = img_u8.shape[:2]
    zeros = np.zeros((h, w), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        M.remap(img_u8, zeros, zeros, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 7)


def test_create_affine_transformation_matrix_matches() -> None:
    params = dict(
        translate={"x": 4.0, "y": -2.0},
        shear={"x": 10.0, "y": -5.0},
        scale={"x": 1.2, "y": 0.8},
        rotate=37.0,
        shift=(30.0, 48.0),
    )
    assert_exact(
        M.create_affine_transformation_matrix(**params),
        FG.create_affine_transformation_matrix(**params),
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0.9, 0.1, 3.0], [-0.2, 1.1, -4.0]]),
        np.array([[2.0, 0.0, -10.0], [0.0, 2.0, 7.0]]),
        np.array([[0.7, -0.3, 1.5], [0.4, 0.6, -2.5]]),
    ],
)
def test_warp_affine_matches_opencv(img_u8: np.ndarray, matrix: np.ndarray) -> None:
    height, width = img_u8.shape[:2]
    got = M.warp_affine(
        img_u8, matrix, cv2.INTER_NEAREST, 0, cv2.BORDER_CONSTANT, (height, width)
    )
    want = cv2.warpAffine(
        img_u8,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    assert_exact(got, want)


def test_warp_affine_identity_returns_input(img_u8: np.ndarray) -> None:
    matrix = np.eye(3)
    h, w = img_u8.shape[:2]
    got = M.warp_affine(img_u8, matrix, cv2.INTER_NEAREST, 0, cv2.BORDER_CONSTANT, (h, w))
    assert got is img_u8


def test_erode_dilate_are_bit_exact(
    img_u8: np.ndarray, struct_kernel: np.ndarray
) -> None:
    assert_exact(M.erode(img_u8, struct_kernel), FG.erode(img_u8, struct_kernel))
    assert_exact(M.dilate(img_u8, struct_kernel), FG.dilate(img_u8, struct_kernel))


@pytest.mark.parametrize("shape", [(3, 3), (4, 6)])
def test_morphology_matches(img_u8: np.ndarray, shape: tuple[int, int]) -> None:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    assert_exact(
        M.morphology(img_u8, kernel, "erosion"),
        FG.morphology(img_u8, kernel, "erosion"),
    )
    assert_exact(
        M.morphology(img_u8, kernel, "dilation"),
        FG.morphology(img_u8, kernel, "dilation"),
    )


def test_morphology_rejects_unknown_operation(img_u8: np.ndarray) -> None:
    kernel = np.ones((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported operation"):
        M.morphology(img_u8, kernel, "opening")


@pytest.mark.parametrize("same_dxdy", [True, False])
@pytest.mark.parametrize("distribution", ["gaussian", "uniform"])
def test_generate_displacement_fields_matches(
    same_dxdy: bool, distribution: str
) -> None:
    got = M.generate_displacement_fields(
        (48, 64), 12.0, 5.0, same_dxdy, (0, 0), np.random.default_rng(99), distribution
    )
    want = FG.generate_displacement_fields(
        (48, 64), 12.0, 5.0, same_dxdy, (0, 0), np.random.default_rng(99), distribution
    )
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, atol=1e-5, rtol=0)