"""Parity tests: `mojo_albumentations` vs upstream `albumentations.pixel.functional`.

Every test compares against the upstream function it claims to port, on real
images, with a tolerance that reflects how the result is produced. Functions
whose port is bit-exact say so and assert equality; functions that inherit an
OpenCV fixed-point rounding assert the measured bound.
"""

from __future__ import annotations

import albumentations.augmentations.pixel.functional as PF
import numpy as np
import pytest

import mojo_albumentations as M

EXACT = 0
ONE_LSB = 1


def assert_exact(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype, f"{got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    np.testing.assert_array_equal(got, want)


def assert_close(got: np.ndarray, want: np.ndarray, atol: float) -> None:
    assert got.shape == want.shape
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    assert diff.max() <= atol, f"max abs diff {diff.max()} > {atol}"


# ---------------------------------------------------------------------------
# LUT transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_solarize_u8_is_bit_exact(img_u8: np.ndarray, threshold: float) -> None:
    assert_exact(M.solarize(img_u8, threshold), PF.solarize(img_u8, threshold))


def test_solarize_float32(img_f32: np.ndarray) -> None:
    assert_close(M.solarize(img_f32, 0.5), PF.solarize(img_f32, 0.5), 0.0)


@pytest.mark.parametrize("bits", [1, 3, 5, 7])
def test_posterize_u8_is_bit_exact(img_u8: np.ndarray, bits: int) -> None:
    assert_exact(M.posterize(img_u8, bits), PF.posterize(img_u8, bits))


def test_posterize_per_channel_u8_is_bit_exact(img_u8: np.ndarray) -> None:
    assert_exact(M.posterize(img_u8, [3, 4, 5]), PF.posterize(img_u8, [3, 4, 5]))


def test_equalize_u8_is_bit_exact(gray_u8: np.ndarray) -> None:
    assert_exact(M.equalize(gray_u8), PF.equalize(gray_u8))


def test_equalize_by_channels_is_bit_exact(img_u8: np.ndarray) -> None:
    assert_exact(M.equalize(img_u8, mode="cv", by_channels=True), PF.equalize(img_u8))


def test_equalize_rejects_mask(gray_u8: np.ndarray) -> None:
    with pytest.raises(NotImplementedError):
        M.equalize(gray_u8, mask=np.ones_like(gray_u8, dtype=np.uint8))


@pytest.mark.parametrize(
    ("low_y", "high_y"), [(0.0, 1.0), (0.25, 0.75), (0.4, 0.6), (0.0, 0.0)]
)
def test_move_tone_curve_scalar_is_bit_exact(
    img_u8: np.ndarray, low_y: float, high_y: float
) -> None:
    assert_exact(
        M.move_tone_curve(img_u8, low_y, high_y),
        PF.move_tone_curve(img_u8, low_y, high_y),
    )


def test_move_tone_curve_per_channel_is_bit_exact(img_u8: np.ndarray) -> None:
    low = np.array([0.1, 0.3, 0.5])
    high = np.array([0.9, 0.7, 0.6])
    assert_exact(
        M.move_tone_curve(img_u8, low, high),
        PF.move_tone_curve(img_u8, low, high),
    )


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.2, 4.0])
def test_gamma_transform_u8_is_bit_exact(img_u8: np.ndarray, gamma: float) -> None:
    assert_exact(M.gamma_transform(img_u8, gamma), PF.gamma_transform(img_u8, gamma))


@pytest.mark.parametrize("gamma", [0.5, 2.2])
def test_gamma_transform_f32(img_f32: np.ndarray, gamma: float) -> None:
    assert_close(M.gamma_transform(img_f32, gamma), PF.gamma_transform(img_f32, gamma), 1e-6)


def test_invert_u8_is_bit_exact(img_u8: np.ndarray) -> None:
    assert_exact(M.invert(img_u8), PF.invert(img_u8))


def test_channel_shuffle_is_bit_exact(img_u8: np.ndarray) -> None:
    for order in ([2, 1, 0], [1, 0, 2], [0, 2, 1]):
        assert_exact(M.channel_shuffle(img_u8, order), PF.channel_shuffle(img_u8, order))


# ---------------------------------------------------------------------------
# channel transforms
# ---------------------------------------------------------------------------


def test_to_gray_weighted_average_u8_within_one_lsb(img_u8: np.ndarray) -> None:
    """OpenCV 5's uint8 RGB2GRAY rounding is not reproducible from the
    coefficients alone; this pins the bound and the rate."""
    got = M.to_gray_weighted_average(img_u8)
    want = PF.to_gray_weighted_average(img_u8)
    assert_close(got, want, ONE_LSB)
    assert int((got != want).sum()) <= 0.005 * got.size


def test_to_gray_weighted_average_u8_divergence_rate(img_u8: np.ndarray) -> None:
    """Document the measured rate, not just the bound, so a regression shows."""
    got = M.to_gray_weighted_average(img_u8)
    want = PF.to_gray_weighted_average(img_u8)
    mismatched = int((got != want).sum())
    assert mismatched <= 0.005 * got.size, f"{mismatched} of {got.size} pixels differ"


def test_to_gray_weighted_average_f32(img_f32: np.ndarray) -> None:
    assert_close(
        M.to_gray_weighted_average(img_f32), PF.to_gray_weighted_average(img_f32), 1e-6
    )


def test_to_gray_desaturation_is_bit_exact(img_f32: np.ndarray) -> None:
    assert_exact(M.to_gray_desaturation(img_f32), PF.to_gray_desaturation(img_f32))


def test_to_gray_average_is_bit_exact(img_u8: np.ndarray, img_f32: np.ndarray) -> None:
    assert_exact(M.to_gray_average(img_u8), PF.to_gray_average(img_u8))
    assert_exact(M.to_gray_average(img_f32), PF.to_gray_average(img_f32))


def test_to_gray_max_is_bit_exact(img_u8: np.ndarray, img_f32: np.ndarray) -> None:
    assert_exact(M.to_gray_max(img_u8), PF.to_gray_max(img_u8))
    assert_exact(M.to_gray_max(img_f32), PF.to_gray_max(img_f32))


def test_grayscale_to_multichannel_is_bit_exact(gray_u8: np.ndarray) -> None:
    assert_exact(
        M.grayscale_to_multichannel(gray_u8, 3), PF.grayscale_to_multichannel(gray_u8, 3)
    )
    assert_exact(
        M.grayscale_to_multichannel(gray_u8, 1), PF.grayscale_to_multichannel(gray_u8, 1)
    )


def test_linear_transformation_rgb_f32(img_f32: np.ndarray) -> None:
    m = np.array(
        [[1.1, -0.2, 0.05], [0.1, 0.9, -0.02], [-0.05, 0.15, 1.0]], dtype=np.float32
    )
    assert_close(
        M.linear_transformation_rgb(img_f32, m),
        PF.linear_transformation_rgb(img_f32, m),
        1e-5,
    )


# ---------------------------------------------------------------------------
# torchvision-style adjustments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.4, 1.7, 2.5])
def test_adjust_brightness_f32_is_exact(img_f32: np.ndarray, factor: float) -> None:
    assert_exact(
        M.adjust_brightness_torchvision(img_f32, factor),
        PF.adjust_brightness_torchvision(img_f32, factor),
    )


@pytest.mark.parametrize("factor", [0.4, 1.7])
def test_adjust_brightness_u8_is_exact(img_u8: np.ndarray, factor: float) -> None:
    assert_exact(
        M.adjust_brightness_torchvision(img_u8, factor),
        PF.adjust_brightness_torchvision(img_u8, factor),
    )


def test_adjust_brightness_identity_branches(img_f32: np.ndarray) -> None:
    assert_exact(
        M.adjust_brightness_torchvision(img_f32, 1), PF.adjust_brightness_torchvision(img_f32, 1)
    )
    assert_exact(
        M.adjust_brightness_torchvision(img_f32, 0), PF.adjust_brightness_torchvision(img_f32, 0)
    )


@pytest.mark.parametrize("factor", [0.3, 1.0, 2.0])
def test_adjust_contrast_f32(img_f32: np.ndarray, factor: float) -> None:
    if factor == 1.0:
        assert_exact(
            M.adjust_contrast_torchvision(img_f32, factor),
            PF.adjust_contrast_torchvision(img_f32, factor),
        )
        return
    assert_close(
        M.adjust_contrast_torchvision(img_f32, factor),
        PF.adjust_contrast_torchvision(img_f32, factor),
        1e-6,
    )


@pytest.mark.parametrize("factor", [0.3, 2.0])
def test_adjust_contrast_u8(img_u8: np.ndarray, factor: float) -> None:
    assert_close(
        M.adjust_contrast_torchvision(img_u8, factor),
        PF.adjust_contrast_torchvision(img_u8, factor),
        2.0,
    )


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.5, 2.0])
def test_adjust_saturation_f32(img_f32: np.ndarray, factor: float) -> None:
    assert_close(
        M.adjust_saturation_torchvision(img_f32, factor),
        PF.adjust_saturation_torchvision(img_f32, factor),
        2e-6,
    )


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.5])
def test_adjust_saturation_u8(img_u8: np.ndarray, factor: float) -> None:
    assert_close(
        M.adjust_saturation_torchvision(img_u8, factor),
        PF.adjust_saturation_torchvision(img_u8, factor),
        2.0,
    )


# ---------------------------------------------------------------------------
# blur-backed pixel transforms
# ---------------------------------------------------------------------------


def test_unsharp_mask_f32(img_f32: np.ndarray) -> None:
    assert_close(
        M.unsharp_mask(img_f32, 5, 1.0, 0.5, 3),
        PF.unsharp_mask(img_f32, 5, 1.0, 0.5, 3),
        2e-4,
    )


def test_sharpen_gaussian_f32(img_f32: np.ndarray) -> None:
    assert_close(
        M.sharpen_gaussian(img_f32, 1.5, 5, 1.2),
        PF.sharpen_gaussian(img_f32, 1.5, 5, 1.2),
        2e-4,
    )


def test_add_noise_f32(img_f32: np.ndarray, rng: np.random.Generator) -> None:
    noise = rng.normal(0, 0.1, img_f32.shape).astype(np.float32)
    assert_exact(M.add_noise(img_f32, noise), PF.add_noise(img_f32, noise))


def test_apply_salt_and_pepper_u8_is_bit_exact(
    img_u8: np.ndarray, rng: np.random.Generator
) -> None:
    salt = rng.integers(0, 2, img_u8.shape[:2]).astype(bool)
    pepper = (rng.integers(0, 2, img_u8.shape[:2]) & ~salt).astype(bool)
    assert_exact(
        M.apply_salt_and_pepper(img_u8, salt, pepper),
        PF.apply_salt_and_pepper(img_u8, salt, pepper),
    )


# ---------------------------------------------------------------------------
# illumination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0, 90, 180, 270, 45, 135, 225, 315, 37.0])
def test_create_directional_gradient(rng: np.random.Generator, angle: float) -> None:
    got = M.create_directional_gradient(32, 48, angle)
    want = PF.create_directional_gradient(32, 48, angle)
    assert got.dtype == want.dtype
    assert_close(got, want, 1e-6)


@pytest.mark.parametrize("intensity", [0.2, -0.3])
@pytest.mark.parametrize("angle", [30.0, 200.0])
def test_apply_linear_illumination(
    img_f32: np.ndarray, intensity: float, angle: float
) -> None:
    assert_close(
        M.apply_linear_illumination(img_f32, intensity, angle),
        PF.apply_linear_illumination(img_f32, intensity, angle),
        1e-6,
    )


@pytest.mark.parametrize("sigma", [0.1, 0.4])
def test_apply_gaussian_illumination(img_f32: np.ndarray, sigma: float) -> None:
    assert_close(
        M.apply_gaussian_illumination(img_f32, 0.5, (0.5, 0.5), sigma),
        PF.apply_gaussian_illumination(img_f32, 0.5, (0.5, 0.5), sigma),
        1e-6,
    )


def test_apply_gaussian_illumination_zero_intensity(img_f32: np.ndarray) -> None:
    assert_exact(
        M.apply_gaussian_illumination(img_f32, 0, (0.5, 0.5), 0.2),
        PF.apply_gaussian_illumination(img_f32, 0, (0.5, 0.5), 0.2),
    )


# ---------------------------------------------------------------------------
# auto_contrast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", [0, 5])
@pytest.mark.parametrize("method", ["cdf", "pil"])
@pytest.mark.parametrize("ignore", [None, 1])
def test_auto_contrast_is_bit_exact(
    img_u8: np.ndarray, cutoff: int, method: str, ignore: int | None
) -> None:
    assert_exact(
        M.auto_contrast(img_u8, cutoff, ignore, method),
        PF.auto_contrast(img_u8, cutoff, ignore, method),
    )


def test_auto_contrast_flat_histogram_uses_special_case() -> None:
    flat = np.full((16, 16, 3), 7, dtype=np.uint8)
    assert_exact(
        M.auto_contrast(flat, 10, None, "cdf"), PF.auto_contrast(flat, 10, None, "cdf")
    )


@pytest.mark.parametrize("cutoff", [0, 1, 5, 20])
def test_get_histogram_bounds_matches(cutoff: int) -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
        assert M.get_histogram_bounds(hist, cutoff) == PF.get_histogram_bounds(
            hist, cutoff
        )


@pytest.mark.parametrize("method", ["cdf", "pil"])
@pytest.mark.parametrize(("lo", "hi"), [(0, 255), (10, 200), (100, 100), (250, 250)])
def test_create_contrast_lut_matches(rng: np.random.Generator, method: str, lo: int, hi: int) -> None:
    hist = rng.integers(0, 500, 256).astype(np.float64)
    assert_exact(
        M.create_contrast_lut(hist, lo, hi, 255, method),
        PF.create_contrast_lut(hist, lo, hi, 255, method),
    )


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def test_rgb_to_optical_density(img_u8: np.ndarray, img_f32: np.ndarray) -> None:
    assert_close(
        M.rgb_to_optical_density(img_u8), PF.rgb_to_optical_density(img_u8), 1e-6
    )
    assert_close(
        M.rgb_to_optical_density(img_f32), PF.rgb_to_optical_density(img_f32), 1e-6
    )


@pytest.mark.parametrize("shape", [(3, 3), (5, 5), (1, 7)])
def test_convolve_f32(img_f32: np.ndarray, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(11)
    kernel = rng.normal(0, 0.3, shape).astype(np.float32)
    assert_close(M.convolve(img_f32, kernel), PF.convolve(img_f32, kernel), 5e-4)


@pytest.mark.parametrize("ksize", [3, 5, 9])
def test_separable_convolve_f32(img_f32: np.ndarray, ksize: int) -> None:
    rng = np.random.default_rng(13)
    kernel = rng.normal(0, 0.2, ksize).astype(np.float32)
    assert_close(
        M.separable_convolve(img_f32, kernel), PF.separable_convolve(img_f32, kernel), 5e-4
    )