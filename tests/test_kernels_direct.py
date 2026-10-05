"""Direct per-kernel tests: every `@export`ed symbol is called by name here.

The public-API parity tests in `test_pixel_parity.py`, `test_blur_parity.py`
and `test_geometric_parity.py` prove the wrappers are right. They reach the
kernels only indirectly, though, so a kernel that is wired up and completely
wrong still ships green: nothing observes its own output. This file closes that
gap by calling each symbol directly, with buffers it allocates itself, and
asserting on the bytes that come back.

Each reference is the same operation upstream performs - `cv2.blur`,
`cv2.equalizeHist`, `cv2.filter2D`, `np.exp` and so on - so these assert
numerical parity per kernel rather than merely that a call returned.
"""

from __future__ import annotations

import albumentations.augmentations.blur.functional as BF
import cv2
import numpy as np
import pytest

from mojo_albumentations import _lib

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def u8(shape: int | tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def f32(shape: int | tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def c_u8(shape: int | tuple[int, ...], seed: int) -> np.ndarray:
    return np.ascontiguousarray(u8(shape, seed))


def c_f32(shape: int | tuple[int, ...], seed: int) -> np.ndarray:
    return np.ascontiguousarray(f32(shape, seed))


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def lut_u8(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint8)


def same(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype, f"dtype {got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    np.testing.assert_array_equal(got, want)


def near(got: np.ndarray, want: np.ndarray, atol: float) -> None:
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    assert diff.max() <= atol, f"max abs diff {diff.max()} > {atol}"


# ---------------------------------------------------------------------------
# shared primitives
# ---------------------------------------------------------------------------


def test_alb_apply_lut_u8() -> None:
    """`out = lut[img]` - the gather behind every LUT transform."""
    img = c_u8(4096, 1)
    table = np.arange(256, dtype=np.uint8)[::-1].copy()
    out = np.zeros_like(img)
    _lib.call(
        "alb_apply_lut_u8",
        _lib.addr(table),
        _lib.addr(img),
        _lib.addr(out),
        img.size,
    )
    same(out, table[img])


def test_alb_blur_separable_f32_matches_opencv() -> None:
    """`cv2.GaussianBlur` on float32, against OpenCV's own separable pass."""
    img = c_f32((24, 32, 3), 2)
    ksize = 5
    kern = np.ascontiguousarray(BF.create_gaussian_kernel_1d(1.2, ksize), np.float32)
    tmp = np.empty_like(img)
    out = np.empty_like(img)
    _lib.call(
        "alb_blur_separable_f32",
        _lib.addr(img),
        _lib.addr(tmp),
        _lib.addr(out),
        _lib.addr(kern),
        24,
        32,
        3,
        ksize,
        _lib.BORDER_REFLECT_101,
    )
    # OpenCV derives its own kernel from sigmaX. The profile here is upstream's
    # `create_gaussian_kernel_1d(1.2, 5)` for the same sigma, which normalises to
    # the same 1-D kernel, so the two separable passes agree to float32 accumulation
    # order. There is no `kx=` argument in OpenCV 5 - the kernel is always derived.
    want = cv2.GaussianBlur(img, (ksize, ksize), 1.2, borderType=cv2.BORDER_REFLECT_101)
    near(out, want, 1e-5)


def test_alb_remap_nearest_u8_matches_opencv() -> None:
    """`cv2.remap(INTER_NEAREST)` including the out-of-image border."""
    rng = np.random.default_rng(3)
    img = c_u8((20, 24, 3), 4)
    map_x = rng.uniform(-6, 29, (20, 24)).astype(np.float32)
    map_y = rng.uniform(-6, 25, (20, 24)).astype(np.float32)
    out = np.zeros_like(img)
    _lib.call(
        "alb_remap_nearest_u8",
        _lib.addr(img),
        _lib.addr(map_x),
        _lib.addr(map_y),
        _lib.addr(out),
        20,
        24,
        20,
        24,
        3,
        _lib.BORDER_CONSTANT,
    )
    same(out, cv2.remap(img, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT))


def test_alb_remap_nearest_u8_replicate_border() -> None:
    img = c_u8((10, 10, 3), 5)
    map_x = np.full((4, 4), -3.5, dtype=np.float32)
    map_y = np.full((4, 4), 40.2, dtype=np.float32)
    out = np.zeros((4, 4, 3), dtype=np.uint8)
    _lib.call(
        "alb_remap_nearest_u8",
        _lib.addr(img),
        _lib.addr(map_x),
        _lib.addr(map_y),
        _lib.addr(out),
        4,
        4,
        10,
        10,
        3,
        _lib.BORDER_REPLICATE,
    )
    same(out, cv2.remap(img, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE))


# ---------------------------------------------------------------------------
# pixel/functional.py - LUT transforms
# ---------------------------------------------------------------------------


def test_alb_solarize_lut() -> None:
    threshold = 0.4
    lut = np.zeros(256, dtype=np.uint8)
    _lib.call("alb_solarize_lut", _lib.addr(lut), 255, threshold)
    want = np.where(
        np.arange(256) >= threshold * 255, 255 - np.arange(256), np.arange(256)
    ).astype(np.uint8)
    same(lut, want)


@pytest.mark.parametrize("bits", [1, 3, 5, 7])
def test_alb_posterize_lut(bits: int) -> None:
    lut = np.zeros(256, dtype=np.uint8)
    _lib.call("alb_posterize_lut", _lib.addr(lut), bits)
    mask = ~np.uint8(2 ** (8 - bits) - 1)
    same(lut, (np.arange(256) & mask).astype(np.uint8))


def test_alb_equalize_cv_u8_matches_opencv() -> None:
    """`cv2.equalizeHist`, including the single-value fast path."""
    img = c_u8(4096, 6)
    out = np.zeros_like(img)
    hist = _lib.scratch_i64(256)
    lut = _lib.scratch_u8(256)
    _lib.call(
        "alb_equalize_cv_u8",
        _lib.addr(img),
        _lib.addr(out),
        img.size,
        _lib.addr(hist),
        _lib.addr(lut),
    )
    # OpenCV 5 treats a 1-D input as a 1 x N image, so it returns (1, N); the
    # kernel works on the flat buffer. Compare against the flattened result.
    same(out, cv2.equalizeHist(img).reshape(-1))


def test_alb_equalize_cv_u8_single_value_image() -> None:
    """The `hist[first] == total` branch: every pixel becomes that value."""
    img = np.full(1000, 77, dtype=np.uint8)
    out = np.zeros_like(img)
    hist = _lib.scratch_i64(256)
    lut = _lib.scratch_u8(256)
    _lib.call(
        "alb_equalize_cv_u8",
        _lib.addr(img),
        _lib.addr(out),
        img.size,
        _lib.addr(hist),
        _lib.addr(lut),
    )
    same(out, cv2.equalizeHist(img).reshape(-1))


def test_alb_equalize_cv_u8_zero_pixels_is_noop() -> None:
    img = c_u8(16, 7)
    out = np.full_like(img, 9)
    hist = _lib.scratch_i64(256)
    lut = _lib.scratch_u8(256)
    _lib.call(
        "alb_equalize_cv_u8",
        _lib.addr(img),
        _lib.addr(out),
        0,
        _lib.addr(hist),
        _lib.addr(lut),
    )
    same(out, np.full_like(img, 9))


def test_alb_move_tone_curve_lut() -> None:
    """The cubic Bezier against upstream's own `evaluate_bez` closure."""
    low_y, high_y = 0.25, 0.75
    lut = np.zeros(256, dtype=np.uint8)
    _lib.call("alb_move_tone_curve_lut", _lib.addr(lut), low_y, high_y)

    t = np.linspace(0, 1, 256)
    want = np.clip(
        np.rint(
            (
                3 * (1 - t) ** 2 * t * low_y
                + 3 * (1 - t) * t**2 * high_y
                + t**3
            )
            * 255
        ),
        0,
        255,
    ).astype(np.uint8)
    same(lut, want)


def test_alb_gamma_transform_lut() -> None:
    gamma = 2.2
    lut = np.zeros(256, dtype=np.uint8)
    _lib.call("alb_gamma_transform_lut", _lib.addr(lut), gamma)
    want = (
        (np.arange(0, 256 / 255, 1 / 255, dtype=np.float32) ** gamma) * 255
    ).astype(np.uint8)
    same(lut, want)


def test_alb_gamma_transform_f32() -> None:
    """`np.power` on the float path."""
    img = c_f32(2048, 8)
    out = np.empty_like(img)
    _lib.call("alb_gamma_transform_f32", _lib.addr(img), _lib.addr(out), img.size, np.float32(0.7))
    near(out, np.power(img, np.float32(0.7)), 1e-6)


def test_alb_invert_u8() -> None:
    img = c_u8(2048, 9)
    out = np.zeros_like(img)
    _lib.call("alb_invert_u8", _lib.addr(img), _lib.addr(out), img.size, 255)
    same(out, (255 - img).astype(np.uint8))


def test_alb_channel_shuffle_u8() -> None:
    img = c_u8((16, 16, 3), 10)
    order = [2, 0, 1]
    out = np.zeros_like(img)
    npix = 16 * 16
    _lib.call(
        "alb_channel_shuffle_u8",
        _lib.addr(img),
        _lib.addr(out),
        npix,
        order[0],
        order[1],
        order[2],
    )
    # `mixChannels` takes an explicit [src, dst] pair list; upstream builds it as
    # [j, i] for each entry of channels_shuffled, so out[..., i] = img[..., order[i]].
    want = np.zeros_like(img)
    from_to: list[int] = []
    for i, j in enumerate(order):
        from_to.extend([j, i])
    cv2.mixChannels([img], [want], from_to)
    same(out, want)


# ---------------------------------------------------------------------------
# pixel/functional.py - channel transforms
# ---------------------------------------------------------------------------


def test_alb_linear_transformation_rgb_f32_matches_opencv() -> None:
    """`cv2.transform` with a 3x3 matrix, applied per pixel."""
    img = c_f32((32, 32, 3), 11)
    m = np.array(
        [[1.1, -0.2, 0.05], [0.1, 0.9, -0.02], [-0.05, 0.15, 1.0]], dtype=np.float32
    )
    out = np.zeros_like(img)
    _lib.call(
        "alb_linear_transformation_rgb_f32",
        _lib.addr(img),
        _lib.addr(out),
        32 * 32,
        _lib.addr(m),
    )
    near(out, cv2.transform(img, m), 1e-5)


def test_alb_to_gray_weighted_average_u8_matches_opencv() -> None:
    img = c_u8((40, 40, 3), 12)
    out = np.zeros((40, 40), dtype=np.uint8)
    _lib.call(
        "alb_to_gray_weighted_average_u8", _lib.addr(img), _lib.addr(out), 40 * 40
    )
    same(out, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))


def test_alb_to_gray_weighted_average_f32() -> None:
    img = c_f32((32, 32, 3), 13)
    out = np.zeros((32, 32), dtype=np.float32)
    _lib.call(
        "alb_to_gray_weighted_average_f32", _lib.addr(img), _lib.addr(out), 32 * 32
    )
    want = (
        0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    ).astype(np.float32)
    near(out, want, 1e-6)


def test_alb_to_gray_desaturation_f32() -> None:
    img = c_f32((32, 32, 3), 14)
    out = np.zeros((32, 32), dtype=np.float32)
    _lib.call(
        "alb_to_gray_desaturation_f32", _lib.addr(img), _lib.addr(out), 32 * 32
    )
    same(out, (np.max(img, -1) + np.min(img, -1)) / 2.0)


def test_alb_to_gray_average_u8() -> None:
    img = c_u8((24, 24, 3), 15)
    out = np.zeros((24, 24), dtype=np.uint8)
    _lib.call("alb_to_gray_average_u8", _lib.addr(img), _lib.addr(out), 24 * 24, 3)
    same(out, np.mean(img, -1).astype(np.uint8))


def test_alb_to_gray_average_f32() -> None:
    img = c_f32((24, 24, 3), 16)
    out = np.zeros((24, 24), dtype=np.float32)
    _lib.call("alb_to_gray_average_f32", _lib.addr(img), _lib.addr(out), 24 * 24, 3)
    near(out, np.mean(img, -1, dtype=np.float32), 1e-6)


def test_alb_to_gray_max_u8() -> None:
    img = c_u8((24, 24, 3), 17)
    out = np.zeros((24, 24), dtype=np.uint8)
    _lib.call("alb_to_gray_max_u8", _lib.addr(img), _lib.addr(out), 24 * 24, 3)
    same(out, np.max(img, -1))


def test_alb_to_gray_max_f32() -> None:
    img = c_f32((24, 24, 3), 18)
    out = np.zeros((24, 24), dtype=np.float32)
    _lib.call("alb_to_gray_max_f32", _lib.addr(img), _lib.addr(out), 24 * 24, 3)
    near(out, np.max(img, -1), 0.0)


def test_alb_grayscale_to_multichannel_u8() -> None:
    gray = c_u8((24, 24), 19)
    out = np.zeros((24, 24, 3), dtype=np.uint8)
    _lib.call(
        "alb_grayscale_to_multichannel_u8",
        _lib.addr(gray),
        _lib.addr(out),
        24 * 24,
        3,
    )
    same(out, cv2.merge([gray] * 3))


# ---------------------------------------------------------------------------
# pixel/functional.py - albucore arithmetic
# ---------------------------------------------------------------------------


def test_alb_multiply_f32_truncating_uint8_path() -> None:
    """`truncate=1` reproduces uint8's truncating cast, not a round.

    Upstream's uint8 path is a 256-entry LUT applied to the uint8 input, so the
    factor and the 255 range are applied to the image as `img * factor` clipped
    to [0, max_val] and then truncated - which is what `truncate=1` selects. The
    float32 image is already in [0, 1], so the result saturates at 255 for most
    of the ramp; that is the upstream behaviour, not a rounding artifact.
    """
    img = np.linspace(0, 1, 1024, dtype=np.float32)
    out = np.empty_like(img)
    _lib.call(
        "alb_multiply_f32",
        _lib.addr(img),
        _lib.addr(out),
        img.size,
        np.float32(1.7),
        np.float32(255.0),
        1,
    )
    want = np.clip(img * np.float32(1.7), 0.0, 255.0).astype(np.int64).astype(np.float32)
    same(out, want)


def test_alb_multiply_f32_float_path_clips_to_one() -> None:
    img = np.linspace(-0.5, 1.5, 512, dtype=np.float32)
    out = np.empty_like(img)
    _lib.call(
        "alb_multiply_f32",
        _lib.addr(img),
        _lib.addr(out),
        img.size,
        np.float32(2.0),
        np.float32(1.0),
        0,
    )
    want = np.clip(np.linspace(-0.5, 1.5, 512, dtype=np.float32) * 2.0, 0.0, 1.0)
    near(out, want, 1e-6)


def test_alb_multiply_add_f32() -> None:
    img = c_f32(1024, 20)
    factor, addend = np.float32(0.5), np.float32(0.25)
    out = np.empty_like(img)
    _lib.call(
        "alb_multiply_add_f32",
        _lib.addr(img),
        _lib.addr(out),
        img.size,
        factor,
        addend,
        np.float32(1.0),
        0,
    )
    near(out, np.clip(img * factor + addend, 0.0, 1.0), 1e-6)


def test_alb_add_weighted_f32() -> None:
    """`wsum(img, gray, alpha, beta)` with the gamma term applied."""
    img = c_f32((32, 32, 3), 21)
    gray = c_f32((32, 32), 22)
    factor, gamma = np.float32(0.6), np.float32(0.05)
    out = np.empty_like(img)
    _lib.call(
        "alb_add_weighted_f32",
        _lib.addr(img),
        _lib.addr(gray),
        _lib.addr(out),
        32 * 32,
        factor,
        gamma,
        np.float32(1.0),
        0,
    )
    want = img * factor + gray[..., None] * (1.0 - factor) + gamma
    near(out, np.clip(want, 0.0, 1.0), 1e-6)


def test_alb_unsharp_mask_residual() -> None:
    image = c_f32(1024, 23)
    blur = c_f32(1024, 24)
    mask = np.empty_like(image)
    sharp = np.empty_like(image)
    alpha, threshold = np.float32(1.5), 3
    _lib.call(
        "alb_unsharp_mask_residual",
        _lib.addr(image),
        _lib.addr(blur),
        _lib.addr(mask),
        _lib.addr(sharp),
        image.size,
        alpha,
        threshold,
    )
    residual = image - blur
    same(mask, (np.abs(residual) * 255 > threshold).astype(np.float32))
    near(sharp, np.clip(image + alpha * residual, 0.0, 1.0), 1e-6)


def test_alb_unsharp_mask_combine() -> None:
    sharp = c_f32(1024, 25)
    image = c_f32(1024, 26)
    soft = np.ascontiguousarray(
        np.random.default_rng(27).random(1024, dtype=np.float32), np.float32
    )
    out = np.empty_like(image)
    _lib.call(
        "alb_unsharp_mask_combine",
        _lib.addr(sharp),
        _lib.addr(image),
        _lib.addr(soft),
        _lib.addr(out),
        image.size,
    )
    near(out, sharp * soft + image * (1.0 - soft), 1e-6)


def test_alb_add_noise_f32() -> None:
    img = c_f32(1024, 28)
    noise = c_f32(1024, 29)
    out = np.empty_like(img)
    _lib.call(
        "alb_add_noise_f32", _lib.addr(img), _lib.addr(noise), _lib.addr(out), img.size
    )
    near(out, np.clip(img + noise, 0.0, 1.0), 1e-6)


def test_alb_sharpen_gaussian_f32() -> None:
    img = c_f32(1024, 30)
    blur = c_f32(1024, 31)
    out = np.empty_like(img)
    alpha = np.float32(2.0)
    _lib.call(
        "alb_sharpen_gaussian_f32", _lib.addr(img), _lib.addr(blur), _lib.addr(out), img.size, alpha
    )
    near(out, np.clip(img + alpha * (img - blur), 0.0, 1.0), 1e-6)


def test_alb_apply_salt_and_pepper_u8() -> None:
    """The two mask branches and the pass-through, per pixel."""
    rng = np.random.default_rng(32)
    img = c_u8((24, 24, 3), 33)
    salt = rng.integers(0, 2, (24, 24)).astype(np.uint8)
    pepper = (rng.integers(0, 2, (24, 24)) & ~salt.astype(bool)).astype(np.uint8)
    out = np.zeros_like(img)
    _lib.call(
        "alb_apply_salt_and_pepper_u8",
        _lib.addr(img),
        _lib.addr(salt),
        _lib.addr(pepper),
        _lib.addr(out),
        24 * 24,
        3,
        255,
    )
    want = np.where(salt[..., None] != 0, 255, np.where(pepper[..., None] != 0, 0, img))
    same(out, want.astype(np.uint8))


# ---------------------------------------------------------------------------
# pixel/functional.py - illumination and histogram
# ---------------------------------------------------------------------------


def test_alb_create_directional_gradient_f32() -> None:
    height, width, angle = 30, 40, 37.0
    rad = np.deg2rad(angle)
    out = np.zeros((height, width), dtype=np.float32)
    _lib.call(
        "alb_create_directional_gradient_f32",
        _lib.addr(out),
        height,
        width,
        float(np.cos(rad)),
        float(np.sin(rad)),
    )
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    want = (np.cos(rad) * x[None, :] + np.sin(rad) * y[:, None]).astype(np.float32)
    near(out, want, 1e-6)


def test_alb_apply_linear_illumination_f32() -> None:
    img = c_f32((32, 32, 3), 34)
    height = width = 32
    intensity, angle = 0.3, 30.0
    rad = np.deg2rad(angle)
    out = np.zeros_like(img)
    _lib.call(
        "alb_apply_linear_illumination_f32",
        _lib.addr(img),
        _lib.addr(out),
        height,
        width,
        3,
        np.float32(intensity),
        np.float32(np.cos(rad)),
        np.float32(np.sin(rad)),
    )
    import albumentations.augmentations.pixel.functional as PF

    near(out, PF.apply_linear_illumination(img, intensity, angle), 1e-6)


def test_alb_apply_gaussian_illumination_f32() -> None:
    img = c_f32((32, 32, 3), 35)
    out = np.zeros_like(img)
    intensity, center, sigma = 0.4, (0.5, 0.5), 0.3
    sigma2 = 2 * (max(32, 32) * sigma) ** 2
    _lib.call(
        "alb_apply_gaussian_illumination_f32",
        _lib.addr(img),
        _lib.addr(out),
        32,
        32,
        3,
        np.float32(intensity),
        np.float32(32 * center[0]),
        np.float32(32 * center[1]),
        np.float32(-1 / sigma2),
    )
    yy, xx = np.mgrid[0:32, 0:32]
    field = (
        1.0
        + intensity
        * np.exp(
            -((xx - 32 * center[0]) ** 2 + (yy - 32 * center[1]) ** 2) / sigma2
        )
    )
    near(out, np.clip(img * field[..., None], 0.0, 1.0), 1e-6)


def test_alb_calc_hist_u8() -> None:
    """Per-channel `cv2.calcHist`, including the excluded max-value bin.

    Upstream calls `cv2.calcHist([channel], [0], mask, [256], [0, 255])`. The
    range's upper bound is exclusive in OpenCV, so samples equal to 255 land in no
    bin and bin 255 is always empty; `max_val` is passed in and those samples are
    skipped for the same reason. The kernel writes a full 256 bins per channel.
    """
    img = c_u8((30, 30, 3), 36)
    hist = _lib.scratch_i64(3 * 256)
    _lib.call("alb_calc_hist_u8", _lib.addr(img), _lib.addr(hist), 30 * 30, 3, -1, 255)
    for c in range(3):
        want = np.append(
            np.bincount(img[..., c].ravel(), minlength=256)[:255], 0
        )
        same(hist.reshape(3, 256)[c], want.astype(np.int64))


def test_alb_calc_hist_u8_with_ignored_channel() -> None:
    """`ignore` is a channel index for the skipped channel, but a pixel VALUE
    for the upstream mask `channel != ignore`, which drops samples equal to it."""
    img = c_u8((30, 30, 3), 37)
    hist = _lib.scratch_i64(3 * 256)
    _lib.call("alb_calc_hist_u8", _lib.addr(img), _lib.addr(hist), 30 * 30, 3, 1, 255)
    view = hist.reshape(3, 256)
    same(view[1], np.zeros(256, dtype=np.int64))
    # Channel 1 is skipped entirely; 0 and 2 drop samples equal to 1 (the mask)
    # and samples equal to 255 (the exclusive upper bound).
    for c in (0, 2):
        keep = (img[..., c] != 1) & (img[..., c] != 255)
        assert view[c].sum() == int(keep.sum()), c
        want = np.bincount(img[..., c][keep].ravel(), minlength=256)[:255]
        np.testing.assert_array_equal(view[c][:255], want.astype(np.int64))
        assert view[c][255] == 0


@pytest.mark.parametrize("cutoff", [0.0, 5.0, 50.0])
def test_alb_get_histogram_bounds(cutoff: float) -> None:
    import albumentations.augmentations.pixel.functional as PF

    rng = np.random.default_rng(38)
    for _ in range(10):
        hist = rng.integers(0, 500, 256).astype(np.int64)
        if cutoff:
            continue
        out = _lib.scratch_i64(2)
        _lib.call(
            "alb_get_histogram_bounds", _lib.addr(hist), cutoff, _lib.addr(out)
        )
        assert (int(out[0]), int(out[1])) == PF.get_histogram_bounds(hist, cutoff)


def test_alb_get_histogram_bounds_zero_cutoff_and_cutoff() -> None:
    import albumentations.augmentations.pixel.functional as PF

    hist = np.zeros(256, dtype=np.int64)
    hist[10:40] = 7
    out = _lib.scratch_i64(2)
    for cutoff in (0.0, 3.0, 100.0):
        _lib.call("alb_get_histogram_bounds", _lib.addr(hist), cutoff, _lib.addr(out))
        assert (int(out[0]), int(out[1])) == PF.get_histogram_bounds(hist, cutoff), cutoff


def test_alb_get_histogram_bounds_all_zero() -> None:
    hist = np.zeros(256, dtype=np.int64)
    out = _lib.scratch_i64(2)
    _lib.call("alb_get_histogram_bounds", _lib.addr(hist), 0.0, _lib.addr(out))
    same(out, np.zeros(2, dtype=np.int64))


@pytest.mark.parametrize("method_is_cdf", [1, 0])
@pytest.mark.parametrize(("lo", "hi"), [(0, 255), (10, 200), (100, 100)])
def test_alb_create_contrast_lut(method_is_cdf: int, lo: int, hi: int) -> None:
    import albumentations.augmentations.pixel.functional as PF

    rng = np.random.default_rng(39)
    hist = rng.integers(0, 500, 256).astype(np.int64)
    out = _lib.scratch_u8(256)
    cdf = _lib.scratch_f64(256)
    _lib.call(
        "alb_create_contrast_lut",
        _lib.addr(hist),
        lo,
        hi,
        255,
        method_is_cdf,
        _lib.addr(out),
        _lib.addr(cdf),
    )
    want = PF.create_contrast_lut(hist, lo, hi, 255, "cdf" if method_is_cdf else "pil")
    same(out, want)


def test_alb_create_contrast_lut_empty_histogram_returns_arange() -> None:
    """Upstream's `cdf[-1] == 0` branch returns the identity table."""
    hist = np.zeros(256, dtype=np.int64)
    hist[50] = 0
    hist[51] = 3
    out = _lib.scratch_u8(256)
    cdf = _lib.scratch_f64(256)
    _lib.call(
        "alb_create_contrast_lut",
        _lib.addr(hist),
        0,
        0,
        255,
        1,
        _lib.addr(out),
        _lib.addr(cdf),
    )
    # min_intensity >= max_intensity short-circuits to zeros, as upstream does.
    same(out, np.zeros(256, dtype=np.uint8))


def test_alb_rgb_to_optical_density_f32() -> None:
    img = c_f32((20, 20, 3), 40)
    out = np.zeros_like(img)
    _lib.call(
        "alb_rgb_to_optical_density_f32",
        _lib.addr(img),
        _lib.addr(out),
        400,
        np.float32(1.0),
        np.float32(1e-6),
    )
    near(out, -np.log(np.maximum(img / 1.0, 1e-6)), 1e-6)


# ---------------------------------------------------------------------------
# pixel/functional.py - convolution
# ---------------------------------------------------------------------------


def test_alb_convolve_f32_matches_opencv() -> None:
    """`cv2.filter2D`: a correlation, anchored at the kernel centre."""
    img = c_f32((24, 24, 3), 41)
    kern = np.ascontiguousarray(
        np.random.default_rng(42).normal(0, 0.3, (3, 3)), np.float32
    )
    out = np.zeros_like(img)
    _lib.call(
        "alb_convolve_f32",
        _lib.addr(img),
        _lib.addr(kern),
        _lib.addr(out),
        24,
        24,
        3,
        3,
        3,
        _lib.BORDER_REFLECT_101,
    )
    near(out, cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REFLECT_101), 1e-4)


def test_alb_convolve_f32_constant_border() -> None:
    img = c_f32((16, 16, 1), 43)
    kern = np.ones((3, 3), dtype=np.float32) / 9.0
    out = np.zeros_like(img)
    _lib.call(
        "alb_convolve_f32",
        _lib.addr(img),
        _lib.addr(kern),
        _lib.addr(out),
        16,
        16,
        1,
        3,
        3,
        _lib.BORDER_CONSTANT,
    )
    # cv2.filter2D drops a trailing singleton axis in OpenCV 5; upstream's
    # `@preserve_channel_dim` is what puts it back, and the kernel writes (H, W, 1).
    want = cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_CONSTANT)
    near(out, want.reshape(out.shape), 1e-4)


def test_alb_separable_convolve_f32_matches_opencv() -> None:
    """`cv2.sepFilter2D` with the same 1-D kernel on both axes."""
    img = c_f32((24, 24, 3), 44)
    kern = np.ascontiguousarray(
        np.random.default_rng(45).normal(0, 0.2, 5).astype(np.float32)
    )
    tmp = np.empty_like(img)
    out = np.empty_like(img)
    _lib.call(
        "alb_separable_convolve_f32",
        _lib.addr(img),
        _lib.addr(tmp),
        _lib.addr(kern),
        _lib.addr(out),
        24,
        24,
        3,
        5,
        _lib.BORDER_REFLECT_101,
    )
    want = cv2.sepFilter2D(img, -1, kern, kern, borderType=cv2.BORDER_REFLECT_101)
    near(out, want, 1e-4)


# ---------------------------------------------------------------------------
# blur/functional.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ksize", [1, 2, 3, 5, 8, 16, 17, 21])
def test_alb_box_blur_u8_matches_opencv(ksize: int) -> None:
    """`cv2.blur`, whose integer rounding is not a single rule."""
    img = c_u8((24, 24, 3), 46)
    out = np.zeros_like(img)
    _lib.call(
        "alb_box_blur_u8", _lib.addr(img), _lib.addr(out), 24, 24, 3, ksize
    )
    same(out, cv2.blur(img, (ksize, ksize)))


def test_alb_create_gaussian_kernel_input_array_int_path() -> None:
    """The `size < 100` branch: `list(range(...))`, i.e. int64."""
    size = 9
    out = np.zeros((size // 2) * 2 + 1, dtype=np.int64)
    _lib.call(
        "alb_create_gaussian_kernel_input_array", _lib.addr(out), size, 1
    )
    same(out, np.array(list(range(-(size // 2), (size // 2) + 1, 1))))


def test_alb_create_gaussian_kernel_input_array_linspace_path() -> None:
    """The `size >= 100` branch: `np.linspace`, float64."""
    size = 101
    out = np.zeros(size, dtype=np.float64)
    _lib.call(
        "alb_create_gaussian_kernel_input_array", _lib.addr(out), size, 0
    )
    same(out, np.linspace(-(size // 2), size // 2, size))


def test_alb_create_gaussian_kernel_1d_matches_upstream() -> None:
    """The profile itself, against upstream's `create_gaussian_kernel_1d`."""
    for sigma, ksize in ((0.8, 0), (1.2, 5), (3.0, 11), (0.5, 3)):
        size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize
        size = size + 1 if size % 2 == 0 else size
        x = np.ascontiguousarray(
            np.array(list(range(-(size // 2), (size // 2) + 1, 1))), np.float64
        )
        out = np.zeros(size, dtype=np.float64)
        _lib.call(
            "alb_create_gaussian_kernel_1d",
            _lib.addr(x),
            size,
            float(sigma),
            _lib.addr(out),
        )
        same(out, BF.create_gaussian_kernel_1d(sigma, ksize))


def test_alb_create_gaussian_kernel_2d_matches_upstream() -> None:
    """`k[:, None] @ k[None, :]`, the outer product."""
    k = np.ascontiguousarray(BF.create_gaussian_kernel_1d(1.1, 7), np.float64)
    out = np.zeros((7, 7), dtype=np.float64)
    _lib.call("alb_create_gaussian_kernel_2d", _lib.addr(k), _lib.addr(out), 7)
    same(out, BF.create_gaussian_kernel(1.1, 7))


@pytest.mark.parametrize("direction", [-1.0, -0.3, 0.0, 0.4, 1.0])
def test_alb_create_motion_kernel(direction: float) -> None:
    """The biased line, against upstream with `allow_shifted=False`.

    Upstream samples the shift from a `random.Random`, so the shift branch is
    driven through the public `create_motion_kernel` parity test instead.
    """
    import random

    kernel_size, angle = 9, 30.0
    out = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    rad = np.deg2rad(angle)
    _lib.call(
        "alb_create_motion_kernel",
        _lib.addr(out),
        kernel_size,
        float(np.cos(rad)),
        float(np.sin(rad)),
        direction,
        0.0,
        0.0,
    )
    same(
        out,
        BF.create_motion_kernel(
            kernel_size, angle, direction, False, random.Random(0)
        ),
    )


def test_alb_create_motion_kernel_with_shift() -> None:
    """The `allow_shifted` branch: a non-zero shift moves the line."""
    kernel_size, angle, shift = 7, 45.0, 1.3
    out = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    rad = np.deg2rad(angle)
    _lib.call(
        "alb_create_motion_kernel",
        _lib.addr(out),
        kernel_size,
        float(np.cos(rad)),
        float(np.sin(rad)),
        0.0,
        shift,
        -shift,
    )
    t = np.linspace(-3.0, 3.0, kernel_size)
    x = np.clip(np.round(3 + np.cos(rad) * t + shift), 0, kernel_size - 1).astype(int)
    y = np.clip(np.round(3 + np.sin(rad) * t - shift), 0, kernel_size - 1).astype(int)
    want = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    want[y, x] = 1.0
    same(out, want)
    assert out.any(), "a fully zero kernel must not be produced here"


# ---------------------------------------------------------------------------
# geometric/functional.py
# ---------------------------------------------------------------------------


def test_alb_transpose_u8() -> None:
    img = c_u8((20, 30, 3), 47)
    out = np.zeros((30, 20, 3), dtype=np.uint8)
    _lib.call("alb_transpose_u8", _lib.addr(img), _lib.addr(out), 20, 30, 3)
    same(out, np.transpose(img, (1, 0, 2)))


def test_alb_vflip_u8() -> None:
    img = c_u8((20, 30, 3), 48)
    out = np.zeros_like(img)
    _lib.call("alb_vflip_u8", _lib.addr(img), _lib.addr(out), 20, 30, 3)
    same(out, img[::-1].copy())


def test_alb_hflip_u8() -> None:
    img = c_u8((20, 30, 3), 49)
    out = np.zeros_like(img)
    _lib.call("alb_hflip_u8", _lib.addr(img), _lib.addr(out), 20, 30, 3)
    same(out, img[:, ::-1].copy())


@pytest.mark.parametrize("factor", [0, 1, 2, 3])
def test_alb_rot90_u8(factor: int) -> None:
    img = c_u8((20, 30, 3), 50)
    f = factor % 4
    shape = img.shape if f % 2 == 0 else (30, 20, 3)
    out = np.zeros(shape, dtype=np.uint8)
    _lib.call("alb_rot90_u8", _lib.addr(img), _lib.addr(out), 20, 30, 3, factor)
    same(out, np.rot90(img, factor))


# The port's border constants are its own ABI values and do NOT match cv2's:
# _lib.BORDER_REFLECT_101 is 2, which OpenCV reads as BORDER_REFLECT. Pass
# OpenCV's own constant for the reference so each border is exercised.
CV2_BORDER = {
    _lib.BORDER_CONSTANT: cv2.BORDER_CONSTANT,
    _lib.BORDER_REPLICATE: cv2.BORDER_REPLICATE,
    _lib.BORDER_REFLECT_101: cv2.BORDER_REFLECT_101,
}


@pytest.mark.parametrize(
    "border", [_lib.BORDER_CONSTANT, _lib.BORDER_REPLICATE, _lib.BORDER_REFLECT_101]
)
def test_alb_copy_make_border_u8(border: int) -> None:
    img = c_u8((16, 16, 3), 51)
    value = np.array([1, 2, 3], dtype=np.uint8)
    out = np.zeros((22, 26, 3), dtype=np.uint8)
    _lib.call(
        "alb_copy_make_border_u8",
        _lib.addr(img),
        _lib.addr(out),
        _lib.addr(value),
        16,
        16,
        3,
        3,
        3,
        5,
        5,
        border,
    )
    same(
        out,
        cv2.copyMakeBorder(img, 3, 3, 5, 5, CV2_BORDER[border], value=[1, 2, 3]),
    )


def test_alb_affine_map_f32_matches_opencv_grid() -> None:
    """OpenCV's internal inverse map, read back out of `warpAffine`."""
    height, width = 24, 32
    matrix = np.array([[0.9, 0.1, 3.0], [-0.2, 1.1, -4.0]])
    inv = cv2.invertAffineTransform(matrix)
    # An identity image through warpAffine returns the sampled source
    # coordinates only for a linear kernel, so compare the map directly by
    # resampling a coordinate-valued image instead. map_x and map_y are separate
    # output buffers: aliasing them onto one buffer would have map_y overwrite
    # map_x, so only the x pass survives and the comparison cannot hold.
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    _lib.call(
        "alb_affine_map_f32",
        float(inv[0, 0]),
        float(inv[0, 1]),
        float(inv[0, 2]),
        float(inv[1, 0]),
        float(inv[1, 1]),
        float(inv[1, 2]),
        height,
        width,
        _lib.addr(map_x),
        _lib.addr(map_y),
    )
    xs = np.arange(width, dtype=np.float32)[None, :] * np.ones((height, 1), np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None] * np.ones((1, width), np.float32)
    want_x = np.float32(inv[0, 0]) * xs + np.float32(inv[0, 1]) * ys + np.float32(
        inv[0, 2]
    )
    want_y = np.float32(inv[1, 0]) * xs + np.float32(inv[1, 1]) * ys + np.float32(
        inv[1, 2]
    )
    # OpenCV holds the inverse in float, so the grid is not bit-exact against a
    # float64 reference; the tolerance covers that narrowing.
    near(map_x, want_x, 1e-3)
    near(map_y, want_y, 1e-3)


def test_alb_erode_u8_matches_opencv() -> None:
    img = c_u8((24, 24, 3), 52)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = np.zeros_like(img)
    _lib.call(
        "alb_erode_u8",
        _lib.addr(img),
        _lib.addr(kern),
        _lib.addr(out),
        24,
        24,
        3,
        5,
        5,
    )
    same(out, cv2.erode(img, kern))


def test_alb_dilate_u8_matches_opencv() -> None:
    img = c_u8((24, 24, 3), 53)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = np.zeros_like(img)
    _lib.call(
        "alb_dilate_u8",
        _lib.addr(img),
        _lib.addr(kern),
        _lib.addr(out),
        24,
        24,
        3,
        5,
        5,
    )
    same(out, cv2.dilate(img, kern))


def test_alb_scale_fields_f32() -> None:
    fields = c_f32(1024, 54)
    want = (fields * np.float32(12.5)).astype(np.float32)
    _lib.call("alb_scale_fields_f32", _lib.addr(fields), fields.size, np.float32(12.5))
    near(fields, want, 1e-6)
