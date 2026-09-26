"""The image transforms whose inner loops are compiled.

Each function mirrors the corresponding
`albumentations.augmentations.*.functional` function - the LUT construction is
kept identical to upstream's so the semantics are the same, and only the
per-pixel pass is compiled. The transform *classes* are not reimplemented;
construct `A.Solarize(...)` etc. from the real package and hand the image here,
or call these directly.
"""

from __future__ import annotations

import numpy as np

from . import _lib
from ._lib import (
    BORDER_CONSTANT,
    BORDER_REPLICATE,
    GRAY_AVERAGE,
    GRAY_MAX,
    GRAY_WEIGHTED,
)

__all__ = [
    "BORDER_CONSTANT",
    "BORDER_REPLICATE",
    "GRAY_AVERAGE",
    "GRAY_MAX",
    "GRAY_WEIGHTED",
    "channel_shuffle",
    "desaturate",
    "gaussian_blur",
    "gaussian_kernel_1d",
    "invert",
    "mean_std",
    "posterize",
    "remap_nearest",
    "solarize",
    "to_gray",
    "warp_affine_map",
]


def _flat_u8(img) -> np.ndarray:
    arr = np.ascontiguousarray(img, dtype=np.uint8)
    return arr


# --------------------------------------------------------------------------
# LUT transforms


def solarize(img, threshold: float) -> np.ndarray:
    """`solarize`: invert every byte at or above ``threshold * 255``.

    Upstream builds exactly this 256-entry table, so the LUT is upstream's and
    only the gather is compiled.
    """
    max_val = 255
    cut = threshold * max_val
    lut = np.array(
        [max_val - i if i >= cut else i for i in range(max_val + 1)], dtype=np.uint8
    )
    return _lib.apply_lut(lut, img)


def posterize(img, bits) -> np.ndarray:
    """`posterize`: keep the ``bits`` highest bits of every byte."""
    bits_array = np.uint8(bits)
    if not bits_array.shape or bits_array.size == 1:
        lut = np.arange(0, 256, dtype=np.uint8)
        lut &= ~np.uint8(2 ** (8 - int(bits_array)) - 1)
        return _lib.apply_lut(lut, img)

    result = np.empty_like(np.ascontiguousarray(img, dtype=np.uint8))
    for i, channel_bits in enumerate(np.atleast_1d(bits_array)):
        lut = np.arange(0, 256, dtype=np.uint8)
        lut &= ~np.uint8(2 ** (8 - int(channel_bits)) - 1)
        result[..., i] = _lib.apply_lut(lut, np.ascontiguousarray(img)[..., i])
    return result


def invert(img) -> np.ndarray:
    """`invert`: ``255 - v`` for every byte."""
    return _lib.apply_lut(255 - np.arange(256, dtype=np.uint8), img)


def move_tone_curve(img, low_y, high_y) -> np.ndarray:
    """`move_tone_curve`: a cubic Bezier tone curve, which upstream already is
    a 256-entry LUT. The curve evaluation is upstream's; the gather is compiled.
    """
    t = np.linspace(0.0, 1.0, 256)
    one_minus_t = 1 - t

    def evaluate_bez(ty, lo, hi):
        return (
            3 * one_minus_t**2 * ty * lo + 3 * one_minus_t * ty**2 * hi + ty**3
        ) * 255

    if np.isscalar(low_y) and np.isscalar(high_y):
        lut = np.clip(np.rint(evaluate_bez(t, low_y, high_y)), 0, 255).astype(np.uint8)
        return _lib.apply_lut(lut, img)
    if isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        luts = np.stack(
            [
                np.clip(np.rint(evaluate_bez(t, lo, hi)), 0, 255).astype(np.uint8)
                for lo, hi in zip(low_y, high_y, strict=True)
            ]
        )
        source = np.ascontiguousarray(img, dtype=np.uint8)
        out = np.empty_like(source)
        for i in range(out.shape[2]):
            out[..., i] = _lib.apply_lut(
                np.ascontiguousarray(luts[i]), source[..., i]
            )
        return out
    raise TypeError(
        f"low_y and high_y must both be of type float or np.ndarray. Got "
        f"{type(low_y)} and {type(high_y)}"
    )


# --------------------------------------------------------------------------
# channel transforms


def to_gray(img, method: str = "weighted") -> np.ndarray:
    """`to_gray_average`, `to_gray_max` or `to_gray_weighted_average`."""
    modes = {
        "average": GRAY_AVERAGE,
        "max": GRAY_MAX,
        "weighted": GRAY_WEIGHTED,
        "desaturation": "desaturation",
    }
    if method not in modes:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(modes)}"
        )
    if modes[method] == "desaturation":
        return desaturate(img)
    if img.dtype == np.float32:
        # The float32 path of `to_gray_weighted_average` is the same luma without
        # the 8-bit quantisation, and only the weighted variant is defined for it.
        if modes[method] != GRAY_WEIGHTED:
            raise ValueError(
                f"{method!r} is only defined for uint8 images upstream; "
                "use 'weighted' or 'desaturation' for float32"
            )
        return _lib.gray_luma_f32(img)
    return _lib.gray_from_rgb(img, modes[method])


def desaturate(img) -> np.ndarray:
    """`to_gray_desaturation`: the midpoint of the per-pixel channel extremes."""
    return _lib.desaturate(np.ascontiguousarray(img, dtype=np.float32))


def channel_shuffle(img, order) -> np.ndarray:
    """`channel_shuffle`: ``out[..., i] = img[..., order[i]]``."""
    return _lib.channel_shuffle(img, order)


# --------------------------------------------------------------------------
# blur


def gaussian_kernel_1d(sigma: float, ksize: int = 0) -> np.ndarray:
    """`create_gaussian_kernel_1d`, with upstream's kernel-size rule."""
    size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize
    size = size + 1 if size % 2 == 0 else size
    if size < 100:
        x = np.array(list(range(-(size // 2), (size // 2) + 1, 1)), dtype=np.float64)
    else:
        x = np.linspace(-(size // 2), size // 2, size)
    return _lib.gaussian_kernel_1d(x, sigma)


def gaussian_blur(
    img, sigma: float, ksize: int = 0, border: int = BORDER_REPLICATE
) -> np.ndarray:
    """Separable Gaussian blur of a float32 image.

    The float32 input is deliberate: `generate_displacement_fields` and
    `sharpen_gaussian` both feed `cv2.GaussianBlur` a float32 displacement field
    or a float32 image, and a float32 blur is where a compiled pass can be
    compared to OpenCV without a quantisation step in between.
    """
    kernel = gaussian_kernel_1d(sigma, ksize)
    return _lib.blur_separable(
        np.ascontiguousarray(img, dtype=np.float32), kernel, border
    )


# --------------------------------------------------------------------------
# resampling


def remap_nearest(img, map_x, map_y, border: int = BORDER_CONSTANT) -> np.ndarray:
    """Nearest resample of a uint8 image through a coordinate map.

    This is the resampling half of the distortion transforms: build a map, then
    resample through it.
    """
    return _lib.remap_nearest(img, map_x, map_y, border)


def warp_affine_map(matrix, height: int, width: int):
    """The sample-coordinate grid of a forward 2x3 affine, for `remap_nearest`.

    Returns ``(map_x, map_y)`` float32 arrays of shape ``(height, width)``. Pass
    them to `remap_nearest` to resample through the affine without materialising
    an index mesh.
    """
    return _lib.affine_map(matrix, height, width)


# --------------------------------------------------------------------------
# statistics


def mean_std(img):
    """Per-channel ``(mean, std)`` of a float64 image, in one pass.

    This is the reduction `Normalize` performs to fill in default
    ``mean``/``std`` values. The standard deviation is the population one,
    ``ddof=0``, matching `np.std`.
    """
    total, total_sq = _lib.channel_stats(np.ascontiguousarray(img, dtype=np.float64))
    n = max(np.prod(np.asarray(img).shape[:2]), 1)
    mean = total / n
    variance = np.maximum(total_sq / n - mean * mean, 0.0)
    return mean, np.sqrt(variance)
