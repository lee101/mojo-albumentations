"""Port of `albumentations.augmentations.pixel.functional`.

Function names, argument order and defaults are upstream's. Each body keeps
upstream's branch structure and calls the Mojo kernel that holds the per-pixel
arithmetic; see ``src/ported.mojo`` for the kernels, which are emitted in the
same order as the functions below.

Where a wrapper has to make a dtype or shape decision that upstream pushes
into an OpenCV call, the decision is made here and the divergence is listed in
the README's coverage section.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from ... import _lib
from ..._core import (
    MAX_VALUES_BY_DTYPE,
    clip,
    from_float,
    get_max_value,
    get_num_channels,
    is_grayscale_image,
    sz_lut,
    to_float,
)

__all__ = [
    "solarize",
    "posterize",
    "equalize",
    "move_tone_curve",
    "linear_transformation_rgb",
    "invert",
    "channel_shuffle",
    "gamma_transform",
    "to_gray_weighted_average",
    "to_gray_desaturation",
    "to_gray_average",
    "to_gray_max",
    "grayscale_to_multichannel",
    "adjust_brightness_torchvision",
    "adjust_contrast_torchvision",
    "adjust_saturation_torchvision",
    "unsharp_mask",
    "add_noise",
    "sharpen_gaussian",
    "apply_salt_and_pepper",
    "create_directional_gradient",
    "apply_linear_illumination",
    "apply_gaussian_illumination",
    "auto_contrast",
    "create_contrast_lut",
    "get_histogram_bounds",
    "rgb_to_optical_density",
    "convolve",
    "separable_convolve",
]


def _multiply_mode(dtype: np.dtype) -> tuple[float, int]:
    """`(max_value, quantise)` for `albucore`'s two multiply paths.

    On uint8 the table is cast back to uint8, which truncates; on float32 the
    result stays float32 and `wsum` does not round.
    """
    return (255.0, 1) if np.dtype(dtype) == np.dtype(np.uint8) else (1.0, 0)


def solarize(img: np.ndarray, threshold: float) -> np.ndarray:
    """Invert all pixel values above a threshold."""
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.uint8:
        lut = _lib.scratch_u8(int(max_val) + 1)
        _lib.call("alb_solarize_lut", _lib.addr(lut), int(max_val), float(threshold))
        prev_shape = img.shape
        img = sz_lut(np.ascontiguousarray(img), lut, inplace=False)
        return img if len(prev_shape) == img.ndim else np.expand_dims(img, -1)
    return np.where(img >= threshold, max_val - img, img)


def posterize(
    img: np.ndarray,
    bits: Literal[1, 2, 3, 4, 5, 6, 7]
    | list[Literal[1, 2, 3, 4, 5, 6, 7]],
) -> np.ndarray:
    """Reduce the number of bits for each color channel by keeping only the highest N bits."""
    bits_array = np.uint8(bits)

    if not bits_array.shape or len(bits_array) == 1:
        lut = _lib.scratch_u8(256)
        _lib.call("alb_posterize_lut", _lib.addr(lut), int(bits_array.reshape(-1)[0]))
        return sz_lut(np.ascontiguousarray(img), lut, inplace=False)

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits_array):
        lut = _lib.scratch_u8(256)
        _lib.call("alb_posterize_lut", _lib.addr(lut), int(channel_bits))
        result_img[..., i] = sz_lut(
            np.ascontiguousarray(img[..., i]), lut, inplace=True
        )
    return result_img


def equalize(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    mode: Literal["cv", "pil"] = "cv",
    by_channels: bool = True,
) -> np.ndarray:
    """Apply histogram equalization to the input image.

    Only `mode="cv"` with `mask=None` and only the `by_channels=True` /
    grayscale split are ported; see the README.
    """
    if mask is not None:
        raise NotImplementedError(
            "mojo-albumentations: equalize with an explicit mask is not ported",
        )
    if mode == "pil":
        raise NotImplementedError(
            "mojo-albumentations: equalize mode='pil' is not ported",
        )
    if not by_channels and not is_grayscale_image(img):
        raise NotImplementedError(
            "mojo-albumentations: equalize by_channels=False is not ported",
        )

    result = np.empty_like(img)
    if is_grayscale_image(img):
        return _equalize_cv(img)

    for i in range(3):
        result[..., i] = _equalize_cv(img[..., i])
    return result


def _equalize_cv(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """`cv2.equalizeHist` of one channel, built and applied in the kernel."""
    plane = np.ascontiguousarray(img, dtype=np.uint8)
    out = np.empty_like(plane)
    hist = _lib.scratch_i64(256)
    lut = _lib.scratch_u8(256)
    _lib.call(
        "alb_equalize_cv_u8",
        _lib.addr(plane),
        _lib.addr(out),
        plane.size,
        _lib.addr(hist),
        _lib.addr(lut),
    )
    return out


def move_tone_curve(
    img: np.ndarray,
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> np.ndarray:
    """Rescale the relationship between bright and dark areas via a Bezier curve."""
    num_channels = get_num_channels(img)

    if np.isscalar(low_y) and np.isscalar(high_y):
        lut = _lib.scratch_u8(256)
        _lib.call("alb_move_tone_curve_lut", _lib.addr(lut), float(low_y), float(high_y))
        return sz_lut(np.ascontiguousarray(img, dtype=np.uint8), lut, inplace=False)
    if isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        plane = np.ascontiguousarray(img, dtype=np.uint8)
        merged = np.empty(plane.shape, dtype=np.uint8)
        for i in range(num_channels):
            lut = _lib.scratch_u8(256)
            _lib.call(
                "alb_move_tone_curve_lut", _lib.addr(lut), float(low_y[i]), float(high_y[i])
            )
            merged[..., i] = sz_lut(np.ascontiguousarray(plane[..., i]), lut, inplace=True)
        return merged

    raise TypeError(
        f"low_y and high_y must both be of type float or np.ndarray. Got {type(low_y)} and {type(high_y)}",
    )


def linear_transformation_rgb(
    img: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply a linear transformation to the RGB channels of an image."""
    plane = _lib.contiguous(img, np.float32)
    matrix = _lib.contiguous(transformation_matrix, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_linear_transformation_rgb_f32",
        _lib.addr(plane),
        _lib.addr(out),
        plane[..., 0].size,
        _lib.addr(matrix),
    )
    return clip(out, img.dtype)


def invert(img: np.ndarray) -> np.ndarray:
    """Invert the colors of an image."""
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    if img.dtype != np.uint8:
        return max_value - img
    plane = _lib.contiguous(img, np.uint8)
    out = _lib.empty(plane.shape, np.uint8)
    _lib.call(
        "alb_invert_u8",
        _lib.addr(plane),
        _lib.addr(out),
        plane.size,
        int(max_value),
    )
    return out


def channel_shuffle(img: np.ndarray, channels_shuffled: list[int]) -> np.ndarray:
    """Shuffle the channels of an image."""
    plane = _lib.contiguous(img, np.uint8)
    out = _lib.empty(plane.shape, np.uint8)
    npix = plane.shape[0] * plane.shape[1]
    _lib.call(
        "alb_channel_shuffle_u8",
        _lib.addr(plane),
        _lib.addr(out),
        npix,
        int(channels_shuffled[0]),
        int(channels_shuffled[1]),
        int(channels_shuffled[2]),
    )
    return out


def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma transformation to an image."""
    if img.dtype == np.uint8:
        lut = _lib.scratch_u8(256)
        _lib.call("alb_gamma_transform_lut", _lib.addr(lut), float(gamma))
        return sz_lut(_lib.contiguous(img, np.uint8), lut, inplace=False)

    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_gamma_transform_f32",
        _lib.addr(plane),
        _lib.addr(out),
        plane.size,
        np.float32(gamma),
    )
    return out


def to_gray_weighted_average(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using the weighted average method."""
    height, width = img.shape[:2]
    npix = height * width
    if img.dtype == np.uint8:
        plane = _lib.contiguous(img, np.uint8)
        out = _lib.empty((height, width), np.uint8)
        _lib.call(
            "alb_to_gray_weighted_average_u8", _lib.addr(plane), _lib.addr(out), npix
        )
        return out
    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty((height, width), np.float32)
    _lib.call(
        "alb_to_gray_weighted_average_f32", _lib.addr(plane), _lib.addr(out), npix
    )
    return out


def to_gray_desaturation(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale using the desaturation method."""
    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty(plane.shape[:-1], np.float32)
    _lib.call(
        "alb_to_gray_desaturation_f32",
        _lib.addr(plane),
        _lib.addr(out),
        plane[..., 0].size,
    )
    return clip(out, img.dtype)


def to_gray_average(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale using the average method."""
    height, width = img.shape[:2]
    npix = height * width
    nchan = get_num_channels(img)
    if img.dtype == np.uint8:
        plane = _lib.contiguous(img, np.uint8)
        out = _lib.empty((height, width), np.uint8)
        _lib.call(
            "alb_to_gray_average_u8",
            _lib.addr(plane),
            _lib.addr(out),
            npix,
            nchan,
        )
        return out
    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty((height, width), np.float32)
    _lib.call(
        "alb_to_gray_average_f32",
        _lib.addr(plane),
        _lib.addr(out),
        npix,
        nchan,
    )
    return out


def to_gray_max(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale using the maximum channel value method."""
    height, width = img.shape[:2]
    npix = height * width
    nchan = get_num_channels(img)
    if img.dtype == np.uint8:
        plane = _lib.contiguous(img, np.uint8)
        out = _lib.empty((height, width), np.uint8)
        _lib.call("alb_to_gray_max_u8", _lib.addr(plane), _lib.addr(out), npix, nchan)
        return out
    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty((height, width), np.float32)
    _lib.call("alb_to_gray_max_f32", _lib.addr(plane), _lib.addr(out), npix, nchan)
    return out


def grayscale_to_multichannel(
    grayscale_image: np.ndarray,
    num_output_channels: int = 3,
) -> np.ndarray:
    """Convert a grayscale image to a multi-channel image."""
    if num_output_channels == 1:
        return grayscale_image

    squeezed = np.squeeze(grayscale_image)
    plane = _lib.contiguous(squeezed, np.uint8)
    npix = plane.shape[0] * plane.shape[1]
    out = _lib.empty((plane.shape[0], plane.shape[1], num_output_channels), np.uint8)
    _lib.call(
        "alb_grayscale_to_multichannel_u8",
        _lib.addr(plane),
        _lib.addr(out),
        npix,
        num_output_channels,
    )
    return out


def adjust_brightness_torchvision(img: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Adjust the brightness of an image."""
    if factor == 0:
        return np.zeros_like(img)
    if factor == 1:
        return img

    max_val, truncate = _multiply_mode(img.dtype)
    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_multiply_f32",
        _lib.addr(plane),
        _lib.addr(out),
        plane.size,
        np.float32(factor),
        np.float32(max_val),
        truncate,
    )
    return out.astype(img.dtype, copy=False)


def adjust_contrast_torchvision(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust the contrast of an image."""
    if factor == 1:
        return img

    gray = to_gray_weighted_average(
        img if img.ndim == 2 else _lib.contiguous(img, img.dtype)
    )
    mean = float(gray.mean())

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    max_val, truncate = _multiply_mode(img.dtype)
    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_multiply_add_f32",
        _lib.addr(plane),
        _lib.addr(out),
        plane.size,
        np.float32(factor),
        np.float32(mean * (1 - factor)),
        np.float32(max_val),
        truncate,
    )
    return out.astype(img.dtype, copy=False)


def adjust_saturation_torchvision(
    img: np.ndarray,
    factor: float,
    gamma: float = 0,
) -> np.ndarray:
    """Adjust the saturation of an image."""
    if factor == 1 or is_grayscale_image(img):
        return img

    max_val, round_out = _multiply_mode(img.dtype)
    plane = _lib.contiguous(img, np.float32)
    gray = to_gray_weighted_average(plane)
    gray_plane = _lib.contiguous(gray, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_add_weighted_f32",
        _lib.addr(plane),
        _lib.addr(gray_plane),
        _lib.addr(out),
        plane.shape[0] * plane.shape[1],
        np.float32(factor),
        np.float32(gamma),
        np.float32(max_val),
        round_out,
    )
    return out.astype(img.dtype, copy=False)


def unsharp_mask(
    image: np.ndarray,
    ksize: int,
    sigma: float,
    alpha: float,
    threshold: int,
) -> np.ndarray:
    """Apply an unsharp mask to an image."""
    from ..blur.functional import create_gaussian_kernel_1d, gaussian_blur_float32

    plane = to_float(image) if image.dtype != np.float32 else _lib.contiguous(image, np.float32)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)

    blur = gaussian_blur_float32(plane, create_gaussian_kernel_1d(sigma, ksize))
    mask = _lib.empty(plane.shape, np.float32)
    sharp = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_unsharp_mask_residual",
        _lib.addr(plane),
        _lib.addr(blur),
        _lib.addr(mask),
        _lib.addr(sharp),
        plane.size,
        np.float32(alpha),
        int(threshold),
    )

    soft_mask = gaussian_blur_float32(
        mask.reshape(height, width, nchan), create_gaussian_kernel_1d(sigma, ksize)
    )
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_unsharp_mask_combine",
        _lib.addr(sharp),
        _lib.addr(plane),
        _lib.addr(soft_mask),
        _lib.addr(out),
        plane.size,
    )
    return out if image.dtype == np.float32 else from_float(out, image.dtype)


def add_noise(img: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Add noise to an image."""
    plane = _lib.contiguous(img, np.float32)
    noise_plane = _lib.contiguous(noise, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_add_noise_f32",
        _lib.addr(plane),
        _lib.addr(noise_plane),
        _lib.addr(out),
        plane.size,
    )
    return clip(out, img.dtype)


def sharpen_gaussian(
    img: np.ndarray,
    alpha: float,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    """Sharpen image using Gaussian blur.

    Upstream keeps the input dtype, so a uint8 image is blurred by OpenCV's
    fixed-point `cv2.GaussianBlur`; this port blurs in float32 and clips to the
    dtype maximum. See the README for the measured bound on uint8 input.
    """
    from ..blur.functional import create_gaussian_kernel_1d, gaussian_blur_float32

    plane = _lib.contiguous(img, np.float32)
    blurred = gaussian_blur_float32(plane, create_gaussian_kernel_1d(sigma, kernel_size))
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_sharpen_gaussian_f32",
        _lib.addr(plane),
        _lib.addr(blurred),
        _lib.addr(out),
        plane.size,
        np.float32(alpha),
    )
    return clip(out, img.dtype)


def apply_salt_and_pepper(
    img: np.ndarray,
    salt_mask: np.ndarray,
    pepper_mask: np.ndarray,
) -> np.ndarray:
    """Apply salt and pepper noise to an image."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    salt_plane = _lib.contiguous(salt_mask, np.uint8).reshape(height, width)
    pepper_plane = _lib.contiguous(pepper_mask, np.uint8).reshape(height, width)
    out = _lib.empty(plane.shape, np.uint8)
    _lib.call(
        "alb_apply_salt_and_pepper_u8",
        _lib.addr(plane),
        _lib.addr(salt_plane),
        _lib.addr(pepper_plane),
        _lib.addr(out),
        height * width,
        nchan,
        int(MAX_VALUES_BY_DTYPE[plane.dtype]),
    )
    return out


def create_directional_gradient(
    height: int,
    width: int,
    angle: float,
) -> np.ndarray:
    """Create a directional gradient in [0, 1] range."""
    # Fast path for horizontal gradients
    if angle == 0:
        return np.linspace(0, 1, width, dtype=np.float32)[None, :] * np.ones(
            (height, 1), dtype=np.float32
        )
    if angle == 180:
        return np.linspace(1, 0, width, dtype=np.float32)[None, :] * np.ones(
            (height, 1), dtype=np.float32
        )

    # Fast path for vertical gradients
    if angle == 90:
        return np.linspace(0, 1, height, dtype=np.float32)[:, None] * np.ones(
            (1, width), dtype=np.float32
        )
    if angle == 270:
        return np.linspace(1, 0, height, dtype=np.float32)[:, None] * np.ones(
            (1, width), dtype=np.float32
        )

    # Fast path for diagonal gradients using broadcasting
    if angle in (45, 135, 225, 315):
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Horizontal
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Vertical

        if angle == 45:  # Bottom-left to top-right
            return _normalize_minmax(x + y)
        if angle == 135:  # Bottom-right to top-left
            return _normalize_minmax((1 - x) + y)
        if angle == 225:  # Top-right to bottom-left
            return _normalize_minmax((1 - x) + (1 - y))
        # angle == 315:  # Top-left to bottom-right
        return _normalize_minmax(x + (1 - y))

    # General case for arbitrary angles
    angle_rad = np.deg2rad(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    out = _lib.empty((height, width), np.float32)
    _lib.call(
        "alb_create_directional_gradient_f32",
        _lib.addr(out),
        height,
        width,
        cos_a,
        sin_a,
    )
    return out


def _normalize_minmax(a: np.ndarray) -> np.ndarray:
    """`cv2.normalize(a, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)`."""
    lo = float(a.min())
    hi = float(a.max())
    if hi == lo:
        return np.zeros(a.shape, dtype=np.float32)
    scale = np.float32(1.0 / (hi - lo))
    return ((a - np.float32(lo)) * scale).astype(np.float32)


def apply_linear_illumination(
    img: np.ndarray, intensity: float, angle: float
) -> np.ndarray:
    """Apply linear illumination to the image."""
    height, width = img.shape[:2]
    nchan = get_num_channels(img)
    abs_intensity = abs(intensity)

    angle_rad = np.deg2rad(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_apply_linear_illumination_f32",
        _lib.addr(plane),
        _lib.addr(out),
        height,
        width,
        nchan,
        np.float32(intensity),
        np.float32(cos_a),
        np.float32(sin_a),
    )
    return clip(out, img.dtype)


def apply_gaussian_illumination(
    img: np.ndarray,
    intensity: float,
    center: tuple[float, float],
    sigma: float,
) -> np.ndarray:
    """Apply gaussian illumination to the image."""
    if intensity == 0:
        return img.copy()

    height, width = img.shape[:2]
    nchan = get_num_channels(img)

    center_x = width * center[0]
    center_y = height * center[1]
    sigma2 = 2 * (max(height, width) * sigma) ** 2  # Pre-compute denominator

    plane = _lib.contiguous(img, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_apply_gaussian_illumination_f32",
        _lib.addr(plane),
        _lib.addr(out),
        height,
        width,
        nchan,
        np.float32(intensity),
        np.float32(center_x),
        np.float32(center_y),
        np.float32(-1 / sigma2),
    )
    return clip(out, img.dtype)


def auto_contrast(
    img: np.ndarray,
    cutoff: float,
    ignore: int | None,
    method: Literal["cdf", "pil"],
) -> np.ndarray:
    """Apply automatic contrast enhancement."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    max_value = MAX_VALUES_BY_DTYPE[plane.dtype]

    result = plane.copy()
    hist = _lib.scratch_i64(nchan * 256)
    _lib.call(
        "alb_calc_hist_u8",
        _lib.addr(plane),
        _lib.addr(hist),
        height * width,
        nchan,
        -1 if ignore is None else int(ignore),
        int(max_value),
    )

    for i in range(nchan):
        if ignore is not None and i == ignore:
            continue

        channel_hist = hist.reshape(nchan, 256)[i]
        lo, hi = get_histogram_bounds(channel_hist, cutoff)
        if hi <= lo:
            continue

        lut = _lib.scratch_u8(256)
        cdf = _lib.scratch_f64(256)
        hist_plane = np.ascontiguousarray(channel_hist)
        _lib.call(
            "alb_create_contrast_lut",
            _lib.addr(hist_plane),
            int(lo),
            int(hi),
            int(max_value),
            _lib.METHOD_CDF if method == "cdf" else _lib.METHOD_PIL,
            _lib.addr(lut),
            _lib.addr(cdf),
        )
        if ignore is not None:
            lut[ignore] = ignore

        if nchan > 1:
            result[..., i] = sz_lut(
                np.ascontiguousarray(result[..., i]), lut, inplace=False
            )
        else:
            result = sz_lut(result, lut, inplace=False)

    return result



def create_contrast_lut(
    hist: np.ndarray,
    min_intensity: int,
    max_intensity: int,
    max_value: int,
    method: Literal["cdf", "pil"],
) -> np.ndarray:
    """Create lookup table for contrast adjustment."""
    lut = _lib.scratch_u8(256)
    cdf = _lib.scratch_f64(256)
    hist_plane = np.ascontiguousarray(hist, dtype=np.int64)
    _lib.call(
        "alb_create_contrast_lut",
        _lib.addr(hist_plane),
        int(min_intensity),
        int(max_intensity),
        int(max_value),
        _lib.METHOD_CDF if method == "cdf" else _lib.METHOD_PIL,
        _lib.addr(lut),
        _lib.addr(cdf),
    )
    return lut


def get_histogram_bounds(hist: np.ndarray, cutoff: float) -> tuple[int, int]:
    """Get the low and high bounds of the histogram."""
    out = _lib.scratch_i64(2)
    hist_plane = np.ascontiguousarray(hist, dtype=np.int64)
    _lib.call(
        "alb_get_histogram_bounds",
        _lib.addr(hist_plane),
        float(cutoff),
        _lib.addr(out),
    )
    return int(out[0]), int(out[1])


def rgb_to_optical_density(
    img: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert RGB image to optical density."""
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    pixel_matrix = _lib.contiguous(img.reshape(-1, 3), np.float32)
    out = _lib.empty(pixel_matrix.shape, np.float32)
    _lib.call(
        "alb_rgb_to_optical_density_f32",
        _lib.addr(pixel_matrix),
        _lib.addr(out),
        pixel_matrix.shape[0],
        np.float32(max_value),
        np.float32(eps),
    )
    return out


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a kernel."""
    plane = _lib.contiguous(img, np.float32)
    kheight, kwidth = kernel.shape[:2]
    kern = _lib.contiguous(kernel, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_convolve_f32",
        _lib.addr(plane),
        _lib.addr(kern),
        _lib.addr(out),
        plane.shape[0],
        plane.shape[1],
        get_num_channels(plane),
        kheight,
        kwidth,
        _lib.BORDER_REFLECT_101,
    )
    return clip(out, img.dtype)


def separable_convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a separable kernel."""
    plane = _lib.contiguous(img, np.float32)
    ksize = kernel.shape[0]
    tmp = _lib.empty(plane.shape, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    kern = _lib.contiguous(kernel, np.float32).reshape(-1)
    _lib.call(
        "alb_separable_convolve_f32",
        _lib.addr(plane),
        _lib.addr(tmp),
        _lib.addr(kern),
        _lib.addr(out),
        plane.shape[0],
        plane.shape[1],
        get_num_channels(plane),
        ksize,
        _lib.BORDER_REFLECT_101,
    )
    return clip(out, img.dtype)