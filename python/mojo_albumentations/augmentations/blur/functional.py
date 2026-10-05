"""Port of `albumentations.augmentations.blur.functional`.

Function names, argument order and defaults are upstream's.
"""

from __future__ import annotations

import math
import random

import numpy as np

from ... import _lib
from ..._core import get_num_channels

__all__ = [
    "box_blur",
    "create_motion_kernel",
    "create_gaussian_kernel",
    "create_gaussian_kernel_1d",
    "create_gaussian_kernel_input_array",
]


def box_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Blur an image with a normalised box filter."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    out = _lib.empty(plane.shape, np.uint8)
    _lib.call(
        "alb_box_blur_u8",
        _lib.addr(plane),
        _lib.addr(out),
        height,
        width,
        nchan,
        int(ksize),
    )
    return out


def create_motion_kernel(
    kernel_size: int,
    angle: float,
    direction: float,
    allow_shifted: bool,
    random_state: random.Random,
) -> np.ndarray:
    """Create a motion blur kernel."""
    direction = float(np.clip(direction, -1.0, 1.0))

    angle_rad = np.deg2rad(angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))

    shift_x = 0.0
    shift_y = 0.0
    if allow_shifted:
        line_length = kernel_size // 2
        shift_x = random_state.uniform(-1, 1) * line_length / 2
        shift_y = random_state.uniform(-1, 1) * line_length / 2

    out = _lib.empty((kernel_size, kernel_size), np.float32)
    _lib.call(
        "alb_create_motion_kernel",
        _lib.addr(out),
        int(kernel_size),
        cos_a,
        sin_a,
        direction,
        shift_x,
        shift_y,
    )
    return out


def create_gaussian_kernel(sigma: float, ksize: int = 0) -> np.ndarray:
    """Create a 2D normalized Gaussian kernel following PIL's approach."""
    size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize
    size = size + 1 if size % 2 == 0 else size

    x = create_gaussian_kernel_input_array(size)
    kernel_1d = _gaussian_profile(x, sigma)

    out = _lib.empty((size, size), np.float64)
    _lib.call(
        "alb_create_gaussian_kernel_2d",
        _lib.addr(kernel_1d),
        _lib.addr(out),
        size,
    )
    return out


def create_gaussian_kernel_1d(sigma: float, ksize: int = 0) -> np.ndarray:
    """Create a 1D Gaussian kernel following PIL's approach."""
    size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize
    size = size + 1 if size % 2 == 0 else size

    x = create_gaussian_kernel_input_array(size)
    return _gaussian_profile(x, sigma)


def create_gaussian_kernel_input_array(size: int) -> np.ndarray:
    """The x-coordinate ramp the Gaussian profile is evaluated on.

    Piecewise function is needed as equivalent python list comprehension is
    faster than np.linspace for values of size < 100. That branch is the
    `int64` a Python `range` produces, and the dtype is part of the result
    upstream hands back, so it is reproduced rather than widened.

    The range is inclusive at both ends, so an even `size` yields one element
    more than `size`. Kernel sizes are always forced odd before they get
    here, but this function is public and mirrors `range` exactly.
    """
    if size < 100:
        out = _lib.empty((size // 2) * 2 + 1, np.int64)
        _lib.call(
            "alb_create_gaussian_kernel_input_array",
            _lib.addr(out),
            int(size),
            1,
        )
        return out
    out = _lib.empty(size, np.float64)
    _lib.call(
        "alb_create_gaussian_kernel_input_array",
        _lib.addr(out),
        int(size),
        0,
    )
    return out


def _gaussian_profile(x: np.ndarray, sigma: float) -> np.ndarray:
    ramp = np.ascontiguousarray(x, dtype=np.float64)
    out = _lib.empty(ramp.shape[0], np.float64)
    _lib.call(
        "alb_create_gaussian_kernel_1d",
        _lib.addr(ramp),
        ramp.shape[0],
        float(sigma),
        _lib.addr(out),
    )
    return out


def gaussian_blur_float32(
    img: np.ndarray,
    kernel_1d: np.ndarray,
    border: int = _lib.BORDER_REFLECT_101,
) -> np.ndarray:
    """`cv2.GaussianBlur` of a float32 image with an explicit 1-D profile.

    Not an upstream public name: `sharpen_gaussian`, `unsharp_mask` and
    `generate_displacement_fields` all reach the same primitive, and upstream
    reaches it through `cv2.GaussianBlur`.
    """
    plane = _lib.contiguous(img, np.float32)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    ksize = kernel_1d.shape[0]
    profile = np.ascontiguousarray(kernel_1d, np.float32)
    tmp = _lib.empty(plane.shape, np.float32)
    out = _lib.empty(plane.shape, np.float32)
    _lib.call(
        "alb_blur_separable_f32",
        _lib.addr(plane),
        _lib.addr(tmp),
        _lib.addr(out),
        _lib.addr(profile),
        height,
        width,
        nchan,
        ksize,
        border,
    )
    return out