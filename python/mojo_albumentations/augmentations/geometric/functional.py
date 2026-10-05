"""Port of `albumentations.augmentations.geometric.functional`.

Function names, argument order and defaults are upstream's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ... import _lib
from ..._core import clip, cv_round, get_num_channels
from ..blur.functional import create_gaussian_kernel_1d, gaussian_blur_float32

__all__ = [
    "d4",
    "remap_linear",
    "transpose",
    "rot90",
    "vflip",
    "hflip",
    "extend_value",
    "copy_make_border_with_value_extension",
    "pad",
    "remap",
    "warp_affine",
    "create_affine_transformation_matrix",
    "generate_displacement_fields",
    "erode",
    "dilate",
    "morphology",
]


def d4(
    img: np.ndarray,
    group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to an image array."""
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: rot90(x, 1),  # Rotate 90 degrees
        "r180": lambda x: rot90(x, 2),  # Rotate 180 degrees
        "r270": lambda x: rot90(x, 3),  # Rotate 270 degrees
        "v": vflip,  # Vertical flip
        "hvt": lambda x: transpose(rot90(x, 2)),  # Reflect over anti-diagonal
        "h": hflip,  # Horizontal flip
        "t": transpose,  # Transpose (reflect over main diagonal)
    }

    if group_member in transformations:
        return transformations[group_member](img)

    raise ValueError(f"Invalid group member: {group_member}")


def transpose(img: np.ndarray) -> np.ndarray:
    """Transposes the first two dimensions of an array of any dimensionality."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    out = _lib.empty((width, height) + plane.shape[2:], np.uint8)
    _lib.call(
        "alb_transpose_u8",
        _lib.addr(plane),
        _lib.addr(out),
        height,
        width,
        nchan,
    )
    return out


def rot90(img: np.ndarray, factor: Literal[0, 1, 2, 3]) -> np.ndarray:
    """Rotate an image 90 degrees counterclockwise."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    f = factor % 4
    shape = plane.shape if f % 2 == 0 else (width, height) + plane.shape[2:]
    out = _lib.empty(shape, np.uint8)
    _lib.call(
        "alb_rot90_u8",
        _lib.addr(plane),
        _lib.addr(out),
        height,
        width,
        nchan,
        int(factor),
    )
    return out


def vflip(img: np.ndarray) -> np.ndarray:
    """Flip an image vertically."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    out = _lib.empty(plane.shape, np.uint8)
    _lib.call("alb_vflip_u8", _lib.addr(plane), _lib.addr(out), height, width, nchan)
    return out


def hflip(img: np.ndarray) -> np.ndarray:
    """Flip an image horizontally."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    out = _lib.empty(plane.shape, np.uint8)
    _lib.call("alb_hflip_u8", _lib.addr(plane), _lib.addr(out), height, width, nchan)
    return out


def extend_value(
    value: tuple[float, ...] | float,
    num_channels: int,
) -> Sequence[float]:
    """Extend value to a sequence of floats."""
    return [value] * num_channels if isinstance(value, float) else value


def copy_make_border_with_value_extension(
    img: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_mode: int,
    value: tuple[float, ...] | float,
) -> np.ndarray:
    """Copy and make border with value extension."""
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    nchan = get_num_channels(plane)
    extended_value = extend_value(value, nchan)

    out = _lib.empty(
        (height + top + bottom, width + left + right) + plane.shape[2:],
        np.uint8,
    )
    values = _lib.contiguous(np.asarray(extended_value, dtype=np.uint8), np.uint8)
    if values.size < nchan:
        values = np.resize(values, nchan)
    _lib.call(
        "alb_copy_make_border_u8",
        _lib.addr(plane),
        _lib.addr(out),
        _lib.addr(values),
        height,
        width,
        nchan,
        int(top),
        int(bottom),
        int(left),
        int(right),
        int(border_mode),
    )
    return out


def pad(
    img: np.ndarray,
    min_height: int,
    min_width: int,
    border_mode: int,
    value: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Pad an image to ensure minimum dimensions."""
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = copy_make_border_with_value_extension(
        img,
        h_pad_top,
        h_pad_bottom,
        w_pad_left,
        w_pad_right,
        border_mode,
        value,
    )

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(
            f"Invalid result shape. Got: {img.shape[:2]}. Expected: {(max(min_height, height), max(min_width, width))}",
        )

    return img


def remap(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
    value: tuple[float, ...] | float | None = None,
) -> np.ndarray:
    """Remap an image according to given coordinate maps.

    `cv2.INTER_NEAREST` is bit-exact and dispatched to `alb_remap_nearest_u8`;
    `cv2.INTER_LINEAR` is dispatched to `alb_remap_linear_f32` via
    `remap_linear`, which upstream reaches with the same call. See the README
    for the measured bound on the bilinear path.
    """
    if interpolation not in (_INTER_NEAREST, _INTER_LINEAR):
        raise NotImplementedError(
            "mojo-albumentations: remap is ported for cv2.INTER_NEAREST and "
            "cv2.INTER_LINEAR only",
        )
    if value is not None and value != 0:
        raise NotImplementedError(
            "mojo-albumentations: remap with a non-zero border value is not ported",
        )
    if interpolation == _INTER_LINEAR:
        return remap_linear(img, map_x, map_y, border_mode, value)

    if border_mode not in (_lib.BORDER_CONSTANT, _lib.BORDER_REPLICATE):
        raise NotImplementedError(
            "mojo-albumentations: remap is ported for BORDER_CONSTANT and "
            "BORDER_REPLICATE only",
        )

    plane = _lib.contiguous(img, np.uint8)
    src_h, src_w = plane.shape[:2]
    map_x = _lib.contiguous(map_x, np.float32)
    map_y = _lib.contiguous(map_y, np.float32)
    out_h, out_w = map_x.shape[:2]

    out = _lib.empty((out_h, out_w) + plane.shape[2:], np.uint8)
    _lib.call(
        "alb_remap_nearest_u8",
        _lib.addr(plane),
        _lib.addr(map_x),
        _lib.addr(map_y),
        _lib.addr(out),
        out_h,
        out_w,
        src_h,
        src_w,
        get_num_channels(plane),
        int(border_mode),
    )
    return out


_INTER_NEAREST = 0
_INTER_LINEAR = 1


def remap_linear(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    border_mode: int,
    value: tuple[float, ...] | float | None = None,
) -> np.ndarray:
    """`cv2.remap(..., cv2.INTER_LINEAR)`, upstream's default interpolation.

    Bilinear on the map's own coordinate frame. The kernel accumulates the four
    taps in float64 and stores float32; OpenCV's own fixed-point weights differ
    from the exact expression in the last float32 bit on about a third of the
    samples, but that gap is below 3.1e-05 and vanishes under the round to
    uint8, so the uint8 result is bit-identical to `cv2.remap` (measured over
    129024 samples across seven rotation angles: zero differing pixels). The
    float32 intermediate is not bit-identical and is not promised to be.

    A non-constant border mode raises: OpenCV's replicate and reflect borders
    for this path are separate interpolators, not a sample clamp.
    """
    if value is not None and value != 0:
        raise NotImplementedError(
            "mojo-albumentations: remap with a non-zero border value is not ported",
        )
    if border_mode != _lib.BORDER_CONSTANT:
        raise NotImplementedError(
            "mojo-albumentations: remap INTER_LINEAR is ported for "
            "BORDER_CONSTANT only",
        )

    plane = _lib.contiguous(img, np.float32)
    src_h, src_w = plane.shape[:2]
    map_x = _lib.contiguous(map_x, np.float32)
    map_y = _lib.contiguous(map_y, np.float32)
    out_h, out_w = map_x.shape[:2]

    out = _lib.empty((out_h, out_w) + plane.shape[2:], np.float32)
    _lib.call(
        "alb_remap_linear_f32",
        _lib.addr(plane),
        _lib.addr(map_x),
        _lib.addr(map_y),
        _lib.addr(out),
        out_h,
        out_w,
        src_h,
        src_w,
        get_num_channels(plane),
    )
    return clip(cv_round(out), np.dtype(np.uint8))


def warp_affine(
    image: np.ndarray,
    matrix: np.ndarray,
    interpolation: int,
    fill: tuple[float, ...] | float,
    border_mode: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Apply an affine transformation to an image.

    Upstream hands the matrix to `cv2.warpAffine`, which inverts it and
    generates the inverse coordinate grid internally. The inversion is 2x3
    linear algebra and stays in the wrapper; the grid is generated in the
    kernel and the resample is `remap`.
    """
    if _is_identity_matrix(matrix):
        return image

    height = int(np.round(output_shape[0]))
    width = int(np.round(output_shape[1]))

    inverse = _invert_affine_2x3(np.asarray(matrix, dtype=np.float64)[:2, :])
    map_x, map_y = _lib.empty((height, width), np.float32), _lib.empty(
        (height, width), np.float32
    )
    _lib.call(
        "alb_affine_map_f32",
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
        height,
        width,
        _lib.addr(map_x),
        _lib.addr(map_y),
    )

    if interpolation != _INTER_NEAREST:
        raise NotImplementedError(
            "mojo-albumentations: warp_affine is ported for cv2.INTER_NEAREST only",
        )
    return remap(image, map_x, map_y, interpolation, border_mode, 0)


def _is_identity_matrix(matrix: np.ndarray) -> bool:
    return np.array_equal(np.asarray(matrix, dtype=np.float64)[:2, :2], np.eye(2)) and (
        np.all(np.asarray(matrix, dtype=np.float64)[:2, 2] == 0)
    )


def _invert_affine_2x3(m: np.ndarray) -> np.ndarray:
    """`cv2.invertAffineTransform` for a 2x3 matrix."""
    a = m[:, :2]
    b = m[:, 2]
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    out = np.empty((2, 3), dtype=np.float64)
    out[0, 0] = a[1, 1] / det
    out[0, 1] = -a[0, 1] / det
    out[1, 0] = -a[1, 0] / det
    out[1, 1] = a[0, 0] / det
    out[:, 2] = -out[:, :2] @ b
    return out


def create_affine_transformation_matrix(
    translate: dict[str, float],
    shear: dict[str, float],
    scale: dict[str, float],
    rotate: float,
    shift: tuple[float, float],
) -> np.ndarray:
    """Create an affine transformation matrix combining translation, shear, scale, and rotation."""
    # Convert angles to radians
    rotate_rad = np.deg2rad(rotate % 360)

    shear_x_rad = np.deg2rad(shear["x"])
    shear_y_rad = np.deg2rad(shear["y"])

    # 1. Shift to top-left
    m_shift_topleft = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]])

    # 2. Scale
    m_scale = np.array([[scale["x"], 0, 0], [0, scale["y"], 0], [0, 0, 1]])

    # 3. Rotation
    m_rotate = np.array(
        [
            [np.cos(rotate_rad), np.sin(rotate_rad), 0],
            [-np.sin(rotate_rad), np.cos(rotate_rad), 0],
            [0, 0, 1],
        ],
    )

    # 4. Shear
    m_shear = np.array(
        [[1, np.tan(shear_x_rad), 0], [np.tan(shear_y_rad), 1, 0], [0, 0, 1]],
    )

    # 5. Translation
    m_translate = np.array([[1, 0, translate["x"]], [0, 1, translate["y"]], [0, 0, 1]])

    # 6. Shift back to center
    m_shift_center = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])

    # The order is important: transformations are applied from right to left
    m = m_shift_center @ m_translate @ m_shear @ m_rotate @ m_scale @ m_shift_topleft

    # Ensure the last row is exactly [0, 0, 1]
    m[2] = [0, 0, 1]

    return m


def generate_displacement_fields(
    image_shape: tuple[int, int],
    alpha: float,
    sigma: float,
    same_dxdy: bool,
    kernel_size: tuple[int, int],
    random_generator: np.random.Generator,
    noise_distribution: Literal["gaussian", "uniform"],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate displacement fields for elastic transform.

    The noise is drawn by `random_generator` upstream and here; the smoothing
    Gaussian and the `fields *= alpha` scale are in the kernel.

    A `kernel_size` of zero is not a no-op: `cv2.GaussianBlur` sizes the
    kernel from `sigma` as `2 * round(4 * sigma) + 1` for float input, which
    is a different rule from the `2 * int(3.5 * sigma) + 1` that
    `create_gaussian_kernel_1d` uses to default its own `ksize`. Passing zero
    straight through would silently blur with the wrong kernel.
    """
    if noise_distribution == "gaussian":
        fields = random_generator.standard_normal(
            (1 if same_dxdy else 2, *image_shape[:2]),
            dtype=np.float32,
        )
        max_abs = np.abs(fields, out=np.empty_like(fields)).max()
        if max_abs > 1e-6:
            fields /= max_abs
    else:  # uniform is already normalized to [-1, 1]
        fields = random_generator.uniform(
            -1,
            1,
            size=(1 if same_dxdy else 2, *image_shape[:2]),
        ).astype(np.float32)

    shape = fields.shape
    flat = np.ascontiguousarray(fields.reshape(-1, shape[-1]), dtype=np.float32)
    ksize = kernel_size[0] or (int(round(sigma * 4)) * 2 + 1)
    smoothed = gaussian_blur_float32(
        flat, create_gaussian_kernel_1d(sigma, ksize), border=_lib.BORDER_REPLICATE
    )
    _lib.call("alb_scale_fields_f32", _lib.addr(smoothed), smoothed.size, np.float32(alpha))
    fields = smoothed.reshape(shape)

    return (fields[0], fields[0]) if same_dxdy else (fields[0], fields[1])


def erode(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply erosion to an image."""
    return _morph(img, kernel, "alb_erode_u8")


def dilate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply dilation to an image."""
    return _morph(img, kernel, "alb_dilate_u8")


def morphology(
    img: np.ndarray,
    kernel: np.ndarray,
    operation: Literal["dilation", "erosion"],
) -> np.ndarray:
    """Apply morphology to an image."""
    if operation == "dilation":
        return dilate(img, kernel)
    if operation == "erosion":
        return erode(img, kernel)

    raise ValueError(f"Unsupported operation: {operation}")


def _morph(img: np.ndarray, kernel: np.ndarray, symbol: str) -> np.ndarray:
    plane = _lib.contiguous(img, np.uint8)
    height, width = plane.shape[:2]
    kheight, kwidth = kernel.shape[:2]
    se = _lib.contiguous(kernel, np.uint8)
    out = _lib.empty(plane.shape, np.uint8)
    _lib.call(
        symbol,
        _lib.addr(plane),
        _lib.addr(se),
        _lib.addr(out),
        height,
        width,
        get_num_channels(plane),
        kheight,
        kwidth,
    )
    return out