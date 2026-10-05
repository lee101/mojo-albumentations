"""The `albucore` helpers the ported functions rely on.

`albumentations` decorates most of `pixel/functional.py` with `uint8_io`,
`float32_io`, `clipped` and `preserve_channel_dim`, and calls `sz_lut`,
`to_float`, `from_float` and `clip` directly. Those are the dtype and channel
contracts a caller sees, so they are reproduced here rather than inlined at
every call site - the same reason upstream keeps them in `albucore`.
"""

from __future__ import annotations

import numpy as np

from . import _lib

MAX_VALUES_BY_DTYPE: dict[np.dtype, float] = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float16"): 1.0,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
    np.dtype("int32"): 2147483647,
}

NUM_MULTI_CHANNEL_DIMENSIONS = 3


def get_num_channels(image: np.ndarray) -> int:
    return image.shape[2] if image.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else 1


def is_grayscale_image(image: np.ndarray) -> bool:
    return get_num_channels(image) == 1


def get_max_value(dtype: np.dtype) -> float:
    return MAX_VALUES_BY_DTYPE[np.dtype(dtype)]


def clip(
    img: np.ndarray, dtype: np.dtype, inplace: bool = False
) -> np.ndarray:
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if inplace:
        return np.clip(img, 0, max_value, out=img)
    return np.clip(img, 0, max_value).astype(dtype, copy=False)


def sz_lut(img: np.ndarray, lut: np.ndarray, inplace: bool = True) -> np.ndarray:
    """`out = lut[img]`, byte for byte."""
    if not inplace:
        img = img.copy()
    table = _lib.contiguous(lut, np.uint8)
    _lib.call(
        "alb_apply_lut_u8",
        _lib.addr(table),
        _lib.addr(img),
        _lib.addr(img),
        img.size,
    )
    return img


def to_float(img: np.ndarray, max_value: float | None = None) -> np.ndarray:
    """uint8 -> float32 in [0, 1], as `cv2.LUT` does it."""
    if img.dtype == np.float32:
        return img
    if img.dtype == np.float64:
        return img.astype(np.float32, copy=False)
    if max_value is None:
        max_value = get_max_value(img.dtype)
    return img.astype(np.float32) / np.float32(max_value)


def from_float(
    img: np.ndarray,
    target_dtype: np.dtype,
    max_value: float | None = None,
) -> np.ndarray:
    """float32 in [0, 1] -> `target_dtype`, rounded half to even."""
    if max_value is None:
        max_value = get_max_value(target_dtype)
    return clip(np.rint(img * max_value), target_dtype, inplace=True)


def cv_round(img: np.ndarray) -> np.ndarray:
    """`cvRound`: round half away from zero.

    `np.rint` rounds half to even, which is the right rule for `sz_lut` and the
    histogram tables but not for `cv2.remap`, whose bilinear path rounds half
    away. On a 40x50 image one sample in 6000 lands exactly on a half and the
    two rules disagree by one LSB there.
    """
    return np.floor(np.abs(img) + 0.5) * np.sign(img)


def preserve_channel_dim(img: np.ndarray) -> np.ndarray:
    """No-op for the image-shaped arrays this port accepts."""
    return img