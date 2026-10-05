"""ctypes bridge to the compiled Mojo kernels.

Everything in this module is plumbing: load the shared library, declare
argument types once, and hand raw buffer addresses across. The arithmetic
lives in ``src/ported.mojo``; the shape, dtype and branch logic that upstream
writes in Python lives in ``augmentations/*/functional.py``.

Buffers cross the C ABI as ``intptr_t`` addresses because ``@export`` rejects
parametric functions. Scratch buffers (a 256-entry histogram, a 256-entry LUT)
are allocated here on the NumPy side and passed in, so the kernels never
allocate.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-albumentations.so"

_i = ctypes.c_int64
_f = ctypes.c_double
_f32 = ctypes.c_float

BORDER_CONSTANT = 0
BORDER_REPLICATE = 1
BORDER_REFLECT_101 = 2

METHOD_CDF = 1
METHOD_PIL = 0


class LibraryNotBuilt(RuntimeError):
    pass


def _load() -> ctypes.CDLL:
    if not _LIB_PATH.exists():
        raise LibraryNotBuilt(
            f"{_LIB_PATH} does not exist. Run `pixi run build` first.",
        )
    return ctypes.CDLL(str(_LIB_PATH))


def loaded_from() -> str:
    """The path the kernel symbols were actually loaded from."""
    return str(_LIB_PATH)


lib = _load()

_SIGNATURES: dict[str, tuple[str, ...]] = {
    "alb_apply_lut_u8": (_i, _i, _i, _i),
    "alb_blur_separable_f32": (_i, _i, _i, _i, _i, _i, _i, _i, _i),
    "alb_remap_nearest_u8": (_i, _i, _i, _i, _i, _i, _i, _i, _i, _i),
    "alb_remap_linear_f32": (_i, _i, _i, _i, _i, _i, _i, _i, _i),
    "alb_solarize_lut": (_i, _i, _f),
    "alb_posterize_lut": (_i, _i),
    "alb_equalize_cv_u8": (_i, _i, _i, _i, _i),
    "alb_move_tone_curve_lut": (_i, _f, _f),
    "alb_linear_transformation_rgb_f32": (_i, _i, _i, _i),
    "alb_invert_u8": (_i, _i, _i, _i),
    "alb_channel_shuffle_u8": (_i, _i, _i, _i, _i, _i),
    "alb_gamma_transform_lut": (_i, _f),
    "alb_gamma_transform_f32": (_i, _i, _i, _f32),
    "alb_to_gray_weighted_average_u8": (_i, _i, _i),
    "alb_to_gray_weighted_average_f32": (_i, _i, _i),
    "alb_to_gray_desaturation_f32": (_i, _i, _i),
    "alb_to_gray_average_u8": (_i, _i, _i, _i),
    "alb_to_gray_average_f32": (_i, _i, _i, _i),
    "alb_to_gray_max_u8": (_i, _i, _i, _i),
    "alb_to_gray_max_f32": (_i, _i, _i, _i),
    "alb_grayscale_to_multichannel_u8": (_i, _i, _i, _i),
    "alb_multiply_f32": (_i, _i, _i, _f32, _f32, _i),
    "alb_multiply_add_f32": (_i, _i, _i, _f32, _f32, _f32, _i),
    "alb_add_weighted_f32": (_i, _i, _i, _i, _f32, _f32, _f32, _i),
    "alb_unsharp_mask_residual": (_i, _i, _i, _i, _i, _f32, _i),
    "alb_unsharp_mask_combine": (_i, _i, _i, _i, _i),
    "alb_add_noise_f32": (_i, _i, _i, _i),
    "alb_sharpen_gaussian_f32": (_i, _i, _i, _i, _f32),
    "alb_apply_salt_and_pepper_u8": (_i, _i, _i, _i, _i, _i, _i),
    "alb_create_directional_gradient_f32": (_i, _i, _i, _f, _f),
    "alb_apply_linear_illumination_f32": (_i, _i, _i, _i, _i, _f32, _f32, _f32),
    "alb_apply_gaussian_illumination_f32": (
        _i, _i, _i, _i, _i, _f32, _f32, _f32, _f32,
    ),
    "alb_calc_hist_u8": (_i, _i, _i, _i, _i, _i),
    "alb_get_histogram_bounds": (_i, _f, _i),
    "alb_create_contrast_lut": (_i, _i, _i, _i, _i, _i, _i),
    "alb_rgb_to_optical_density_f32": (_i, _i, _i, _f32, _f32),
    "alb_convolve_f32": (_i, _i, _i, _i, _i, _i, _i, _i, _i),
    "alb_separable_convolve_f32": (_i, _i, _i, _i, _i, _i, _i, _i, _i),
    "alb_box_blur_u8": (_i, _i, _i, _i, _i, _i),
    "alb_create_gaussian_kernel_input_array": (_i, _i),
    "alb_create_gaussian_kernel_1d": (_i, _i, _f, _i),
    "alb_create_gaussian_kernel_2d": (_i, _i, _i),
    "alb_create_motion_kernel": (_i, _i, _f, _f, _f, _f, _f),
    "alb_transpose_u8": (_i, _i, _i, _i, _i),
    "alb_vflip_u8": (_i, _i, _i, _i, _i),
    "alb_hflip_u8": (_i, _i, _i, _i, _i),
    "alb_rot90_u8": (_i, _i, _i, _i, _i, _i),
    "alb_copy_make_border_u8": (_i, _i, _i, _i, _i, _i, _i, _i, _i, _i, _i),
    "alb_affine_map_f32": (
        _f, _f, _f, _f, _f, _f, _i, _i, _i, _i,
    ),
    "alb_erode_u8": (_i, _i, _i, _i, _i, _i, _i, _i),
    "alb_dilate_u8": (_i, _i, _i, _i, _i, _i, _i, _i),
    "alb_scale_fields_f32": (_i, _i, _f32),
}

_fn = {
    name: getattr(lib, name) for name in _SIGNATURES
}
for _name, _argtypes in _SIGNATURES.items():
    _fn[_name].argtypes = list(_argtypes)
    _fn[_name].restype = None
del _name, _argtypes


def addr(a: np.ndarray) -> int:
    """Address of a C-contiguous array's first element."""
    return a.ctypes.data


def scratch_u8(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.uint8)


def scratch_i64(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def scratch_f64(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def empty(shape, dtype) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def contiguous(a, dtype) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=dtype)


def call(name: str, *args):
    return _fn[name](*args)