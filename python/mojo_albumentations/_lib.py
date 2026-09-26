"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64``; ``c_int`` truncates them
and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-albumentations.so"

_A = ctypes.c_int64
_I = ctypes.c_int64
_D = ctypes.c_double

GRAY_AVERAGE, GRAY_MAX, GRAY_WEIGHTED = 0, 1, 2
BORDER_REPLICATE, BORDER_CONSTANT = 0, 1


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.alb_lut_apply_u8.restype = None
    lib.alb_lut_apply_u8.argtypes = [_A, _A, _A, _I]
    lib.alb_gray_u8.restype = None
    lib.alb_gray_u8.argtypes = [_A, _A, _I, _I]
    lib.alb_gray_luma_f32.restype = None
    lib.alb_gray_luma_f32.argtypes = [_A, _A, _I]
    lib.alb_desaturate_f32.restype = None
    lib.alb_desaturate_f32.argtypes = [_A, _A, _I]
    lib.alb_channel_shuffle_u8.restype = None
    lib.alb_channel_shuffle_u8.argtypes = [_A, _A, _I, _I, _I, _I]
    lib.alb_gaussian_kernel_1d.restype = None
    lib.alb_gaussian_kernel_1d.argtypes = [_A, _I, _D, _A]
    lib.alb_blur_separable_f32.restype = None
    lib.alb_blur_separable_f32.argtypes = [_A, _A, _A, _A, _I, _I, _I, _I, _I]
    lib.alb_remap_nearest_u8.restype = None
    lib.alb_remap_nearest_u8.argtypes = [_A, _A, _A, _A, _I, _I, _I, _I, _I, _I]
    lib.alb_affine_map_f32.restype = None
    lib.alb_affine_map_f32.argtypes = [_D, _D, _D, _D, _D, _D, _I, _I, _A, _A]
    lib.alb_channel_stats_f64.restype = None
    lib.alb_channel_stats_f64.argtypes = [_A, _A, _I, _I]
    return lib


lib = _load()
_lut_apply = lib.alb_lut_apply_u8
_gray = lib.alb_gray_u8
_gray_luma = lib.alb_gray_luma_f32
_desaturate = lib.alb_desaturate_f32
_shuffle = lib.alb_channel_shuffle_u8
_gauss1d = lib.alb_gaussian_kernel_1d
_blur = lib.alb_blur_separable_f32
_remap = lib.alb_remap_nearest_u8
_affine_map = lib.alb_affine_map_f32
_channel_stats = lib.alb_channel_stats_f64


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _u8(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.uint8)


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def apply_lut(lut, img) -> np.ndarray:
    """Apply a 256-entry uint8 table to a uint8 image. Bit-exact."""
    lut = _u8(lut).reshape(-1)
    if lut.size != 256:
        raise ValueError("the lookup table must have 256 entries")
    buf = _u8(img)
    shape = buf.shape
    flat = buf.reshape(-1)
    out = np.empty(flat.size, dtype=np.uint8)
    if flat.size:
        _lut_apply(_addr(lut), _addr(flat), _addr(out), flat.size)
    return out.reshape(shape)


def gray_from_rgb(img, mode: int = GRAY_WEIGHTED) -> np.ndarray:
    """Per-pixel channel reduction of a 3-channel uint8 image.

    ``mode`` selects `to_gray_average`, `to_gray_max` or
    `to_gray_weighted_average`. Returns a 2-D uint8 array, as upstream does.
    """
    buf = _u8(img)
    if buf.ndim != 3 or buf.shape[2] != 3:
        raise ValueError("expected a 3-channel (H, W, 3) uint8 image")
    height, width, _ = buf.shape
    out = np.empty((height, width), dtype=np.uint8)
    _gray(_addr(buf), _addr(out), height * width, mode)
    return out


def gray_luma_f32(img) -> np.ndarray:
    """`to_gray_weighted_average` of a 3-channel float32 image."""
    buf = _f32(img)
    if buf.ndim != 3 or buf.shape[2] != 3:
        raise ValueError("expected a 3-channel (H, W, 3) float32 image")
    height, width, _ = buf.shape
    out = np.empty((height, width), dtype=np.float32)
    _gray_luma(_addr(buf), _addr(out), height * width)
    return out


def desaturate(img) -> np.ndarray:
    """`to_gray_desaturation` of a 3-channel float32 image. Bit-exact."""
    buf = _f32(img)
    if buf.ndim != 3 or buf.shape[2] != 3:
        raise ValueError("expected a 3-channel (H, W, 3) float32 image")
    height, width, _ = buf.shape
    out = np.empty((height, width), dtype=np.float32)
    _desaturate(_addr(buf), _addr(out), height * width)
    return out


def channel_shuffle(img, order) -> np.ndarray:
    """`channel_shuffle`: ``out[..., i] = img[..., order[i]]``. Bit-exact."""
    buf = _u8(img)
    if buf.ndim != 3 or buf.shape[2] != 3:
        raise ValueError("expected a 3-channel (H, W, 3) uint8 image")
    order = list(order)
    if len(order) != 3 or sorted(order) != [0, 1, 2]:
        raise ValueError("order must be a permutation of (0, 1, 2)")
    height, width, _ = buf.shape
    out = np.empty_like(buf)
    _shuffle(_addr(buf), _addr(out), height * width, *order)
    return out


def gaussian_kernel_1d(x, sigma: float) -> np.ndarray:
    """`create_gaussian_kernel_1d` of a coordinate ramp: normalised exp ramp."""
    x = _f64(x).reshape(-1)
    out = np.empty(x.size, dtype=np.float64)
    if x.size:
        _gauss1d(_addr(x), x.size, ctypes.c_double(sigma), _addr(out))
    return out


def blur_separable(
    img, kernel, border: int = BORDER_REPLICATE, out=None
) -> np.ndarray:
    """Separable Gaussian blur of a float32 image, two passes.

    ``kernel`` is the 1-D profile, as produced by `gaussian_kernel_1d`. The
    vertical pass runs on the horizontal result, as a separable filter does, so
    the border handling in the second pass sees the intermediate.
    """
    src = _f32(img)
    if src.ndim != 3:
        raise ValueError("expected an (H, W, C) float32 image")
    kern = _f32(kernel).reshape(-1)
    if kern.size % 2 == 0:
        raise ValueError("the kernel length must be odd")
    height, width, nchan = src.shape
    tmp = np.empty_like(src)
    dst = np.empty_like(src) if out is None else out
    if height and width and nchan and kern.size:
        _blur(
            _addr(src), _addr(tmp), _addr(dst), _addr(kern),
            height, width, nchan, kern.size, border,
        )
    return dst


def remap_nearest(img, map_x, map_y, border: int = BORDER_CONSTANT) -> np.ndarray:
    """Nearest resample of a uint8 image through a coordinate map. Bit-exact.

    The map is indexed over the *output* grid, so its shape is the output shape
    and it need not match the source image - that is what makes a scale or a
    crop expressible as a map.
    """
    buf = _u8(img)
    if buf.ndim != 3:
        raise ValueError("expected an (H, W, C) uint8 image")
    src_h, src_w, nchan = buf.shape
    mx = _f32(map_x)
    my = _f32(map_y)
    if mx.ndim != 2 or my.shape != mx.shape:
        raise ValueError("the coordinate maps must be 2-D and the same shape")
    height, width = mx.shape
    mx = mx.reshape(-1)
    my = my.reshape(-1)
    out = np.empty((height, width, nchan), dtype=np.uint8)
    if height and width and src_h and src_w:
        _remap(
            _addr(buf), _addr(mx), _addr(my), _addr(out),
            height, width, src_h, src_w, nchan, border,
        )
    return out


def affine_map(matrix, height: int, width: int):
    """The inverse-affine coordinate grid for a 2x3 forward affine.

    Returns ``(map_x, map_y)`` as float32, each of shape ``(height, width)``:
    the sample coordinate for every output pixel.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    map_x = np.empty(height * width, dtype=np.float32)
    map_y = np.empty(height * width, dtype=np.float32)
    if height and width:
        _affine_map(
            ctypes.c_double(matrix[0, 0]), ctypes.c_double(matrix[0, 1]),
            ctypes.c_double(matrix[0, 2]), ctypes.c_double(matrix[1, 0]),
            ctypes.c_double(matrix[1, 1]), ctypes.c_double(matrix[1, 2]),
            height, width, _addr(map_x), _addr(map_y),
        )
    return map_x.reshape(height, width), map_y.reshape(height, width)


def channel_stats(img):
    """Per-channel ``(sum, sumsq)`` of a float64 image, one pass.

    This is the reduction `Normalize` needs for the mean and standard deviation.
    """
    buf = _f64(img)
    if buf.ndim != 3:
        raise ValueError("expected an (H, W, C) float64 image")
    height, width, nchan = buf.shape
    out = np.empty(2 * nchan, dtype=np.float64)
    if height and width and nchan:
        _channel_stats(_addr(buf), _addr(out), height * width, nchan)
    return out[:nchan], out[nchan:]
