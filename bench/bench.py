"""Correctness-gated benchmark for mojo-albumentations.

Every case checks its output against the real `albumentations` functional API -
or against `cv2`, which is the call upstream itself makes - *before* it is
timed. The baselines are the fastest fair formulation for each operation, not a
Python loop that NumPy or OpenCV would never use:

* the LUT and the resample are compared against NumPy indexing, which is the
  vectorised equivalent;
* the blur is compared against `cv2.GaussianBlur`, the reference implementation
  and the one `generate_displacement_fields` calls;
* the channel statistics are compared against NumPy.
"""

from __future__ import annotations

import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import albumentations.augmentations.blur.functional as BF  # noqa: E402
import albumentations.augmentations.pixel.functional as PF  # noqa: E402

import mojo_albumentations as mal  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_lut(height=1024, width=1024):
    """`posterize`: a 256-entry table gathered over a megapixel."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    lut = np.arange(256, dtype=np.uint8)
    lut &= ~np.uint8(2 ** (8 - 3) - 1)
    flat = img.reshape(-1)

    np.testing.assert_array_equal(mal.apply_lut(lut, img), PF.posterize(img, 3))
    np.testing.assert_array_equal(lut[flat], PF.posterize(img, 3).reshape(-1))

    ref = _time(lambda: lut[flat], 5)
    mine = _time(lambda: mal.apply_lut(lut, img), 5)
    return f"posterize {height}x{width}", ref, mine


def bench_grayscale(height=1024, width=1024):
    """`to_gray_average`: a truncating three-term mean over a megapixel."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    flat = img.reshape(-1, 3).astype(np.uint16)

    np.testing.assert_array_equal(
        mal.to_gray(img, "average"), PF.to_gray_average(img)
    )
    np.testing.assert_array_equal(
        mal.to_gray(img, "max"), PF.to_gray_max(img)
    )

    def numpy_both():
        return (flat.sum(axis=1) // 3).astype(np.uint8), flat.max(axis=1).astype(np.uint8)

    ref = _time(numpy_both, 5)
    mine = _time(lambda: (mal.to_gray(img, "average"), mal.to_gray(img, "max")), 5)
    return f"to_gray average+max {height}x{width}", ref, mine


def bench_blur(height=512, width=512, nchan=2, sigma=2.0, ksize=0):
    """Separable Gaussian blur against `cv2.GaussianBlur`, the upstream call."""
    rng = np.random.default_rng(2)
    img = rng.standard_normal((height, width, nchan)).astype(np.float32)
    size = ksize or (int(sigma * 3.5) * 2 + 1)
    theirs = cv2.GaussianBlur(img, ksize=(size, size), sigmaX=sigma, sigmaY=sigma,
                              borderType=cv2.BORDER_REPLICATE)
    got = mal.gaussian_blur(img, sigma, size)
    np.testing.assert_allclose(got, theirs, rtol=1e-4, atol=1e-4)

    ref = _time(
        lambda: cv2.GaussianBlur(img, ksize=(size, size), sigmaX=sigma, sigmaY=sigma,
                                 borderType=cv2.BORDER_REPLICATE),
        5,
    )
    mine = _time(lambda: mal.gaussian_blur(img, sigma, size), 5)
    return f"gaussian blur {height}x{width}x{nchan} k={size}", ref, mine


def bench_gaussian_kernel(sigmas=200):
    def numpy_kernels():
        return [BF.create_gaussian_kernel_1d(s) for s in np.linspace(0.5, 20.0, sigmas)]

    np.testing.assert_allclose(
        mal.gaussian_kernel_1d(1.7), BF.create_gaussian_kernel_1d(1.7), rtol=1e-9
    )
    ref = _time(numpy_kernels, 3)
    mine = _time(lambda: [mal.gaussian_kernel_1d(s)
                          for s in np.linspace(0.5, 20.0, sigmas)], 3)
    return f"gaussian kernels x{sigmas}", ref, mine


def bench_remap(height=512, width=512, scale=1.05):
    """Nearest resample through an affine map: a scattered gather."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    matrix = np.array([
        [np.cos(0.05), -np.sin(0.05), 6.0],
        [np.sin(0.05), np.cos(0.05), -4.0],
    ]) * scale
    map_x, map_y = mal.warp_affine_map(matrix, height, width)
    theirs = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT)
    np.testing.assert_array_equal(mal.remap_nearest(img, map_x, map_y), theirs)

    rx, ry = np.round(map_x), np.round(map_y)
    inside = (rx >= 0) & (rx < width) & (ry >= 0) & (ry < height)
    xi = np.clip(rx.astype(np.intp), 0, width - 1)
    yi = np.clip(ry.astype(np.intp), 0, height - 1)

    def numpy_gather():
        # BORDER_CONSTANT: samples outside the source are zero, not clamped.
        return np.where(inside[..., np.newaxis], img[yi, xi], 0)

    np.testing.assert_array_equal(numpy_gather(), theirs)
    ref = _time(numpy_gather, 5)
    mine = _time(lambda: mal.remap_nearest(img, map_x, map_y), 5)
    return f"remap nearest {height}x{width}", ref, mine


def bench_channel_stats(height=1024, width=1024, nchan=3):
    """Per-channel sum and sum of squares, against NumPy."""
    rng = np.random.default_rng(4)
    img = rng.standard_normal((height, width, nchan))
    total, total_sq = mal.channel_stats(img)
    np.testing.assert_allclose(total, img.sum(axis=(0, 1)), rtol=1e-10)
    np.testing.assert_allclose(total_sq, (img**2).sum(axis=(0, 1)), rtol=1e-10)

    def numpy_sums():
        return img.sum(axis=(0, 1)), (img**2).sum(axis=(0, 1))

    ref = _time(numpy_sums, 5)
    mine = _time(lambda: mal.channel_stats(img), 5)
    return f"channel stats {height}x{width}x{nchan}", ref, mine


def main():
    print(f"{'case':<38}{'reference':>12}{'mojo-albumentations':>20}{'ratio':>10}")
    print(f"{'':<38}{'':>12}{'':>20}{'(ref/mojo)':>10}")
    print("-" * 80)
    for fn in (bench_lut, bench_grayscale, bench_blur, bench_gaussian_kernel,
               bench_remap, bench_channel_stats):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        verdict = "faster" if ratio > 1.02 else ("slower" if ratio < 0.98 else "parity")
        print(f"{label:<38}{ref * 1e3:>10.2f}ms{got * 1e3:>18.2f}ms{ratio:>9.2f}x  {verdict}")
    print("\nreferences: numpy fancy indexing, cv2.GaussianBlur, "
          "cv2.remap(INTER_NEAREST), numpy reductions")


if __name__ == "__main__":
    main()
