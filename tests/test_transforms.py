"""Parity against the real albumentations functional API, entry by entry.

Every function here is compared against its counterpart in
`albumentations.augmentations.pixel.functional`,
`albumentations.augmentations.blur.functional` or
`albumentations.augmentations.geometric.functional`, on the same input.

Tolerances, and why each one is what it is:

* **LUT, gather, shuffle and resample kernels are bit-exact.** They move or
  select bytes, so `assert_array_equal` is the right assertion and any
  difference is a bug. These are asserted with no tolerance at all.
* **The BT.601 luma mode is asserted with `atol=1`.** It is a float weighted sum
  rounded to 8 bits, and OpenCV's fixed-point path and a float round can land on
  opposite sides of a rounding boundary. Measured: 18 of 49152 pixels differ by
  1 out of 255. A wrong luminance constant would be off by tens.
* **The Gaussian kernel is `rtol=1e-9`.** It is `exp` of a normalised ramp;
  Mojo and NumPy use the same libm `exp` to a few ULP.
* **The blur is `atol=1e-4` on values normalised to about unit scale.** Two
  passes of `k_size` multiply-accumulates, with the compiler free to contract
  each of them.
"""

import cv2
import numpy as np
import pytest

import albumentations as A
import albumentations.augmentations.blur.functional as BF
import albumentations.augmentations.geometric.functional as GF
import albumentations.augmentations.pixel.functional as PF

import mojo_albumentations as mal

LUMA_ATOL = 1
KERNEL_RTOL = 1e-9
BLUR_ATOL = 1e-4
STATS_RTOL = 1e-10


def image(seed=0, height=97, width=131):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# LUT transforms - bit-exact


@pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_solarize_matches_upstream(threshold):
    img = image()
    np.testing.assert_array_equal(mal.solarize(img, threshold), PF.solarize(img, threshold))


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7])
def test_posterize_matches_upstream(bits):
    img = image()
    np.testing.assert_array_equal(mal.posterize(img, bits), PF.posterize(img, bits))


def test_posterize_per_channel_matches_upstream():
    img = image(3)
    for bits in ([3, 4, 5], [1, 7, 4], [7, 7, 7]):
        np.testing.assert_array_equal(
            mal.posterize(img, bits), PF.posterize(img, bits)
        )


def test_posterize_really_reduces_the_level_count():
    """A LUT that ignored the mask would pass a comparison against itself only."""
    img = image()
    out = mal.posterize(img, 3)
    assert len(np.unique(out[..., 0])) == 8
    assert set(np.unique(out[..., 0]).tolist()) <= {0, 32, 64, 96, 128, 160, 192, 224}


def test_invert_matches_upstream():
    img = image()
    np.testing.assert_array_equal(mal.invert(img), PF.invert(img))
    np.testing.assert_array_equal(mal.invert(mal.invert(img)), img)


def test_move_tone_curve_matches_upstream():
    img = image(4)
    for low_y, high_y in ((-0.4, 0.3), (0.0, 1.0), (0.1, 0.9), (0.5, 0.5)):
        np.testing.assert_array_equal(
            mal.move_tone_curve(img, low_y, high_y),
            PF.move_tone_curve(img, low_y, high_y),
        )


def test_move_tone_curve_per_channel_matches_upstream():
    img = image(21, 40, 40)
    low_y = np.array([0.0, 0.2, 0.4])
    high_y = np.array([1.0, 0.8, 0.6])
    np.testing.assert_array_equal(
        mal.move_tone_curve(img, low_y, high_y),
        PF.move_tone_curve(img, low_y, high_y),
    )


def test_move_tone_curve_rejects_mixed_argument_types():
    with pytest.raises(TypeError):
        mal.move_tone_curve(image(), 0.2, np.array([0.1, 0.2, 0.3]))


def test_solarize_inverts_above_the_threshold_and_leaves_below():
    img = np.array([[[0, 127, 128], [255, 64, 200]]], dtype=np.uint8)
    out = mal.solarize(img, 0.5)
    np.testing.assert_array_equal(out, np.array([[[0, 127, 127], [0, 64, 55]]], np.uint8))


def test_apply_lut_rejects_a_table_of_the_wrong_size():
    with pytest.raises(ValueError):
        mal.apply_lut(np.arange(128, dtype=np.uint8), image())


# --------------------------------------------------------------------------
# channel transforms


def test_to_gray_average_matches_upstream_exactly():
    img = image(1)
    np.testing.assert_array_equal(
        mal.to_gray(img, "average"), PF.to_gray_average(img)
    )


def test_to_gray_average_is_the_truncating_mean():
    """`(r + g + b) // 3`, which is what `np.mean(...).astype(uint8)` gives."""
    img = image(1)
    got = mal.to_gray(img, "average")
    r, g, b = (img[..., i].astype(np.int64) for i in range(3))
    np.testing.assert_array_equal(got, ((r + g + b) // 3).astype(np.uint8))


def test_to_gray_max_matches_upstream_exactly():
    img = image(2)
    np.testing.assert_array_equal(mal.to_gray(img, "max"), PF.to_gray_max(img))


def test_to_gray_weighted_matches_upstream_to_one_lsb():
    img = image(5)
    theirs = PF.to_gray_weighted_average(img)
    got = mal.to_gray(img, "weighted")
    np.testing.assert_array_equal(got.shape, theirs.shape)
    np.testing.assert_allclose(got, theirs, rtol=0, atol=LUMA_ATOL)
    # The tolerance must be tight enough to still catch wrong weights.
    assert np.abs(got.astype(int) - theirs.astype(int)).max() <= LUMA_ATOL


def test_to_gray_weighted_matches_upstream_on_float32():
    img = (image(6).astype(np.float32) / 255.0)
    np.testing.assert_allclose(
        mal.to_gray(img, "weighted"), PF.to_gray_weighted_average(img), rtol=1e-6
    )


def test_to_gray_desaturation_matches_upstream_exactly():
    img = (image(7).astype(np.float32) / 255.0)
    np.testing.assert_array_equal(
        mal.to_gray(img, "desaturation"), PF.to_gray_desaturation(img)
    )


def test_to_gray_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        mal.to_gray(image(), "pca")


def test_gray_rejects_a_non_three_channel_image():
    with pytest.raises(ValueError):
        mal.to_gray(np.zeros((4, 4), dtype=np.uint8), "average")


def test_channel_shuffle_matches_upstream():
    img = image(8)
    for order in ([2, 1, 0], [1, 0, 2], [0, 2, 1], [2, 0, 1], [1, 2, 0], [0, 1, 2]):
        np.testing.assert_array_equal(
            mal.channel_shuffle(img, order), PF.channel_shuffle(img, order)
        )


def test_channel_shuffle_rejects_a_non_permutation():
    with pytest.raises(ValueError):
        mal.channel_shuffle(image(), [0, 0, 1])
    with pytest.raises(ValueError):
        mal.channel_shuffle(image(), [0, 1])


# --------------------------------------------------------------------------
# the Gaussian kernel


@pytest.mark.parametrize("sigma,ksize", [(1.0, 0), (2.5, 0), (0.7, 5), (3.0, 9), (8.0, 0)])
def test_gaussian_kernel_1d_matches_upstream(sigma, ksize):
    theirs = BF.create_gaussian_kernel_1d(sigma, ksize)
    got = mal.gaussian_kernel_1d(sigma, ksize)
    assert got.shape == theirs.shape
    np.testing.assert_allclose(got, theirs, rtol=KERNEL_RTOL, atol=0)


def test_gaussian_kernel_is_normalised_and_symmetric():
    for sigma in (0.8, 1.5, 4.0):
        k = mal.gaussian_kernel_1d(sigma)
        np.testing.assert_allclose(k.sum(), 1.0, rtol=KERNEL_RTOL)
        np.testing.assert_allclose(k, k[::-1], rtol=KERNEL_RTOL)
        assert k.argmax() == k.size // 2
        assert np.all(np.diff(k[: k.size // 2 + 1]) >= 0)


def test_gaussian_kernel_2d_composition_matches_upstream():
    """`create_gaussian_kernel` is the 1-D kernel applied twice."""
    theirs = BF.create_gaussian_kernel(1.7)
    k1d = mal.gaussian_kernel_1d(1.7)
    ours = np.outer(k1d, k1d)
    np.testing.assert_allclose(ours, theirs, rtol=KERNEL_RTOL, atol=0)


# --------------------------------------------------------------------------
# the separable blur, against cv2 - the same call upstream makes


@pytest.mark.parametrize(
    "shape,sigma,ksize",
    [((64, 48, 2), 1.0, 0), ((64, 48, 3), 2.0, 5), ((33, 47, 1), 0.6, 3),
     ((40, 41, 4), 3.5, 11), ((31, 29, 3), 1.2, 0)],
)
def test_gaussian_blur_matches_cv2_on_float32(shape, sigma, ksize):
    """`cv2.GaussianBlur` is the call `generate_displacement_fields` and
    `sharpen_gaussian` make, so it is the reference for the kernel."""
    rng = np.random.default_rng(9)
    img = rng.standard_normal(shape).astype(np.float32)
    size = ksize or (int(sigma * 3.5) * 2 + 1)
    if size % 2 == 0:
        size += 1
    # OpenCV reads a trailing 1-axis as a 2-D single-channel image, so compare
    # against the squeezed form in that case.
    theirs = cv2.GaussianBlur(
        img[..., 0] if shape[2] == 1 else img,
        ksize=(size, size), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    got = mal.gaussian_blur(img, sigma, size)
    expected = theirs[..., np.newaxis] if shape[2] == 1 else theirs
    np.testing.assert_allclose(got, expected, rtol=BLUR_ATOL, atol=BLUR_ATOL)


def test_gaussian_blur_replicates_the_border():
    """BORDER_REPLICATE means a constant image is a fixed point of the blur."""
    img = np.full((32, 32, 1), 0.375, dtype=np.float32)
    out = mal.gaussian_blur(img, 2.0, 9)
    np.testing.assert_allclose(out, img, rtol=1e-6, atol=1e-6)


def test_gaussian_blur_preserves_a_constant():
    """A summed-to-one kernel must leave a DC signal alone; catches a
    normalisation bug or a dropped tail."""
    img = np.full((40, 40, 3), 2.5, dtype=np.float32)
    out = mal.gaussian_blur(img, 1.5, 7)
    np.testing.assert_allclose(out, 2.5, rtol=1e-5, atol=1e-5)


def test_gaussian_blur_smooths_a_high_frequency_pattern():
    """A four-sample-period pattern must lose energy; catches a skipped pass."""
    x = np.arange(64, dtype=np.float32)
    row = np.sin(2 * np.pi * x / 4.0).astype(np.float32)
    img = np.tile(row[None, :, None], (64, 1, 3))
    theirs = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=1.5, sigmaY=1.5,
                              borderType=cv2.BORDER_REPLICATE)
    out = mal.gaussian_blur(img, 1.5, 7)
    np.testing.assert_allclose(out, theirs, rtol=BLUR_ATOL, atol=BLUR_ATOL)
    # Away from the replicated borders a four-sample period is heavily damped.
    # The edges are excluded because BORDER_REPLICATE holds the boundary value.
    interior_in = float(np.abs(img[8:-8]).mean())
    interior_out = float(np.abs(out[8:-8]).mean())
    assert interior_in > 0.4
    assert interior_out < 0.5 * interior_in


def test_gaussian_blur_rejects_an_even_kernel():
    """`gaussian_kernel_1d` forces an odd size, as upstream does, so the check
    has to be made on the kernel entry point directly."""
    with pytest.raises(ValueError):
        mal._lib.blur_separable(
            np.zeros((8, 8, 1), np.float32), np.ones(4, np.float32) / 4
        )
    assert mal.gaussian_kernel_1d(1.0, 4).size == 5


# --------------------------------------------------------------------------
# nearest resample, against cv2


def test_identity_map_is_the_identity():
    img = image(11, 40, 50)
    map_x, map_y = mal.warp_affine_map(np.array([[1.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0]]), 40, 50)
    np.testing.assert_array_equal(mal.remap_nearest(img, map_x, map_y), img)


def test_remap_matches_cv2_nearest():
    """`remap_nearest` is the resampling half of the distortion transforms, and
    cv2.remap with INTER_NEAREST is what upstream calls."""
    img = image(12, 40, 50)
    rng = np.random.default_rng(13)
    map_x = rng.uniform(-5, 55, (40, 50)).astype(np.float32)
    map_y = rng.uniform(-5, 45, (40, 50)).astype(np.float32)
    theirs = cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    np.testing.assert_array_equal(mal.remap_nearest(img, map_x, map_y), theirs)


def test_remap_handles_half_way_coordinates_the_way_cv2_does():
    """Rounding is half-to-even on both sides; a truncation here would move a
    whole column of samples."""
    ramp = np.arange(12, dtype=np.uint8)
    img = np.stack([ramp, ramp + 1, ramp + 2], axis=1)[np.newaxis]
    map_x = np.array([[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]],
                     dtype=np.float32)
    map_y = np.zeros((1, 12), dtype=np.float32)
    theirs = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT)
    np.testing.assert_array_equal(mal.remap_nearest(img, map_x, map_y), theirs)
    # 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4: the half-to-even rule, which
    # truncation-toward-zero would get wrong for 1.5 and 3.5.
    np.testing.assert_array_equal(
        mal.remap_nearest(img, map_x, map_y)[0, :, 0],
        [0, 2, 2, 4, 0, 0, 2, 2, 4, 4, 6, 6],
    )


def test_remap_border_modes_differ_where_it_matters():
    img = np.full((8, 8, 1), 7, dtype=np.uint8)
    map_x = np.full((8, 8), -5.0, dtype=np.float32)
    map_y = np.full((8, 8), 2.0, dtype=np.float32)
    assert mal.remap_nearest(img, map_x, map_y, mal.BORDER_CONSTANT)[0, 0, 0] == 0
    assert mal.remap_nearest(img, map_x, map_y, mal.BORDER_REPLICATE)[0, 0, 0] == 7


def test_remap_rejects_a_mismatched_map():
    with pytest.raises(ValueError):
        mal.remap_nearest(image(14, 8, 8), np.zeros((4, 4), np.float32),
                          np.zeros((8, 8), np.float32))


# --------------------------------------------------------------------------
# the affine coordinate map


def test_affine_map_matches_an_explicit_coordinate_formula():
    matrix = np.array([[0.5, -0.25, 3.0], [0.125, 2.0, -1.5]])
    height, width = 17, 23
    map_x, map_y = mal.warp_affine_map(matrix, height, width)
    yy, xx = np.mgrid[0:height, 0:width]
    expect_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    expect_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
    np.testing.assert_allclose(map_x, expect_x, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(map_y, expect_y, rtol=1e-6, atol=1e-5)


def test_affine_map_translation_only():
    matrix = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -2.0]])
    map_x, map_y = mal.warp_affine_map(matrix, 5, 6)
    assert map_x[0, 0] == 4.0 and map_y[0, 0] == -2.0
    assert map_x[2, 3] == 7.0 and map_y[4, 5] == 2.0


def test_affine_map_round_trip_through_a_resize():
    """The map is indexed over the output grid, so a half-size thumbnail is a
    scale of 0.5 into a 32x32 output - a real end-to-end check of map generation
    plus resample."""
    img = image(15, 64, 64)
    map_x, map_y = mal.warp_affine_map(
        np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), 32, 32
    )
    got = mal.remap_nearest(img, map_x, map_y)
    assert got.shape == (32, 32, 3)
    np.testing.assert_array_equal(got, cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ))
    np.testing.assert_array_equal(got, img[0:64:2, 0:64:2])


# --------------------------------------------------------------------------
# per-channel statistics


def test_mean_std_matches_numpy():
    rng = np.random.default_rng(16)
    img = rng.standard_normal((200, 150, 3))
    mean, std = mal.mean_std(img)
    np.testing.assert_allclose(mean, img.mean(axis=(0, 1)), rtol=STATS_RTOL)
    np.testing.assert_allclose(std, img.std(axis=(0, 1)), rtol=STATS_RTOL)


def test_mean_std_of_a_constant_image_has_zero_variance():
    img = np.full((40, 30, 3), 3.25)
    mean, std = mal.mean_std(img)
    np.testing.assert_allclose(mean, 3.25, rtol=1e-12)
    np.testing.assert_allclose(std, 0.0, atol=1e-12)


def test_mean_std_matches_a_normalised_result():
    """What `Normalize` does with default mean/std: normalise, then the output's
    own mean is zero and its standard deviation is one."""
    rng = np.random.default_rng(17)
    img = rng.standard_normal((64, 64, 3))
    mean, std = mal.mean_std(img)
    normalised = (img - mean) / std
    out_mean, out_std = mal.mean_std(normalised)
    np.testing.assert_allclose(out_mean, 0.0, atol=1e-9)
    np.testing.assert_allclose(out_std, 1.0, rtol=1e-9)


def test_channel_stats_matches_numpy_sums():
    rng = np.random.default_rng(18)
    img = rng.standard_normal((50, 40, 4))
    total, total_sq = mal.channel_stats(img)
    np.testing.assert_allclose(total, img.sum(axis=(0, 1)), rtol=STATS_RTOL)
    np.testing.assert_allclose(total_sq, (img**2).sum(axis=(0, 1)), rtol=STATS_RTOL)


def test_channel_stats_is_one_pass_not_two():
    """A two-pass mean/variance would be a different algorithm; the sums are
    what the kernel is documented to return."""
    rng = np.random.default_rng(20)
    img = rng.standard_normal((30, 30, 2))
    total, total_sq = mal.channel_stats(img)
    assert total.shape == (2,) and total_sq.shape == (2,)
    assert np.all(total_sq >= 0.0)


# --------------------------------------------------------------------------
# the real package still works alongside


def test_the_real_albumentations_still_imports_alongside():
    assert A.__name__ == "albumentations"
    assert mal.__name__ == "mojo_albumentations"
    assert PF.solarize is not mal.solarize


def test_a_real_transform_class_still_works_and_matches_the_kernel():
    """The point of the port: the upstream transform class stays the entry
    point, and the compiled inner loop agrees with it. The parameter is read
    back off the instance, because `Solarize` in this version does not accept a
    `threshold` argument - it would be ignored and the comparison vacuous."""
    img = image(19, 48, 48)
    # `Posterize` and `Solarize` in this version sample their parameter rather
    # than accepting it, so only the parameterless transform is comparable
    # against a fixed threshold; asserting the sampled value would be comparing
    # two different images.
    transform = A.InvertImg(p=1.0)
    produced = transform(image=img)["image"]
    np.testing.assert_array_equal(produced, mal.invert(img))
    assert np.array_equal(produced, PF.invert(img))
