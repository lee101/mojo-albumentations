# mojo-albumentations

`mojo-albumentations` is the compute-oriented subset of
[albumentations](https://albumentations.ai): the per-pixel and
per-pixel-neighbourhood inner loops of the transform library, compiled into one
Mojo shared library.

The Python package is named `mojo_albumentations`, so it installs alongside the
real `albumentations` and the tests compare the two entry by entry.

```python
import albumentations as A
import mojo_albumentations as mal

img = ...  # (H, W, 3) uint8
mal.solarize(img, 0.6)            # == PF.solarize(img, 0.6)
mal.to_gray(img, "weighted")      # == PF.to_gray_weighted_average(img)
mal.gaussian_blur(field, 5.0)     # == cv2.GaussianBlur(..., BORDER_REPLICATE)

# or let the upstream transform class stay the entry point
A.Posterize(bits=3, p=1.0)(image=img)["image"]   # == mal.posterize(img, 3)
```

## Why this is the compute core

`albumentations` is 43,378 lines. Most of it is transform classes, parameter
samplers, replay buffers, bboxes, keypoints, composition and serialisation -
control flow. The numeric core is a small set of loops over image bytes, and
those are what is compiled here:

| upstream | loop |
| --- | --- |
| `solarize`, `posterize`, `invert`, `move_tone_curve`, `equalize` | a 256-entry table gathered once per byte |
| `to_gray_average`, `to_gray_max`, `to_gray_weighted_average` | a three-term channel reduction per pixel |
| `to_gray_desaturation` | `(max + min) / 2` per pixel, in float32 |
| `channel_shuffle` | a channel permutation per pixel |
| `create_gaussian_kernel_1d` | `exp` of a normalised ramp |
| `generate_displacement_fields`, `sharpen_gaussian` | a separable Gaussian blur, `BORDER_REPLICATE` |
| the resample of ElasticTransform / GridDistortion / Perspective | nearest resampling through a coordinate map |
| the map of `warp_affine` | an inverse-affine coordinate grid |
| `Normalize` | per-channel sum and sum of squares |

Everything else in the package stays upstream.

## Covered subset

| area | implemented API |
| --- | --- |
| LUT transforms | `solarize`, `posterize` (per-channel too), `invert`, `move_tone_curve`, `apply_lut` |
| Channel transforms | `to_gray` (average / max / weighted / desaturation), `desaturate`, `channel_shuffle` |
| Blur | `gaussian_kernel_1d`, `gaussian_blur`, `blur_separable` |
| Resampling | `remap_nearest`, `warp_affine_map` |
| Statistics | `channel_stats`, `mean_std` |
| Kernels | `alb_lut_apply_u8`, `alb_gray_u8`, `alb_gray_luma_f32`, `alb_desaturate_f32`, `alb_channel_shuffle_u8`, `alb_gaussian_kernel_1d`, `alb_blur_separable_f32`, `alb_remap_nearest_u8`, `alb_affine_map_f32`, `alb_channel_stats_f64` |

### Not implemented, and why

* **The transform classes.** `A.Solarize`, `A.ElasticTransform` and the other
  ~200 are parameter samplers, shape inference, bboxes, keypoints and replay
  bookkeeping. Construct them from the real package and hand the image here, or
  call these functions directly.
* **`shift_hsv` and therefore `HueSaturationValue` / `ColorJitter`.** Upstream
  delegates the colour-space conversion to `cv2.cvtColor`, and OpenCV's 8-bit
  RGB-to-HSV is not a formula - it is a fixed-point SIMD path. The measured
  round trip of that conversion is itself lossy by up to 5 out of 255, and a
  conventional float implementation lands a full hue step away on 0.4% of
  pixels. Writing one and calling it parity would be false, so it is left out
  rather than approximated.
* **`to_gray_pca`, `to_gray_from_lab`, `fancy_pca`** - OpenCV and eigen
  decompositions upstream, and the same argument applies.
* **Bilinear and bicubic resampling.** `albumentations` uses `cv2.INTER_LINEAR`
  and `cv2.INTER_CUBIC`; only the nearest path is compiled here.
* **`perspective` and `warp_affine` proper**, `OpticalDistortion`,
  `GridDistortion`, `ElasticTransform` - the resampling *step* is compiled, and
  the map generation for the affine case; the perspective matrix, the mesh
  generation and the cell-wise blending stay upstream.
* **Noise models, weather effects, `Equalize`, `CLAHE`, `HistogramMatching`** -
  RNG-driven or histogram-driven, or delegated to OpenCV.
* **NaN in a float32 channel.** The luma and the blur follow IEEE arithmetic,
  so NaN propagates, but no test covers it.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-albumentations.so
PYTHONPATH=python python -m pytest tests -q
PYTHONPATH=python python bench/bench.py
```

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo == 1.2.0.dev2026092605`); use the shared environment rather than
`pixi install`. Set `PYTHONPATH=python` when using the package outside a task.

## Performance

Best-of-five wall clock, same process. Every case checks its output against
`albumentations` or against `cv2` *before* it is timed. The baselines are the
fastest fair formulation for each operation: NumPy fancy indexing for the LUT and
the gather, `cv2.GaussianBlur` for the blur, `cv2.remap(INTER_NEAREST)` for the
resample, NumPy reductions for the statistics.

| case | reference | mojo-albumentations | result |
| --- | ---: | ---: | ---: |
| posterize, 1024x1024 | 18.07 ms | 3.14 ms | 5.75x faster |
| to_gray average+max, 1024x1024 | 171.43 ms | 8.41 ms | 20.39x faster |
| gaussian blur, 512x512x2, k=15 | 5.67 ms | 108.32 ms | **0.05x - 19x slower** |
| gaussian kernels x200 | 5.57 ms | 5.55 ms | 1.00x - parity |
| remap nearest, 512x512 | 23.83 ms | 9.81 ms | 2.43x faster |
| channel stats, 1024x1024x3 | 68.99 ms | 7.31 ms | 9.44x faster |

**The blur is nineteen times slower than OpenCV and there is no way round it
here.** `cv2.GaussianBlur` is a heavily optimised separable SIMD kernel, and in
this build a 15-tap kernel over half a megapixel takes 5.7 ms. This is a
straightforward scalar reference implementation: two passes, `ksize`
multiply-accumulates per sample, no vectorisation and no threading. The porting
brief is explicit that threading only pays above roughly two flops per byte, and
this is the case where a hand-written scalar kernel loses to a library that has
had twenty years of attention. It is here because it is the right reference to
have, and because a scalar, obviously-correct implementation of a two-pass
filter is what a port should ship when it cannot beat OpenCV - not because it
is fast.

The wins are the dispatch wins. A LUT gather, a three-term channel reduction, a
scattered nearest gather and a fused two-output reduction are all operations
where NumPy's generic machinery has to build index arrays, promote integers and
allocate temporaries, and a compiled loop does not. The channel statistics
kernel wins specifically because it produces the sum and the sum of squares in
one traversal; NumPy needs two passes.

The Gaussian kernel is parity, which is the right answer: it is 17 to 161 `exp`
calls.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit. `build/build.sh`
compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-albumentations.so`.

`python/mojo_albumentations` owns every array. Buffers cross the C ABI as 64-bit
addresses (`ctypes.c_int64`; `c_int` truncates and segfaults) and are rebuilt
inside the kernel as `Pointer[UInt8, ...]`, `Pointer[Float32, ...]` or
`Pointer[Float64, ...]`, which keeps the exported symbols non-parametric.

**The kernels are left serial.** Every one of them is a single streaming pass
over contiguous image data. The LUT, the channel reduction, the shuffle and the
resample are bandwidth-bound and a thread pool would add call overhead; the blur
is the only compute-heavy one, and it loses to OpenCV by so much that a threaded
version would still be slower. No `parallelize` is used: 1.2.0 cannot pass
pointers into a parallel body.

The layouts are explicit because getting them wrong is silent. The image is
`(H, W, C)` contiguous, so the blur treats it as one plane of `H` rows of `W*C`
samples, not `C` planes of `W`. The resample is indexed over the *output* grid
with the map's own shape and a separate source width, which is what lets a scale
or a crop be expressed as a map at all.

## Numerical parity

| operation | agreement | asserted |
| --- | --- | --- |
| LUT, shuffle, resample, desaturation | byte movement or float32 selects | exact - `assert_array_equal` |
| `to_gray_average` | `(r + g + b) // 3`, the truncating mean of `np.mean(...).astype(uint8)` | exact |
| `to_gray_max` | a select | exact |
| `to_gray_weighted_average`, uint8 | one 8-bit LSB. OpenCV's fixed-point path and a float round can land either side of a boundary: measured, 18 of 49152 pixels differ by 1 | `atol=1`, which a wrong luminance constant would fail by tens |
| `to_gray_weighted_average`, float32 | Mojo emits FMA, so a few ULP | `rtol=1e-6` |
| Gaussian kernel | the same libm `exp` to a few ULP | `rtol=1e-9` |
| Blur | two passes of `ksize` multiply-accumulates, each free to contract | `rtol=atol=1e-4` on unit-scale data |
| Channel statistics | a float64 running total; the mean and variance differ from NumPy's pairwise summation by accumulation order | `rtol=1e-10` |

## Tests

`tests/test_transforms.py`, 62 tests. Every function is compared against
`albumentations.augmentations.pixel.functional`,
`albumentations.augmentations.blur.functional` or
`albumentations.augmentations.geometric.functional` on the same input, and the
blur and the resample are compared against `cv2.GaussianBlur` and
`cv2.remap(INTER_NEAREST)` - the calls upstream itself makes.

Beyond the direct comparisons, the tests that would fail on a plausible bug:

* `posterize_really_reduces_the_level_count` - a LUT that ignored the mask would
  still match a reference built from the same broken LUT;
* `to_gray_average_is_the_truncating_mean` and
  `remap_handles_half_way_coordinates_the_way_cv2_does` - truncation instead of
  rounding, and half-to-even instead of half-away-from-zero, which moves a whole
  column of samples;
* `affine_map_matches_an_explicit_coordinate_formula` and
  `affine_map_round_trip_through_a_resize` - a transposed matrix or a swapped
  row stride, checked both against the formula and end to end;
* `gaussian_blur_replicates_the_border`, `gaussian_blur_preserves_a_constant`
  and `gaussian_blur_smooths_a_high_frequency_pattern` - a skipped pass, a
  dropped kernel tail or a bad normalisation each break one of these;
* `blur_separable_rejects_an_even_kernel` and
  `remap_rejects_a_mismatched_map` - the input validation;
* `mean_std_matches_a_normalised_result` - normalise with the computed mean and
  standard deviation, and the output's own must then be zero and one.

## License

MIT
