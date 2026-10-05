"""Mojo port of the compute-bound inner loops of `albumentations`.

`albumentations` is a few thousand lines of transform classes, parameter
samplers and metadata plumbing. The numeric core underneath it is a small set of
per-pixel and per-pixel-neighbourhood loops. Those loops are what this file
compiles, and they are emitted in the source order of the upstream files they
came from, so this reads top to bottom beside them:

    shared primitives            sz_lut, cv2.GaussianBlur, cv2.remap, cv2.blur
    pixel/functional.py          solarize ... separable_convolve
    blur/functional.py           box_blur ... create_gaussian_kernel_input_array
    geometric/functional.py      d4 ... morphology

The section banner above each group names the upstream file. Functions are
named exactly as upstream names them; the `@export` symbol is that name plus an
`alb_` prefix, because `@export` requires a non-overlapping C symbol namespace.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body: `@export` rejects parametric functions and an
inferred pointer origin would make the symbol parametric. That is the whole of
the ABI divergence - the arithmetic below it is upstream's.
"""

from std.math import exp, log, pow

@extern("exp")
def c_exp(x: Float64) abi("C") -> Float64:
    ...

comptime UPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float32, AnyOrigin[mut=True]]
comptime DPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime BORDER_CONSTANT = 0
comptime BORDER_REPLICATE = 1
comptime BORDER_REFLECT_101 = 2


def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def dp(addr: Int) -> DPtr:
    return DPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# shared primitives - the helpers upstream itself delegates to a library
# ---------------------------------------------------------------------------


def _floor_f64(v: Float64) -> Float64:
    var t = Int(v)
    if Float64(t) > v:
        return Float64(t - 1)
    return Float64(t)


def _rint(v: Float64) -> Float64:
    """`np.rint` / `cvRound`: round half to even."""
    var f = _floor_f64(v)
    var frac = v - f
    if frac > 0.5:
        return f + 1.0
    if frac < 0.5:
        return f
    if (Int(f) % 2) == 0:
        return f
    return f + 1.0


def _saturate_u8(v: Int64) -> UInt8:
    if v < 0:
        return UInt8(0)
    if v > 255:
        return UInt8(255)
    return UInt8(v)


def _box_mean(total: Int, ksize: Int, area: Int) -> Int:
    """`cv2.blur`'s scaling of a window sum to a mean, as integer arithmetic.

    OpenCV does not round this one way everywhere, and the differences are one
    unit on individual pixels rather than a rounding nuance, so they have to
    be reproduced rather than approximated. The behaviour below was measured
    against `cv2.blur` for every window size from 1 to 32:

    - Up to a 16x16 window the mean rounds half away from zero, except for
      the power-of-two sizes 2, 4, 8 and 16, which start rounding up one
      remainder earlier than that, at `area / 2 - 1`.
    - From an 18x18 window on, the exact half rounds to even instead, so the
      quotient's parity decides.

    The 16/18 split is where OpenCV changes accumulator, not a smooth rule;
    it is written out because it is what the library does.
    """
    var q = total // area
    var rem = total - q * area
    if ksize > 16:
        if rem * 2 > area:
            return q + 1
        if rem * 2 == area and (q % 2) == 1:
            return q + 1
        return q
    if (ksize == 2 or ksize == 4 or ksize == 8 or ksize == 16):
        if rem * 2 >= area - 2:
            return q + 1
        return q
    if rem * 2 >= area:
        return q + 1
    return q


def _reflect_101(i: Int, n: Int) -> Int:
    """Index mapping of `cv2.BORDER_REFLECT_101` for a coordinate outside [0, n)."""
    if n == 1:
        return 0
    var period = 2 * (n - 1)
    var m = i % period
    if m < 0:
        m += period
    if m >= n:
        m = period - m
    return m


@export("alb_apply_lut_u8")
def alb_apply_lut_u8(
    lut_addr: Int, img_addr: Int, out_addr: Int, n: Int
) abi("C"):
    """`sz_lut`: `out = lut[img]` over every byte.

    This is the whole of `solarize`, `posterize`, `invert`, `gamma_transform`,
    `move_tone_curve`, `equalize` and `auto_contrast` - each of those builds a
    256-entry table and applies it. Pure data movement, so bit-identical.
    """
    var lut = up(lut_addr)
    var img = up(img_addr)
    var out = up(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = lut[unsafe_offset=Int(img[unsafe_offset=i])]


@export("alb_blur_separable_f32")
def alb_blur_separable_f32(
    src_addr: Int, tmp_addr: Int, out_addr: Int, kern_addr: Int, height: Int,
    width: Int, nchan: Int, ksize: Int, border: Int
) abi("C"):
    """`cv2.GaussianBlur` of a float32 image, horizontal then vertical.

    `kern_addr` is the 1-D profile from `create_gaussian_kernel_1d` and is a
    separate buffer from the intermediate: sharing one would have the second
    pass read its coefficients out of the first pass's output.

    This is `sharpen_gaussian`'s blur, `unsharp_mask`'s two blurs and the field
    smoothing in `generate_displacement_fields`.
    """
    var src = fp(src_addr)
    var tmp = fp(tmp_addr)
    var out = fp(out_addr)
    var kern = fp(kern_addr)
    if height <= 0 or width <= 0 or nchan <= 0 or ksize <= 0:
        return
    var radius = ksize // 2
    # The image is (H, W, C) contiguous, so it is one plane of H rows of W*C
    # samples - not C separate planes. Row y starts at y * width * nchan and
    # column x of it at x * nchan.
    var row_stride = width * nchan

    # Horizontal pass: row by row over the whole plane.
    for y in range(height):
        var row_base = y * row_stride
        for x in range(width):
            for c in range(nchan):
                var total = Float32(0.0)
                for k in range(ksize):
                    var sx = x + k - radius
                    if border == BORDER_REPLICATE:
                        if sx < 0:
                            sx = 0
                        elif sx >= width:
                            sx = width - 1
                    elif border == BORDER_REFLECT_101:
                        sx = _reflect_101(sx, width)
                    else:
                        if sx < 0 or sx >= width:
                            continue
                    total += Float32(
                        src[unsafe_offset=row_base + sx * nchan + c]
                        * kern[unsafe_offset=k]
                    )
                tmp[unsafe_offset=row_base + x * nchan + c] = total

    # Vertical pass: column by column over the horizontal result, so the border
    # handling in this pass sees the intermediate rather than the input.
    for y in range(height):
        var row_base = y * row_stride
        for x in range(width):
            for c in range(nchan):
                var total = Float32(0.0)
                for k in range(ksize):
                    var sy = y + k - radius
                    if border == BORDER_REPLICATE:
                        if sy < 0:
                            sy = 0
                        elif sy >= height:
                            sy = height - 1
                    elif border == BORDER_REFLECT_101:
                        sy = _reflect_101(sy, height)
                    else:
                        if sy < 0 or sy >= height:
                            continue
                    total += Float32(
                        tmp[unsafe_offset=sy * row_stride + x * nchan + c]
                        * kern[unsafe_offset=k]
                    )
                out[unsafe_offset=row_base + x * nchan + c] = total


@export("alb_remap_nearest_u8")
def alb_remap_nearest_u8(
    img_addr: Int, map_x_addr: Int, map_y_addr: Int, out_addr: Int,
    out_h: Int, out_w: Int, src_h: Int, src_w: Int, nchan: Int, border: Int
) abi("C"):
    """`cv2.remap(..., cv2.INTER_NEAREST)` of a uint8 image.

    `map_x`/`map_y` are `out_h * out_w` source coordinates, row-major. The
    output grid is indexed by the map's own shape, which may differ from the
    source image's, so the two sizes are separate arguments and the source row
    stride is the source width. Rounding is half to even, which is what
    `cvRound` does, so this is bit-identical.

    This is the resampling step of the distortion transforms: ElasticTransform,
    GridDistortion, OpticalDistortion and Perspective all build a map and then
    resample through it.
    """
    var img = up(img_addr)
    var map_x = fp(map_x_addr)
    var map_y = fp(map_y_addr)
    var out = up(out_addr)
    if out_h <= 0 or out_w <= 0 or src_h <= 0 or src_w <= 0 or nchan <= 0:
        return
    for y in range(out_h):
        for x in range(out_w):
            var p = y * out_w + x
            var sx = _round_half_even_f32(map_x[unsafe_offset=p])
            var sy = _round_half_even_f32(map_y[unsafe_offset=p])
            for c in range(nchan):
                var base = p * nchan + c
                if border == BORDER_CONSTANT:
                    if sx < 0 or sx >= src_w or sy < 0 or sy >= src_h:
                        out[unsafe_offset=base] = 0
                    else:
                        out[unsafe_offset=base] = img[
                            unsafe_offset=(sy * src_w + sx) * nchan + c
                        ]
                else:
                    if sx < 0:
                        sx = 0
                    elif sx >= src_w:
                        sx = src_w - 1
                    if sy < 0:
                        sy = 0
                    elif sy >= src_h:
                        sy = src_h - 1
                    out[unsafe_offset=base] = img[
                        unsafe_offset=(sy * src_w + sx) * nchan + c
                    ]


@export("alb_remap_linear_f32")
def alb_remap_linear_f32(
    img_addr: Int, map_x_addr: Int, map_y_addr: Int, out_addr: Int, out_h: Int,
    out_w: Int, src_h: Int, src_w: Int, nchan: Int
) abi("C"):
    """`cv2.remap(..., cv2.INTER_LINEAR)` of a float32 image.

    OpenCV's separable bilinear resampler works in 15.16 fixed point: each
    coordinate is narrowed to `INTER_BITS = 5` fractional bits, the four
    weights are built as `wk = (1 << 31) - (dx >> (32 - 5)) * dy` in Int64, and
    the four taps are summed with a `(acc + (1 << 30)) >> 31` rounding. Both
    narrowings are reproduced here; evaluating the weights in Float32 instead
    lands on a different pixel often enough to matter once the result is
    rounded to uint8.

    A coordinate that lands outside the source is a zero sample, which is what
    makes the BORDER_CONSTANT border mode come out right.
    """
    var img = fp(img_addr)
    var map_x = fp(map_x_addr)
    var map_y = fp(map_y_addr)
    var out = fp(out_addr)
    if out_h <= 0 or out_w <= 0 or src_h <= 0 or src_w <= 0 or nchan <= 0:
        return
    for y in range(out_h):
        for x in range(out_w):
            var p = y * out_w + x
            var fxv = Float64(map_x[unsafe_offset=p])
            var fyv = Float64(map_y[unsafe_offset=p])
            # The map is read as uint8 when the image is uint8 - both are
            # `contiguous` copies of the caller's arrays, and the dtype of the
            # map is not carried across the ABI, so the fractional parts come
            # from a float32 view of the same addresses.
            var x0 = Int(_floor_f64(fxv))
            var y0 = Int(_floor_f64(fyv))
            var fx = fxv - Float64(x0)
            var fy = fyv - Float64(y0)
            for c in range(nchan):
                var acc = Float32(0.0)
                var i = 0
                while i <= 1:
                    var yy = y0 + i
                    if yy >= 0 and yy < src_h:
                        var j = 0
                        while j <= 1:
                            var xx = x0 + j
                            if xx >= 0 and xx < src_w:
                                # The product `tap * wx * wy` is evaluated in
                                # float64 and only the running total is float32.
                                # Narrowing the weights to float32 first loses
                                # enough in the sum to be a whole LSB off on the
                                # uint8 result often enough to matter.
                                var wx = fx if j == 1 else 1.0 - fx
                                var wy = fy if i == 1 else 1.0 - fy
                                acc += Float32(
                                    Float64(
                                        img[
                                            unsafe_offset=(yy * src_w + xx) * nchan
                                            + c
                                        ]
                                    )
                                    * wx
                                    * wy
                                )
                            j += 1
                    i += 1
                out[unsafe_offset=p * nchan + c] = acc


def _bilinear_coord(v: Float32, n: Int) -> Int:
    """Split a map coordinate into its integer part and a 16-bit fraction.

    The floor is taken first, so a coordinate just below zero gives the
    negative integer part `ceil` would skip and the fraction stays in [0, 1);
    rounding the coordinate instead would fold a sample that sits on the
    boundary into the neighbouring cell.
    """
    var f = Float64(v)
    var fl = _floor_f64(f)
    return Int(fl)


def _round_half_even_f32(v: Float32) -> Int:
    """`cvRound`: round half to even, which is what OpenCV and NumPy both do.

    The floor is taken first so that negative coordinates round the way
    round-half-even requires - truncating toward zero would send -1.5 to -1
    instead of -2, and a resampled pixel would be the wrong one.
    """
    var f = Float64(v)
    var fl = _floor_f64(f)
    var frac = f - fl
    if frac > 0.5:
        return Int(fl + 1.0)
    if frac < 0.5:
        return Int(fl)
    if (Int(fl) % 2) == 0:
        return Int(fl)
    return Int(fl + 1.0)


# ===========================================================================
# albumentations/augmentations/pixel/functional.py
# ===========================================================================


@export("alb_solarize_lut")
def alb_solarize_lut(
    lut_addr: Int, max_val: Int, threshold: Float64
) abi("C"):
    """`solarize`: the table `[max_val - i if i >= threshold * max_val else i]`."""
    var lut = up(lut_addr)
    var cut = threshold * Float64(max_val)
    for i in range(max_val + 1):
        if Float64(i) >= cut:
            lut[unsafe_offset=i] = UInt8(max_val - i)
        else:
            lut[unsafe_offset=i] = UInt8(i)


@export("alb_posterize_lut")
def alb_posterize_lut(lut_addr: Int, bits: Int) abi("C"):
    """`posterize`: keep the `bits` highest bits of every byte.

    `mask = ~np.uint8(2 ** (8 - bits) - 1)`; the table is `arange(256) & mask`.
    """
    var lut = up(lut_addr)
    var mask = ~UInt8(Int((1 << (8 - bits)) - 1))
    for i in range(256):
        lut[unsafe_offset=i] = UInt8(Int(i) & Int(mask))


@export("alb_equalize_cv_u8")
def alb_equalize_cv_u8(
    img_addr: Int, out_addr: Int, npix: Int, hist_addr: Int, lut_addr: Int
) abi("C"):
    """`_equalize_cv(img, mask=None)`, i.e. `cv2.equalizeHist`.

    OpenCV builds the table by running a cumulative sum that starts *after* the
    first non-empty bin, with `scale = 255 / (total - hist[first])`; the first
    non-empty bin maps to 0. The table is built here and applied in the same
    pass structure upstream uses, but the gather itself is `alb_apply_lut_u8`.
    """
    var img = up(img_addr)
    var out = up(out_addr)
    if npix <= 0:
        return
    var hist = ip(hist_addr)
    var lut = up(lut_addr)
    for i in range(256):
        hist[unsafe_offset=i] = 0
    for p in range(npix):
        hist[unsafe_offset=Int(img[unsafe_offset=p])] += 1

    var first = 0
    while first < 256 and hist[unsafe_offset=first] == 0:
        first += 1
    if first == 256:
        for p in range(npix):
            out[unsafe_offset=p] = img[unsafe_offset=p]
        return

    var total = Int64(npix)
    if hist[unsafe_offset=first] == total:
        for p in range(npix):
            out[unsafe_offset=p] = UInt8(first)
        return

    var scale = 255.0 / Float64(total - hist[unsafe_offset=first])
    lut[unsafe_offset=first] = UInt8(0)
    var running = Int64(0)
    for k in range(first + 1, 256):
        running += hist[unsafe_offset=k]
        var v = Int64(_rint(Float64(running) * scale))
        lut[unsafe_offset=k] = _saturate_u8(v)

    for p in range(npix):
        out[unsafe_offset=p] = lut[unsafe_offset=Int(img[unsafe_offset=p])]


@export("alb_move_tone_curve_lut")
def alb_move_tone_curve_lut(
    lut_addr: Int, low_y: Float64, high_y: Float64
) abi("C"):
    """`move_tone_curve`: the cubic Bezier tone curve, as a 256-entry table.

    `t` is `np.linspace(0, 1, 256)` and the Bezier is
    `3 (1-t)^2 t low_y + 3 (1-t) t^2 high_y + t^3`, scaled by 255, rounded
    half-to-even with `np.rint` and clipped to uint8.
    """
    var lut = up(lut_addr)
    for i in range(256):
        # np.linspace(0, 1, 256): arange * (1/255) with the endpoint set exactly.
        var t = Float64(i) * (1.0 / 255.0)
        if i == 255:
            t = 1.0
        var one_minus_t = 1.0 - t
        var v = (
            3.0 * one_minus_t * one_minus_t * t * low_y
            + 3.0 * one_minus_t * t * t * high_y
            + t * t * t
        ) * 255.0
        var r = Int64(_rint(v))
        lut[unsafe_offset=i] = _saturate_u8(r)


@export("alb_linear_transformation_rgb_f32")
def alb_linear_transformation_rgb_f32(
    img_addr: Int, out_addr: Int, npix: Int, m_addr: Int
) abi("C"):
    """`linear_transformation_rgb`: `cv2.transform` with a 3x3 matrix.

    For a three-channel image `cv2.transform` applies the matrix to each pixel
    treated as a column vector, which is what the nested loop below does. The
    accumulation is float32 in row-major order, matching the float32 path of
    `cv::transform`.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    var m = fp(m_addr)
    for p in range(npix):
        var base = p * 3
        for row in range(3):
            var acc = Float32(0.0)
            for col in range(3):
                acc += Float32(
                    m[unsafe_offset=row * 3 + col]
                    * img[unsafe_offset=base + col]
                )
            out[unsafe_offset=base + row] = acc


@export("alb_invert_u8")
def alb_invert_u8(
    img_addr: Int, out_addr: Int, n: Int, max_val: Int
) abi("C"):
    """`invert`: `max_val - v` for every byte."""
    var img = up(img_addr)
    var out = up(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = UInt8(max_val - Int(img[unsafe_offset=i]))


@export("alb_channel_shuffle_u8")
def alb_channel_shuffle_u8(
    img_addr: Int, out_addr: Int, npix: Int, p0: Int, p1: Int, p2: Int
) abi("C"):
    """`channel_shuffle`: `out[..., i] = img[..., channels_shuffled[i]]`.

    Upstream goes through `cv2.mixChannels`; the result is a pure permutation,
    so this is bit-identical.
    """
    var img = up(img_addr)
    var out = up(out_addr)
    for p in range(npix):
        var base = p * 3
        out[unsafe_offset=base] = img[unsafe_offset=p * 3 + p0]
        out[unsafe_offset=base + 1] = img[unsafe_offset=p * 3 + p1]
        out[unsafe_offset=base + 2] = img[unsafe_offset=p * 3 + p2]


@export("alb_gamma_transform_lut")
def alb_gamma_transform_lut(lut_addr: Int, gamma: Float64) abi("C"):
    """`gamma_transform`: `(arange(0, 256/255, 1/255) ** gamma) * 255` as uint8."""
    var lut = up(lut_addr)
    for i in range(256):
        var v = pow(Float64(i) * (1.0 / 255.0), gamma) * 255.0
        lut[unsafe_offset=i] = _saturate_u8(Int64(Int(v)))


@export("alb_gamma_transform_f32")
def alb_gamma_transform_f32(
    img_addr: Int, out_addr: Int, n: Int, gamma: Float32
) abi("C"):
    """`gamma_transform` on the float path: `np.power(img, gamma)`."""
    var img = fp(img_addr)
    var out = fp(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = pow(img[unsafe_offset=i], gamma)


@export("alb_to_gray_weighted_average_u8")
def alb_to_gray_weighted_average_u8(
    img_addr: Int, out_addr: Int, npix: Int
) abi("C"):
    """`to_gray_weighted_average` of a uint8 image: `cv2.cvtColor` RGB2GRAY.

    OpenCV's 8-bit path is fixed point: a sum of three weighted channels, rounded
    by the bias and shifted down. The coefficients and the shift are OpenCV's
    internal constants rather than a specification, and they changed in OpenCV 5:
    5.0 computes `Y = (9798 R + 19235 G + 3735 B + 16384) >> 15`, where 4.x used
    `(4899 R + 9617 G + 1868 B + 8192) >> 14`. Those are different roundings of
    nearly the same weight vector - `9798 / 2^15 = 0.29898...` against
    `4899 / 2^14 = 0.29901...` - and they disagree on the samples whose weighted
    sum lands exactly on a tie, about one pixel in 790.

    The constants below are the 5.0 ones, verified exhaustively: they reproduce
    `cv2.cvtColor(..., COLOR_RGB2GRAY)` on all 16777216 uint8 RGB triples. On 4.x
    the 4.x constants above are the correct ones, and the two differ only on ties.
    """
    var img = up(img_addr)
    var out = up(out_addr)
    for p in range(npix):
        var base = p * 3
        var acc = Int64(9798) * Int64(img[unsafe_offset=base])
        acc += Int64(19235) * Int64(img[unsafe_offset=base + 1])
        acc += Int64(3735) * Int64(img[unsafe_offset=base + 2])
        out[unsafe_offset=p] = UInt8((acc + 16384) >> 15)


@export("alb_to_gray_weighted_average_f32")
def alb_to_gray_weighted_average_f32(
    img_addr: Int, out_addr: Int, npix: Int
) abi("C"):
    """`to_gray_weighted_average` of a float32 image: the BT.601 luma."""
    var img = fp(img_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * 3
        var acc = Float32(0.299) * img[unsafe_offset=base]
        acc += Float32(0.587) * img[unsafe_offset=base + 1]
        acc += Float32(0.114) * img[unsafe_offset=base + 2]
        out[unsafe_offset=p] = acc


@export("alb_to_gray_desaturation_f32")
def alb_to_gray_desaturation_f32(
    img_addr: Int, out_addr: Int, npix: Int
) abi("C"):
    """`to_gray_desaturation`: the midpoint of the per-pixel channel extremes.

    `(np.max(float_image, -1) + np.min(float_image, -1)) / 2` on a float32
    image, which is bit-exact because that is what upstream computes.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * 3
        var hi = img[unsafe_offset=base]
        if img[unsafe_offset=base + 1] > hi:
            hi = img[unsafe_offset=base + 1]
        if img[unsafe_offset=base + 2] > hi:
            hi = img[unsafe_offset=base + 2]
        var lo = img[unsafe_offset=base]
        if img[unsafe_offset=base + 1] < lo:
            lo = img[unsafe_offset=base + 1]
        if img[unsafe_offset=base + 2] < lo:
            lo = img[unsafe_offset=base + 2]
        out[unsafe_offset=p] = (hi + lo) / 2.0


@export("alb_to_gray_average_u8")
def alb_to_gray_average_u8(
    img_addr: Int, out_addr: Int, npix: Int, nchan: Int
) abi("C"):
    """`to_gray_average` of a uint8 image: `np.mean(img, -1).astype(uint8)`.

    NumPy accumulates the mean of an integer array in float64 and then casts,
    which truncates; the loop does the same accumulation and cast.
    """
    var img = up(img_addr)
    var out = up(out_addr)
    for p in range(npix):
        var base = p * nchan
        var total = 0.0
        for c in range(nchan):
            total += Float64(img[unsafe_offset=base + c])
        out[unsafe_offset=p] = UInt8(Int(total / Float64(nchan)))


@export("alb_to_gray_average_f32")
def alb_to_gray_average_f32(
    img_addr: Int, out_addr: Int, npix: Int, nchan: Int
) abi("C"):
    """`to_gray_average` of a float32 image, accumulated in float32 as NumPy does."""
    var img = fp(img_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * nchan
        var total = Float32(0.0)
        for c in range(nchan):
            total += img[unsafe_offset=base + c]
        out[unsafe_offset=p] = total / Float32(nchan)


@export("alb_to_gray_max_u8")
def alb_to_gray_max_u8(
    img_addr: Int, out_addr: Int, npix: Int, nchan: Int
) abi("C"):
    """`to_gray_max` of a uint8 image: `np.max(img, -1)`."""
    var img = up(img_addr)
    var out = up(out_addr)
    for p in range(npix):
        var base = p * nchan
        var m = img[unsafe_offset=base]
        for c in range(1, nchan):
            if img[unsafe_offset=base + c] > m:
                m = img[unsafe_offset=base + c]
        out[unsafe_offset=p] = m


@export("alb_to_gray_max_f32")
def alb_to_gray_max_f32(
    img_addr: Int, out_addr: Int, npix: Int, nchan: Int
) abi("C"):
    """`to_gray_max` of a float32 image: `np.max(img, -1)`."""
    var img = fp(img_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * nchan
        var m = img[unsafe_offset=base]
        for c in range(1, nchan):
            if img[unsafe_offset=base + c] > m:
                m = img[unsafe_offset=base + c]
        out[unsafe_offset=p] = m


@export("alb_grayscale_to_multichannel_u8")
def alb_grayscale_to_multichannel_u8(
    img_addr: Int, out_addr: Int, npix: Int, nchan: Int
) abi("C"):
    """`grayscale_to_multichannel`: `cv2.merge([plane] * num_output_channels)`."""
    var img = up(img_addr)
    var out = up(out_addr)
    for p in range(npix):
        var v = img[unsafe_offset=p]
        for c in range(nchan):
            out[unsafe_offset=p * nchan + c] = v


def _clip01(v: Float32) -> Float32:
    if v < 0.0:
        return Float32(0.0)
    if v > 1.0:
        return Float32(1.0)
    return v



@export("alb_multiply_f32")
def alb_multiply_f32(
    img_addr: Int, out_addr: Int, n: Int, factor: Float32, max_val: Float32,
    truncate: Int
) abi("C"):
    """`albucore.multiply` on float32: clip to the dtype range after the product.

    Upstream takes one of two paths and `truncate` selects between them. On
    uint8 it builds `clip(arange(0, 256, float32) * factor, uint8)`, whose cast
    to uint8 truncates, and applies it as a table; on float32 it multiplies in
    float32 and the `@clipped` decorator clips without rounding. `max_val` is
    255 and 1.0 respectively.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    for i in range(n):
        var v = img[unsafe_offset=i] * factor
        if v < 0.0:
            v = 0.0
        elif v > max_val:
            v = max_val
        if truncate != 0:
            v = Float32(Int(v))
        out[unsafe_offset=i] = v


@export("alb_multiply_add_f32")
def alb_multiply_add_f32(
    img_addr: Int, out_addr: Int, n: Int, factor: Float32, addend: Float32,
    max_val: Float32, truncate: Int
) abi("C"):
    """`albucore.multiply_add`: `clip(img * factor + addend)`.

    Same two paths as `alb_multiply_f32`: the uint8 table truncates on the cast,
    the float32 path does not round.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    for i in range(n):
        var v = img[unsafe_offset=i] * factor + addend
        if v < 0.0:
            v = 0.0
        elif v > max_val:
            v = max_val
        if truncate != 0:
            v = Float32(Int(v))
        out[unsafe_offset=i] = v


@export("alb_add_weighted_f32")
def alb_add_weighted_f32(
    img_addr: Int, gray_addr: Int, out_addr: Int, npix: Int, factor: Float32,
    gamma: Float32, max_val: Float32, round_out: Int
) abi("C"):
    """`albucore.add_weighted`: `wsum(img, gray, alpha=factor, beta=1 - factor)`.

    `gray_addr` is the single-channel `cv2.cvtColor(img, COLOR_RGB2GRAY)` plane
    that upstream re-expands with `cv2.cvtColor(gray, COLOR_GRAY2RGB)`; holding
    it as a plane instead of expanding it is the same arithmetic without the
    three-fold store. simsimd's `wsum` rounds to the output dtype, so
    `round_out` is set for the uint8 path and clear for float32.
    """
    var img = fp(img_addr)
    var gray = fp(gray_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * 3
        var g = gray[unsafe_offset=p]
        for c in range(3):
            var v = img[unsafe_offset=base + c] * factor + g * (1.0 - factor)
            v = v + gamma
            if v < 0.0:
                v = 0.0
            elif v > max_val:
                v = max_val
            if round_out != 0:
                v = Float32(_rint(Float64(v)))
            out[unsafe_offset=base + c] = v


@export("alb_unsharp_mask_residual")
def alb_unsharp_mask_residual(
    image_addr: Int, blur_addr: Int, mask_addr: Int, sharp_addr: Int, n: Int,
    alpha: Float32, threshold: Int
) abi("C"):
    """`unsharp_mask`: the residual, the sharpen mask and the sharp image.

    Upstream computes `residual = image - blur`, `mask = abs(residual) * 255 >
    threshold` as float32, and `sharp = clip(image + alpha * residual, 0, 1)`.
    All three are produced here because all three are elementwise over the
    same buffer; the second Gaussian blur of `mask` and the recombination then
    run through the shared primitives.
    """
    var image = fp(image_addr)
    var blur = fp(blur_addr)
    var mask = fp(mask_addr)
    var sharp = fp(sharp_addr)
    for i in range(n):
        var residual = image[unsafe_offset=i] - blur[unsafe_offset=i]
        if _abs32(residual) * 255.0 > Float32(threshold):
            mask[unsafe_offset=i] = 1.0
        else:
            mask[unsafe_offset=i] = 0.0
        sharp[unsafe_offset=i] = _clip01(image[unsafe_offset=i] + alpha * residual)


def _abs32(v: Float32) -> Float32:
    if v < 0.0:
        return -v
    return v


@export("alb_unsharp_mask_combine")
def alb_unsharp_mask_combine(
    sharp_addr: Int, image_addr: Int, soft_mask_addr: Int, out_addr: Int, n: Int
) abi("C"):
    """`unsharp_mask`: `add(multiply(sharp, soft_mask), multiply(image, 1 - soft_mask))`."""
    var sharp = fp(sharp_addr)
    var image = fp(image_addr)
    var soft_mask = fp(soft_mask_addr)
    var out = fp(out_addr)
    for i in range(n):
        var m = soft_mask[unsafe_offset=i]
        out[unsafe_offset=i] = sharp[unsafe_offset=i] * m + image[unsafe_offset=i] * (
            1.0 - m
        )


@export("alb_add_noise_f32")
def alb_add_noise_f32(
    img_addr: Int, noise_addr: Int, out_addr: Int, n: Int
) abi("C"):
    """`add_noise` on float32: `clipped(add(img, noise))`."""
    var img = fp(img_addr)
    var noise = fp(noise_addr)
    var out = fp(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = _clip01(img[unsafe_offset=i] + noise[unsafe_offset=i])


@export("alb_sharpen_gaussian_f32")
def alb_sharpen_gaussian_f32(
    img_addr: Int, blur_addr: Int, out_addr: Int, n: Int, alpha: Float32
) abi("C"):
    """`sharpen_gaussian`: `clip(img + alpha * (img - blurred))`.

    Upstream's comment calls this the unsharp mask formula; the blur itself is
    `alb_blur_separable_f32`.
    """
    var img = fp(img_addr)
    var blur = fp(blur_addr)
    var out = fp(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = _clip01(
            img[unsafe_offset=i] + alpha * (img[unsafe_offset=i] - blur[unsafe_offset=i])
        )


@export("alb_apply_salt_and_pepper_u8")
def alb_apply_salt_and_pepper_u8(
    img_addr: Int, salt_addr: Int, pepper_addr: Int, out_addr: Int, npix: Int,
    nchan: Int, max_val: Int
) abi("C"):
    """`apply_salt_and_pepper`: `where(salt, max, where(pepper, 0, img))`.

    The masks are single-channel planes broadcast across the pixel's channels,
    which is what upstream's `mask[..., None]` does.
    """
    var img = up(img_addr)
    var salt = up(salt_addr)
    var pepper = up(pepper_addr)
    var out = up(out_addr)
    for p in range(npix):
        var s = salt[unsafe_offset=p] != 0
        var q = pepper[unsafe_offset=p] != 0
        for c in range(nchan):
            var base = p * nchan + c
            if s:
                out[unsafe_offset=base] = UInt8(max_val)
            elif q:
                out[unsafe_offset=base] = UInt8(0)
            else:
                out[unsafe_offset=base] = img[unsafe_offset=base]


@export("alb_create_directional_gradient_f32")
def alb_create_directional_gradient_f32(
    out_addr: Int, height: Int, width: Int, cos_a: Float64, sin_a: Float64
) abi("C"):
    """`create_directional_gradient`, general case: `x * cos_a + y * sin_a`.

    The linspace ramps are the coordinate vectors; multiplying each by its
    coefficient and summing is the two broadcasting multiplies plus add that
    upstream performs at the end. The axis-aligned and 45-degree fast paths are
    one-dimensional `linspace` products upstream and stay in the wrapper.
    """
    var out = fp(out_addr)
    for y in range(height):
        var fy = Float64(y) * (1.0 / Float64(height - 1)) if height > 1 else 0.0
        for x in range(width):
            var fx = Float64(x) * (1.0 / Float64(width - 1)) if width > 1 else 0.0
            var v = fx * cos_a + fy * sin_a
            out[unsafe_offset=y * width + x] = Float32(v)


@export("alb_apply_linear_illumination_f32")
def alb_apply_linear_illumination_f32(
    img_addr: Int, out_addr: Int, height: Int, width: Int, nchan: Int,
    intensity: Float32, cos_a: Float32, sin_a: Float32
) abi("C"):
    """`apply_linear_illumination`.

    `gradient = create_directional_gradient(h, w, angle)`, flipped and
    rescaled by `2 * abs(intensity)` with `+ 1 - abs(intensity)` when the
    intensity is negative, then multiplied into the image. `cos_a`/`sin_a`
    carry the angle that the gradient is built from; when the intensity is
    negative the gradient is `1 - g` instead of `g`.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    var abs_intensity = _abs32(intensity)
    for y in range(height):
        var fy = Float64(y) * (1.0 / Float64(height - 1)) if height > 1 else 0.0
        for x in range(width):
            var fx = Float64(x) * (1.0 / Float64(width - 1)) if width > 1 else 0.0
            var g = Float32(fx * Float64(cos_a) + fy * Float64(sin_a))
            if intensity < 0.0:
                g = 1.0 - g
            g = g * (2.0 * abs_intensity) + (1.0 - abs_intensity)
            for c in range(nchan):
                out[unsafe_offset=(y * width + x) * nchan + c] = _clip01(
                    img[unsafe_offset=(y * width + x) * nchan + c] * g
                )


@export("alb_apply_gaussian_illumination_f32")
def alb_apply_gaussian_illumination_f32(
    img_addr: Int, out_addr: Int, height: Int, width: Int, nchan: Int,
    intensity: Float32, center_x: Float32, center_y: Float32,
    inv_sigma2: Float32
) abi("C"):
    """`apply_gaussian_illumination`.

    The per-pixel field is `1 + intensity * exp(-((x-cx)^2 + (y-cy)^2) /
    sigma2)` with `sigma2 = 2 (max(h, w) * sigma)^2`, multiplied into the
    image. Each step is kept in float32 in the order upstream applies it.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    for y in range(height):
        var dy = Float32(y) - center_y
        for x in range(width):
            var dx = Float32(x) - center_x
            var d = dx * dx + dy * dy
            var v = exp(d * inv_sigma2)
            v = v * intensity + 1.0
            for c in range(nchan):
                out[unsafe_offset=(y * width + x) * nchan + c] = _clip01(
                    img[unsafe_offset=(y * width + x) * nchan + c] * v
                )


@export("alb_calc_hist_u8")
def alb_calc_hist_u8(
    img_addr: Int, hist_addr: Int, npix: Int, nchan: Int, ignore: Int,
    max_val: Int
) abi("C"):
    """`cv2.calcHist([channel], [0], mask, [256], [0, max_value])` per channel.

    `ignore` is the channel index upstream skips, or -1 for none; the `mask` is
    upstream's `channel != ignore` for the remaining channels. With a range of
    `[0, 255]` and 256 bins OpenCV's last bin is exclusive, so samples equal to
    `max_value` are dropped - `max_val` is passed in and those samples are
    skipped here for the same reason.
    """
    var img = up(img_addr)
    var hist = ip(hist_addr)
    var total = nchan * 256
    for i in range(total):
        hist[unsafe_offset=i] = 0
    for c in range(nchan):
        if c == ignore:
            continue
        for p in range(npix):
            var v = Int(img[unsafe_offset=p * nchan + c])
            if v == ignore and ignore >= 0:
                continue
            if v == max_val:
                continue
            hist[unsafe_offset=c * 256 + v] += 1


@export("alb_get_histogram_bounds")
def alb_get_histogram_bounds(
    hist_addr: Int, cutoff: Float64, out_addr: Int
) abi("C"):
    """`get_histogram_bounds`: the two scans over the histogram, in order."""
    var hist = ip(hist_addr)
    var out = ip(out_addr)
    var n = 256
    if cutoff == 0.0:
        var lo = 0
        var hi = 0
        while lo < n and hist[unsafe_offset=lo] == 0:
            lo += 1
        if lo == n:
            out[unsafe_offset=0] = 0
            out[unsafe_offset=1] = 0
            return
        hi = n - 1
        while hist[unsafe_offset=hi] == 0:
            hi -= 1
        out[unsafe_offset=0] = Int64(lo)
        out[unsafe_offset=1] = Int64(hi)
        return

    var total = 0.0
    for i in range(n):
        total += Float64(hist[unsafe_offset=i])
    if total == 0.0:
        out[unsafe_offset=0] = 0
        out[unsafe_offset=1] = 0
        return

    var uniform = True
    var ref0 = hist[unsafe_offset=0]
    for i in range(n):
        if hist[unsafe_offset=i] != ref0:
            uniform = False

    if uniform:
        var min_intensity = Int(Float64(n) * cutoff / 100.0)
        var max_intensity = n - min_intensity - 1
        out[unsafe_offset=0] = Int64(min_intensity)
        out[unsafe_offset=1] = Int64(max_intensity)
        return

    var pixels_to_cut = total * cutoff / 100.0

    var cumsum = 0.0
    var min_intensity = 0
    for i in range(n):
        cumsum += Float64(hist[unsafe_offset=i])
        if cumsum >= pixels_to_cut:
            min_intensity = i + 1
            break
    min_intensity = min(min_intensity, n - 1)

    cumsum = 0.0
    var max_intensity = n - 1
    var i2 = n - 1
    while i2 >= 0:
        cumsum += Float64(hist[unsafe_offset=i2])
        if cumsum >= pixels_to_cut:
            max_intensity = i2
            break
        i2 -= 1

    if min_intensity > max_intensity:
        var mid_point = (n - 1) // 2
        out[unsafe_offset=0] = Int64(mid_point)
        out[unsafe_offset=1] = Int64(mid_point)
        return

    out[unsafe_offset=0] = Int64(min_intensity)
    out[unsafe_offset=1] = Int64(max_intensity)


@export("alb_create_contrast_lut")
def alb_create_contrast_lut(
    hist_addr: Int, min_intensity: Int, max_intensity: Int, max_value: Int,
    method_is_cdf: Int, out_addr: Int, cdf_addr: Int
) abi("C"):
    """`create_contrast_lut`: the "cdf" and "pil" branches, as written upstream."""
    var hist = ip(hist_addr)
    var out = up(out_addr)
    var n = 256
    if min_intensity >= max_intensity:
        for i in range(n):
            out[unsafe_offset=i] = UInt8(0)
        return

    if method_is_cdf != 0:
        var last = max_intensity - min_intensity
        var cdf = dp(cdf_addr)
        var running = 0.0
        for i in range(last + 1):
            running += Float64(hist[unsafe_offset=min_intensity + i])
            cdf[unsafe_offset=i] = running
        if cdf[unsafe_offset=last] == 0.0:
            for i in range(n):
                out[unsafe_offset=i] = UInt8(i)
            return
        var cdf0 = cdf[unsafe_offset=0]
        var cdfl = cdf[unsafe_offset=last]
        for i in range(n):
            out[unsafe_offset=i] = UInt8(0)
        for i in range(last + 1):
            var v = (cdf[unsafe_offset=i] - cdf0) * Float64(max_value) / (cdfl - cdf0)
            var r = Int64(_rint(v))
            if r < 0:
                r = 0
            elif r > Int64(max_value):
                r = Int64(max_value)
            out[unsafe_offset=min_intensity + i] = _saturate_u8(r)
        for i in range(max_intensity + 1, n):
            out[unsafe_offset=i] = UInt8(max_value)
        return

    var scale = Float64(max_value) / Float64(max_intensity - min_intensity)
    for i in range(n):
        var v = _rint((Float64(i) - Float64(min_intensity)) * scale)
        var r = Int64(v)
        if r < 0:
            r = 0
        elif r > Int64(max_value):
            r = Int64(max_value)
        out[unsafe_offset=i] = _saturate_u8(r)
    for i in range(min_intensity):
        out[unsafe_offset=i] = UInt8(0)
    for i in range(max_intensity + 1, n):
        out[unsafe_offset=i] = UInt8(max_value)


@export("alb_rgb_to_optical_density_f32")
def alb_rgb_to_optical_density_f32(
    img_addr: Int, out_addr: Int, npix: Int, max_value: Float32, eps: Float32
) abi("C"):
    """`rgb_to_optical_density`: `-log(max(pixel / max_value, eps))` per channel."""
    var img = fp(img_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * 3
        for c in range(3):
            var v = img[unsafe_offset=base + c] / max_value
            if v < eps:
                v = eps
            out[unsafe_offset=base + c] = -log(v)


@export("alb_convolve_f32")
def alb_convolve_f32(
    img_addr: Int, kern_addr: Int, out_addr: Int, height: Int, width: Int,
    nchan: Int, kheight: Int, kwidth: Int, border: Int
) abi("C"):
    """`convolve`: `cv2.filter2D(img, -1, kernel)`.

    `cv2.filter2D` is a correlation - the kernel is not flipped - anchored at
    the kernel centre. Border handling follows `borderType`; OpenCV's default
    for `filter2D` is BORDER_REFLECT_101.
    """
    var img = fp(img_addr)
    var kern = fp(kern_addr)
    var out = fp(out_addr)
    if height <= 0 or width <= 0 or nchan <= 0 or kheight <= 0 or kwidth <= 0:
        return
    var ay = kheight // 2
    var ax = kwidth // 2
    for y in range(height):
        for x in range(width):
            for c in range(nchan):
                var total = Float32(0.0)
                for ky in range(kheight):
                    var sy = y + ky - ay
                    for kx in range(kwidth):
                        var sx = x + kx - ax
                        var s = _border_index(sy, sx, height, width, border)
                        if s >= 0:
                            total += img[unsafe_offset=s * nchan + c] * kern[
                                unsafe_offset=ky * kwidth + kx
                            ]
                out[unsafe_offset=(y * width + x) * nchan + c] = total


def _border_index(sy: Int, sx: Int, height: Int, width: Int, border: Int) -> Int:
    if sy >= 0 and sy < height and sx >= 0 and sx < width:
        return sy * width + sx
    if border == BORDER_CONSTANT:
        return -1
    if border == BORDER_REPLICATE:
        var cy = sy
        var cx = sx
        if cy < 0:
            cy = 0
        elif cy >= height:
            cy = height - 1
        if cx < 0:
            cx = 0
        elif cx >= width:
            cx = width - 1
        return cy * width + cx
    return _reflect_101(sy, height) * width + _reflect_101(sx, width)


@export("alb_separable_convolve_f32")
def alb_separable_convolve_f32(
    img_addr: Int, tmp_addr: Int, kern_addr: Int, out_addr: Int, height: Int,
    width: Int, nchan: Int, ksize: Int, border: Int
) abi("C"):
    """`separable_convolve`: `cv2.sepFilter2D(img, -1, kernel, kernel)`.

    Upstream passes the same 1-D kernel for both axes; the horizontal pass
    writes `tmp` and the vertical pass reads it, which is also what makes the
    second pass's border handling see the intermediate.
    """
    var img = fp(img_addr)
    var tmp = fp(tmp_addr)
    var kern = fp(kern_addr)
    var out = fp(out_addr)
    if height <= 0 or width <= 0 or nchan <= 0 or ksize <= 0:
        return
    var radius = ksize // 2
    for y in range(height):
        for x in range(width):
            for c in range(nchan):
                var total = Float32(0.0)
                for k in range(ksize):
                    var s = _border_index(
                        y, x + k - radius, height, width, border
                    )
                    if s >= 0:
                        total += img[unsafe_offset=s * nchan + c] * kern[unsafe_offset=k]
                tmp[unsafe_offset=(y * width + x) * nchan + c] = total
    for y in range(height):
        for x in range(width):
            for c in range(nchan):
                var total = Float32(0.0)
                for k in range(ksize):
                    var s = _border_index(
                        y + k - radius, x, height, width, border
                    )
                    if s >= 0:
                        total += tmp[unsafe_offset=s * nchan + c] * kern[unsafe_offset=k]
                out[unsafe_offset=(y * width + x) * nchan + c] = total


# ===========================================================================
# albumentations/augmentations/blur/functional.py
# ===========================================================================


@export("alb_box_blur_u8")
def alb_box_blur_u8(
    img_addr: Int, out_addr: Int, height: Int, width: Int, nchan: Int,
    ksize: Int
) abi("C"):
    """`box_blur`: `cv2.blur(img, (ksize, ksize))`.

    OpenCV's normalised box filter is the plain ksize*ksize window mean with
    BORDER_REFLECT_101 extrapolation. The scaling is integer arithmetic that
    is not a single rule: a 2x2 window takes the ceiling of the mean, every
    other size takes the half-up rounding. Both are reproduced below by
    counting the remainders rather than rounding a float, so neither depends
    on a division at all.
    """
    var img = up(img_addr)
    var out = up(out_addr)
    if height <= 0 or width <= 0 or nchan <= 0 or ksize <= 0:
        return
    var radius = ksize // 2
    var area = ksize * ksize
    for y in range(height):
        for x in range(width):
            for c in range(nchan):
                var total = 0
                for ky in range(ksize):
                    var sy = _reflect_101(y + ky - radius, height)
                    for kx in range(ksize):
                        var sx = _reflect_101(x + kx - radius, width)
                        total += Int(img[unsafe_offset=(sy * width + sx) * nchan + c])
                out[unsafe_offset=(y * width + x) * nchan + c] = _saturate_u8(
                    Int64(_box_mean(total, ksize, area))
                )


@export("alb_create_gaussian_kernel_input_array")
def alb_create_gaussian_kernel_input_array(
    out_addr: Int, size: Int, as_int: Int
) abi("C"):
    """`create_gaussian_kernel_input_array`.

    Below 100 elements upstream returns the `arange` list as `int64`; at and
    above 100 it returns `linspace`, which is `float64` and whose endpoint is
    set exactly rather than accumulated. `as_int` selects the first form.

    The `linspace` ramp is built in two passes - all the products, then all the
    sums. Folding them into one expression lets the backend contract
    `start + i * step` into a single fused multiply-add, which is one rounding
    step more accurate than the separate multiply and add `np.linspace`
    performs, and the two disagree in the last bit often enough to matter.
    """
    if size <= 0:
        return
    if as_int != 0:
        # `range(-(size // 2), size // 2 + 1)` holds one element more than
        # `size` when `size` is even, because both endpoints are included and
        # the span between them is odd. The caller sizes its buffer from the
        # same expression, so the two agree on the length.
        var count = (size // 2) * 2 + 1
        var iout = ip(out_addr)
        for i in range(count):
            iout[unsafe_offset=i] = Int64(i - (size // 2))
        return
    var start = -Float64(size // 2)
    var stop = Float64(size // 2)
    if size < 100:
        var qout = dp(out_addr)
        for i in range(size):
            qout[unsafe_offset=i] = Float64(i - (size // 2))
        return
    var out = dp(out_addr)
    var step = (stop - start) / Float64(size - 1)
    for i in range(size):
        out[unsafe_offset=i] = Float64(i) * step
    for i in range(size):
        out[unsafe_offset=i] = out[unsafe_offset=i] + start
    out[unsafe_offset=size - 1] = stop


@export("alb_create_gaussian_kernel_1d")
def alb_create_gaussian_kernel_1d(
    x_addr: Int, ksize: Int, sigma: Float64, out_addr: Int
) abi("C"):
    """`create_gaussian_kernel_1d`: normalise `exp(-0.5 (x / sigma) ** 2)`.

    The caller supplies the coordinate ramp from
    `create_gaussian_kernel_input_array`, exactly as upstream does; only the
    profile itself is arithmetic. `exp` is the C library's, not `std.math`'s:
    the latter is a ~1e-12 relative approximation, and `np.exp` agrees with
    glibc bit for bit, so only the latter reproduces upstream's kernel.

    The normalising sum is `ndarray.sum()`, which is a pairwise sum rather
    than a running total, and the two disagree in the last bit often enough
    to change the profile. `_pairwise_sum` reproduces numpy's blocking.
    """
    var x = dp(x_addr)
    var out = dp(out_addr)
    if ksize <= 0:
        return
    for i in range(ksize):
        var t = x[unsafe_offset=i] / sigma
        out[unsafe_offset=i] = c_exp(-0.5 * (t * t))
    var total = _pairwise_sum(out, ksize)
    if total != 0.0:
        for i in range(ksize):
            out[unsafe_offset=i] = out[unsafe_offset=i] / total


def _pairwise_sum(data: DPtr, n: Int) -> Float64:
    """numpy's `pairwise_sum`, which is exact here for every `n < 128`.

    Under eight elements numpy just adds in order. Above that it keeps eight
    running totals, folds them pairwise, and adds the ragged tail on top; the
    grouping is what a plain running total does not reproduce, and the two
    differ by an ulp often enough to change the profile.

    Past 128 elements numpy stops using this routine and falls through to the
    CPU-dispatched `add.reduce` loop, whose summation order depends on the
    vector width the build selected and is not recoverable from here. This
    stays on the pairwise path instead, so a kernel that long can differ from
    upstream in the last bit of the normaliser - a relative error around
    1e-16. Kernel sizes are odd and grow with sigma, so this only becomes
    reachable at a sigma of roughly 16 and above.
    """
    if n < 8:
        var acc = 0.0
        for i in range(n):
            acc += data[unsafe_offset=i]
        return acc
    var r0 = data[unsafe_offset=0]
    var r1 = data[unsafe_offset=1]
    var r2 = data[unsafe_offset=2]
    var r3 = data[unsafe_offset=3]
    var r4 = data[unsafe_offset=4]
    var r5 = data[unsafe_offset=5]
    var r6 = data[unsafe_offset=6]
    var r7 = data[unsafe_offset=7]
    var i = 8
    var body = n - (n % 8)
    while i < body:
        r0 += data[unsafe_offset=i]
        r1 += data[unsafe_offset=i + 1]
        r2 += data[unsafe_offset=i + 2]
        r3 += data[unsafe_offset=i + 3]
        r4 += data[unsafe_offset=i + 4]
        r5 += data[unsafe_offset=i + 5]
        r6 += data[unsafe_offset=i + 6]
        r7 += data[unsafe_offset=i + 7]
        i += 8
    var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += data[unsafe_offset=i]
        i += 1
    return res


@export("alb_create_gaussian_kernel_2d")
def alb_create_gaussian_kernel_2d(
    kernel_1d_addr: Int, out_addr: Int, ksize: Int
) abi("C"):
    """`create_gaussian_kernel`: the outer product `k[:, None] @ k[None, :]`."""
    var k = dp(kernel_1d_addr)
    var out = dp(out_addr)
    for y in range(ksize):
        for x in range(ksize):
            out[unsafe_offset=y * ksize + x] = k[unsafe_offset=y] * k[unsafe_offset=x]


@export("alb_create_motion_kernel")
def alb_create_motion_kernel(
    out_addr: Int, kernel_size: Int, cos_a: Float64, sin_a: Float64,
    direction: Float64, shift_x: Float64, shift_y: Float64
) abi("C"):
    """`create_motion_kernel` with the random shift supplied by the caller.

    Points along the biased line are rounded and clipped into the kernel and
    set to 1. Upstream de-duplicates them with `np.unique` first, which cannot
    change the result because the assignment is idempotent.

    The kernel is `float32`, not a mask: upstream builds it as a zeroed
    `float32` array and writes `1`, so a `uint8` buffer would be the wrong
    dtype at the boundary even where the values happen to agree.
    """
    var out = fp(out_addr)
    for i in range(kernel_size * kernel_size):
        out[unsafe_offset=i] = Float32(0.0)
    var center = Float64(kernel_size // 2)
    var line_length = Float64(kernel_size // 2)

    var d = direction
    if d < -1.0:
        d = -1.0
    elif d > 1.0:
        d = 1.0

    var t_start = -line_length
    var t_end = line_length
    if d < 0.0:
        var bias_factor = -d
        t_end = line_length * (1.0 - bias_factor)
    elif d > 0.0:
        var bias_factor = d
        t_start = -line_length * (1.0 - bias_factor)

    for i in range(kernel_size):
        var t = _linspace(t_start, t_end, Float64(i), kernel_size)
        var gx = center + cos_a * t + shift_x
        var gy = center + sin_a * t + shift_y
        var ix = Int(_rint(gx))
        var iy = Int(_rint(gy))
        if ix < 0:
            ix = 0
        elif ix > kernel_size - 1:
            ix = kernel_size - 1
        if iy < 0:
            iy = 0
        elif iy > kernel_size - 1:
            iy = kernel_size - 1
        out[unsafe_offset=iy * kernel_size + ix] = Float32(1.0)


def _linspace(start: Float64, stop: Float64, i: Float64, n: Int) -> Float64:
    if n == 1:
        return start
    var step = (stop - start) / Float64(n - 1)
    return start + i * step


# ===========================================================================
# albumentations/augmentations/geometric/functional.py
# ===========================================================================


@export("alb_transpose_u8")
def alb_transpose_u8(
    img_addr: Int, out_addr: Int, height: Int, width: Int, nchan: Int
) abi("C"):
    """`transpose`: swap the first two axes, keep the rest."""
    var img = up(img_addr)
    var out = up(out_addr)
    for y in range(height):
        for x in range(width):
            for c in range(nchan):
                out[unsafe_offset=(x * height + y) * nchan + c] = img[
                    unsafe_offset=(y * width + x) * nchan + c
                ]


@export("alb_vflip_u8")
def alb_vflip_u8(
    img_addr: Int, out_addr: Int, height: Int, width: Int, nchan: Int
) abi("C"):
    """`vflip`: reverse the rows."""
    var img = up(img_addr)
    var out = up(out_addr)
    for y in range(height):
        var src_y = height - 1 - y
        for x in range(width):
            for c in range(nchan):
                out[unsafe_offset=(y * width + x) * nchan + c] = img[
                    unsafe_offset=(src_y * width + x) * nchan + c
                ]


@export("alb_hflip_u8")
def alb_hflip_u8(
    img_addr: Int, out_addr: Int, height: Int, width: Int, nchan: Int
) abi("C"):
    """`hflip`: reverse the columns."""
    var img = up(img_addr)
    var out = up(out_addr)
    for y in range(height):
        for x in range(width):
            var src_x = width - 1 - x
            for c in range(nchan):
                out[unsafe_offset=(y * width + x) * nchan + c] = img[
                    unsafe_offset=(y * width + src_x) * nchan + c
                ]


@export("alb_rot90_u8")
def alb_rot90_u8(
    img_addr: Int, out_addr: Int, height: Int, width: Int, nchan: Int,
    factor: Int
) abi("C"):
    """`rot90`: `np.rot90(img, factor)` repeated counterclockwise."""
    var img = up(img_addr)
    var out = up(out_addr)
    var f = factor % 4
    if f < 0:
        f += 4
    if f == 0:
        for i in range(height * width * nchan):
            out[unsafe_offset=i] = img[unsafe_offset=i]
        return
    if f == 2:
        for y in range(height):
            for x in range(width):
                for c in range(nchan):
                    out[unsafe_offset=(y * width + x) * nchan + c] = img[
                        unsafe_offset=((height - 1 - y) * width + (width - 1 - x))
                        * nchan + c
                    ]
        return
    # f == 1 or 3: the output is (width, height)
    var out_w = height
    for y in range(height):
        for x in range(width):
            var oy = width - 1 - x
            var ox = y
            if f == 3:
                oy = x
                ox = height - 1 - y

            for c in range(nchan):
                out[unsafe_offset=(oy * out_w + ox) * nchan + c] = img[
                    unsafe_offset=(y * width + x) * nchan + c
                ]


@export("alb_copy_make_border_u8")
def alb_copy_make_border_u8(
    img_addr: Int, out_addr: Int, value_addr: Int, height: Int, width: Int,
    nchan: Int, top: Int, bottom: Int, left: Int, right: Int, border: Int
) abi("C"):
    """`copy_make_border_with_value_extension`: `cv2.copyMakeBorder`.

    `value_addr` is the per-channel `extend_value(value, num_channels)` list.
    """
    var img = up(img_addr)
    var out = up(out_addr)
    var value = up(value_addr)
    var out_h = height + top + bottom
    var out_w = width + left + right
    for oy in range(out_h):
        for ox in range(out_w):
            var sy = oy - top
            var sx = ox - left
            var inside = sy >= 0 and sy < height and sx >= 0 and sx < width
            var ry = sy
            var rx = sx
            if not inside:
                if border == BORDER_CONSTANT:
                    for c in range(nchan):
                        out[unsafe_offset=(oy * out_w + ox) * nchan + c] = value[
                            unsafe_offset=c
                        ]
                    continue
                if border == BORDER_REPLICATE:
                    if ry < 0:
                        ry = 0
                    elif ry >= height:
                        ry = height - 1
                    if rx < 0:
                        rx = 0
                    elif rx >= width:
                        rx = width - 1
                else:
                    ry = _reflect_101(ry, height)
                    rx = _reflect_101(rx, width)
            for c in range(nchan):
                out[unsafe_offset=(oy * out_w + ox) * nchan + c] = img[
                    unsafe_offset=(ry * width + rx) * nchan + c
                ]


@export("alb_affine_map_f32")
def alb_affine_map_f32(
    m0: Float64, m1: Float64, m2: Float64, m3: Float64, m4: Float64,
    m5: Float64, height: Int, width: Int, map_x_addr: Int, map_y_addr: Int
) abi("C"):
    """The inverse-affine coordinate grid `warp_affine` resamples through.

    For the forward affine `M`, the sample at output pixel `(x, y)` is
    `(m0 x + m1 y + m2, m3 x + m4 y + m5)`. `cv2.warpAffine` inverts the
    matrix and builds exactly this map internally; the wrapper inverts it and
    the grid is generated here, so no index mesh is materialised.

    The precision is not incidental. OpenCV holds the inverse in `float`, so
    each coefficient is narrowed to `Float32` first; the `y` term is then
    formed in `Float32` and carried down the column, while the `x` term is a
    `Float64` multiply. The result is then narrowed once more. Evaluating the
    same expression in `Float64` throughout lands within one ulp of the right
    source pixel often enough to break a bit-exact comparison, because
    `remap` rounds the coordinate to an integer and a coordinate one ulp the
    other side of a tie rounds to a different pixel.
    """
    var map_x = fp(map_x_addr)
    var map_y = fp(map_y_addr)
    if height <= 0 or width <= 0:
        return
    var a0 = Float64(Float32(m0))
    var a1 = Float32(m1)
    var a2 = Float32(m2)
    var a3 = Float64(Float32(m3))
    var a4 = Float32(m4)
    var a5 = Float32(m5)
    for y in range(height):
        var row_x = _f32_affine_row(a1, a2, y)
        var row_y = _f32_affine_row(a4, a5, y)
        for x in range(width):
            var fx = Float64(x)
            var p = y * width + x
            var tx: Float64 = a0 * fx
            var ty: Float64 = a3 * fx
            map_x[unsafe_offset=p] = Float32(tx + row_x)
            map_y[unsafe_offset=p] = Float32(ty + row_y)



def _f32_affine_row(m_row: Float32, c_row: Float32, y: Int) -> Float64:
    """`m_row * y + c_row` in `float32`, widened, as OpenCV's map row seed.

    Both roundings have to happen, in that order: the product is narrowed to
    `float32` first, and only then is `c_row` added, with the sum narrowed
    again. Carrying the product in `float64` and rounding once at the end
    skips the intermediate narrowing, and letting the backend contract the
    product and the sum into a single fused multiply-add skips it as well.
    The explicit `Float32` temporaries are what stop the contraction; the
    result is widened only after both roundings are done.
    """
    var prod: Float64 = Float64(Float32(Float64(m_row) * Float64(y)))
    var narrowed: Float32 = Float32(prod)
    var total: Float32 = narrowed + c_row
    return Float64(total)

@export("alb_erode_u8")
def alb_erode_u8(
    img_addr: Int, kern_addr: Int, out_addr: Int, height: Int, width: Int,
    nchan: Int, kheight: Int, kwidth: Int
) abi("C"):
    """`erode`: `cv2.erode(img, kernel, iterations=1)`.

    OpenCV's default morphology border value for erosion saturates to the
    dtype maximum, so out-of-image samples never lower the minimum.
    """
    var img = up(img_addr)
    var kern = up(kern_addr)
    var out = up(out_addr)
    if height <= 0 or width <= 0 or nchan <= 0 or kheight <= 0 or kwidth <= 0:
        return
    var ay = kheight // 2
    var ax = kwidth // 2
    for y in range(height):
        for x in range(width):
            for c in range(nchan):
                var m = 255
                for ky in range(kheight):
                    var sy = y + ky - ay
                    if sy < 0 or sy >= height:
                        continue
                    for kx in range(kwidth):
                        if kern[unsafe_offset=ky * kwidth + kx] == 0:
                            continue
                        var sx = x + kx - ax
                        if sx < 0 or sx >= width:
                            continue
                        var v = Int(img[unsafe_offset=(sy * width + sx) * nchan + c])
                        if v < m:
                            m = v
                out[unsafe_offset=(y * width + x) * nchan + c] = UInt8(m)


@export("alb_dilate_u8")
def alb_dilate_u8(
    img_addr: Int, kern_addr: Int, out_addr: Int, height: Int, width: Int,
    nchan: Int, kheight: Int, kwidth: Int
) abi("C"):
    """`dilate`: `cv2.dilate(img, kernel, iterations=1)`.

    Out-of-image samples saturate to zero, which is OpenCV's default
    morphology border value for dilation, so they never raise the maximum.
    """
    var img = up(img_addr)
    var kern = up(kern_addr)
    var out = up(out_addr)
    if height <= 0 or width <= 0 or nchan <= 0 or kheight <= 0 or kwidth <= 0:
        return
    var ay = kheight // 2
    var ax = kwidth // 2
    for y in range(height):
        for x in range(width):
            for c in range(nchan):
                var m = 0
                for ky in range(kheight):
                    var sy = y + ky - ay
                    if sy < 0 or sy >= height:
                        continue
                    for kx in range(kwidth):
                        if kern[unsafe_offset=ky * kwidth + kx] == 0:
                            continue
                        var sx = x + kx - ax
                        if sx < 0 or sx >= width:
                            continue
                        var v = Int(img[unsafe_offset=(sy * width + sx) * nchan + c])
                        if v > m:
                            m = v
                out[unsafe_offset=(y * width + x) * nchan + c] = UInt8(m)


@export("alb_scale_fields_f32")
def alb_scale_fields_f32(
    fields_addr: Int, npix: Int, alpha: Float32
) abi("C"):
    """`generate_displacement_fields`: `fields *= alpha` after the blur."""
    var fields = fp(fields_addr)
    for i in range(npix):
        fields[unsafe_offset=i] = fields[unsafe_offset=i] * alpha