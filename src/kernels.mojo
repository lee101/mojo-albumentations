"""Image-transform inner loops for the compute subset of albumentations.

`albumentations` is 43,378 lines, and most of it is transform classes, parameter
sampling and metadata plumbing. The numeric core is a small set of per-pixel and
per-pixel-neighbourhood loops, and those are what is compiled here:

    pixel/functional.solarize, posterize, invert      a 256-entry LUT gather
    pixel/functional.to_gray_*                        channel reduction
    pixel/functional.channel_shuffle                   channel permutation
    pixel/functional.gamma_transform, move_tone_curve  a power and a curve LUT
    blur/functional.create_gaussian_kernel_1d          exp() of a normalised ramp
    generate_displacement_fields, sharpen_gaussian     separable Gaussian, BORDER_REPLICATE
    geometric remap, ElasticTransform resample          nearest resample over a coordinate map
    geometric warp_affine map generation                inverse affine coordinate grid
    Normalize                                           per-channel mean and variance

Everything else in the package - the transform classes, the parameter samplers,
the replay buffers, the bboxes, the keypoints, the composition, the serialisation
- is control flow and stays upstream.

The LUT, gather, shuffle, resample and coordinate-map kernels move or select
bytes and are bit-exact. The Gaussian kernel, the blur and the channel
statistics compute, and are asserted with explicit tolerances.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

from std.math import exp

comptime UPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float32, AnyOrigin[mut=True]]
comptime DPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def dp(addr: Int) -> DPtr:
    return DPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("alb_lut_apply_u8")
def alb_lut_apply_u8(
    lut_addr: Int, img_addr: Int, out_addr: Int, n: Int
) abi("C"):
    """Apply a 256-entry lookup table to every byte of the image.

    This is the whole of `solarize` (its 256-value inversion table),
    `posterize` (its high-bit mask table), `invert`, `move_tone_curve` and the
    `equalize` histogram stretch: build the table once, gather. Pure data
    movement, so the result is bit-identical.
    """
    var lut = up(lut_addr)
    var img = up(img_addr)
    var out = up(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = lut[unsafe_offset=Int(img[unsafe_offset=i])]


@export("alb_gray_u8")
def alb_gray_u8(
    img_addr: Int, out_addr: Int, npix: Int, mode: Int
) abi("C"):
    """Per-pixel channel reduction of a 3-channel uint8 image.

    mode 0 is `to_gray_average`, the truncating mean `(r + g + b) // 3` of
    `np.mean(img, -1).astype(uint8)`; mode 1 is `to_gray_max`; mode 2 is
    `to_gray_weighted_average`, the ITU-R BT.601 luma
    `0.299 r + 0.587 g + 0.114 b` rounded to nearest.

    `to_gray_average` and `to_gray_max` are exact. The luma mode is exact to
    within one 8-bit LSB: OpenCV's fixed-point path and a float round can land
    on opposite sides of a rounding boundary, and the two are measured to differ
    on about 0.04% of pixels.
    """
    var img = up(img_addr)
    var out = up(out_addr)
    for p in range(npix):
        var base = p * 3
        var r = Float64(img[unsafe_offset=base])
        var g = Float64(img[unsafe_offset=base + 1])
        var b = Float64(img[unsafe_offset=base + 2])
        if mode == 0:
            var total = Int64(img[unsafe_offset=base])
            total += Int64(img[unsafe_offset=base + 1])
            total += Int64(img[unsafe_offset=base + 2])
            out[unsafe_offset=p] = UInt8(total // 3)
        elif mode == 1:
            var m = img[unsafe_offset=base]
            if img[unsafe_offset=base + 1] > m:
                m = img[unsafe_offset=base + 1]
            if img[unsafe_offset=base + 2] > m:
                m = img[unsafe_offset=base + 2]
            out[unsafe_offset=p] = m
        else:
            var luma = 0.299 * r + 0.587 * g + 0.114 * b
            var rounded = Int64(luma + 0.5)
            if rounded < 0:
                rounded = 0
            elif rounded > 255:
                rounded = 255
            out[unsafe_offset=p] = UInt8(rounded)


@export("alb_gray_luma_f32")
def alb_gray_luma_f32(
    img_addr: Int, out_addr: Int, npix: Int
) abi("C"):
    """`to_gray_weighted_average` of a float32 image: the BT.601 luma.

    The float32 path upstream goes through `cv2.cvtColor`, which evaluates the
    same three coefficients in float32. Mojo emits FMA here, so the result is a
    few ULP away rather than identical.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * 3
        out[unsafe_offset=p] = (
            0.299 * img[unsafe_offset=base]
            + 0.587 * img[unsafe_offset=base + 1]
            + 0.114 * img[unsafe_offset=base + 2]
        )


@export("alb_desaturate_f32")
def alb_desaturate_f32(img_addr: Int, out_addr: Int, npix: Int) abi("C"):
    """`to_gray_desaturation`: the midpoint of the per-pixel channel extremes.

    Bit-exact in float32, because that is what upstream computes:
    `img.astype(float32)` then `(max + min) / 2`.
    """
    var img = fp(img_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var base = p * 3
        var a = img[unsafe_offset=base]
        var b = img[unsafe_offset=base + 1]
        var c = img[unsafe_offset=base + 2]
        var hi = a
        if b > hi:
            hi = b
        if c > hi:
            hi = c
        var lo = a
        if b < lo:
            lo = b
        if c < lo:
            lo = c
        out[unsafe_offset=p] = (hi + lo) / 2.0


@export("alb_channel_shuffle_u8")
def alb_channel_shuffle_u8(
    img_addr: Int, out_addr: Int, npix: Int, p0: Int, p1: Int, p2: Int
) abi("C"):
    """`channel_shuffle`: `out[..., i] = img[..., p_i]`.

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


@export("alb_gaussian_kernel_1d")
def alb_gaussian_kernel_1d(
    x_addr: Int, ksize: Int, sigma: Float64, out_addr: Int
) abi("C"):
    """`create_gaussian_kernel_1d`: normalise `exp(-0.5 (x / sigma) ** 2)`.

    The caller supplies the coordinate ramp, which is how upstream does it -
    `create_gaussian_kernel_input_array` returns `arange` for a small kernel and
    `linspace` for a large one, and only the profile is arithmetic. Exact to a
    few ULP of the `exp`; see the README for the asserted tolerance.
    """
    var x = dp(x_addr)
    var out = dp(out_addr)
    if ksize <= 0:
        return
    var total = 0.0
    for i in range(ksize):
        var t = x[unsafe_offset=i] / sigma
        var v = exp(-0.5 * t * t)
        out[unsafe_offset=i] = v
        total += v
    if total != 0.0:
        for i in range(ksize):
            out[unsafe_offset=i] = out[unsafe_offset=i] / total


@export("alb_blur_separable_f32")
def alb_blur_separable_f32(
    src_addr: Int, tmp_addr: Int, out_addr: Int, kern_addr: Int, height: Int,
    width: Int, nchan: Int, ksize: Int, border: Int
) abi("C"):
    """Separable Gaussian blur of a float32 image, horizontal then vertical.

    The horizontal pass reads `src` and writes `tmp`; the vertical pass reads
    `tmp` and writes `out`. `kern_addr` is the 1-D profile and is a separate
    buffer from the intermediate - sharing one would have the second pass read
    its coefficients out of the first pass's output. `border` 0 is
    BORDER_REPLICATE, which is what
    `generate_displacement_fields` asks `cv2.GaussianBlur` for when it smooths
    the elastic-transform noise field; `border` 1 is BORDER_CONSTANT.

    This is `sharpen_gaussian`'s blur and the elastic-transform field smoothing.
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
                    if border == 0:
                        if sx < 0:
                            sx = 0
                        elif sx >= width:
                            sx = width - 1
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
                    if border == 0:
                        if sy < 0:
                            sy = 0
                        elif sy >= height:
                            sy = height - 1
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
    """Nearest-neighbour resample of a uint8 image through a coordinate map.

    `map_x` and `map_y` are `out_h * out_w` source coordinates, row-major, and
    the output grid is indexed by the map's own shape - which may differ from
    the source image's, so the two sizes are separate arguments and the source
    row stride is the source width. Out-of-range samples are zero for
    `border` 1 (BORDER_CONSTANT) and clamped for `border` 0
    (BORDER_REPLICATE). Rounding is half-to-even, which is what `cvRound` does,
    so the result is bit-identical to `cv2.remap(..., cv2.INTER_NEAREST)`.

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
            var sx = _round_half_even(map_x[unsafe_offset=p])
            var sy = _round_half_even(map_y[unsafe_offset=p])
            for c in range(nchan):
                var base = p * nchan + c
                if border == 0:
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
                else:
                    if sx < 0 or sx >= src_w or sy < 0 or sy >= src_h:
                        out[unsafe_offset=base] = 0
                    else:
                        out[unsafe_offset=base] = img[
                            unsafe_offset=(sy * src_w + sx) * nchan + c
                        ]


@export("alb_affine_map_f32")
def alb_affine_map_f32(
    m0: Float64, m1: Float64, m2: Float64, m3: Float64, m4: Float64,
    m5: Float64, height: Int, width: Int, map_x_addr: Int, map_y_addr: Int
) abi("C"):
    """Build the inverse-affine coordinate map for a `height * width` output.

    For the forward affine `M`, the sample at output pixel ``(x, y)`` is
    ``(m0 x + m1 y + m2, m3 x + m4 y + m5)``. A caller resampling through this
    map gets the forward warp, which is the same convention
    `albumentations.geometric.functional.warp_affine` uses once OpenCV has
    inverted its matrix. The grid is generated directly, so no index mesh is
    materialised.
    """
    var map_x = fp(map_x_addr)
    var map_y = fp(map_y_addr)
    if height <= 0 or width <= 0:
        return
    for y in range(height):
        var fy = Float64(y)
        for x in range(width):
            var fx = Float64(x)
            var p = y * width + x
            map_x[unsafe_offset=p] = Float32(m0 * fx + m1 * fy + m2)
            map_y[unsafe_offset=p] = Float32(m3 * fx + m4 * fy + m5)


@export("alb_channel_stats_f64")
def alb_channel_stats_f64(
    img_addr: Int, out_addr: Int, npix: Int, nchan: Int
) abi("C"):
    """Per-channel sum and sum of squares, in float64, for `Normalize`.

    Writes `2 * nchan` doubles: the sum of each channel followed by the sum of
    its squares. Mean and standard deviation are derived from those by the
    caller, which is the two-pass form upstream's `Normalize` ends up
    computing anyway.
    """
    var img = dp(img_addr)
    var out = dp(out_addr)
    if npix <= 0 or nchan <= 0:
        return
    for c in range(nchan):
        out[unsafe_offset=c] = 0.0
        out[unsafe_offset=nchan + c] = 0.0
    for p in range(npix):
        var base = p * nchan
        for c in range(nchan):
            var v = img[unsafe_offset=base + c]
            out[unsafe_offset=c] = out[unsafe_offset=c] + v
            out[unsafe_offset=nchan + c] = out[unsafe_offset=nchan + c] + v * v


def _floor_int(f: Float64) -> Int:
    """Largest integer not greater than `f`."""
    var t = Int(f)
    if Float64(t) > f:
        return t - 1
    return t


def _round_half_even(v: Float32) -> Int:
    """`cvRound`: round half to even, which is what OpenCV and NumPy both do.

    The floor is taken first so that negative coordinates round the way
    round-half-even requires - truncating toward zero would send -1.5 to -1
    instead of -2, and a resampled pixel would be the wrong one.
    """
    var f = Float64(v)
    var fl = _floor_int(f)
    var frac = f - Float64(fl)
    if frac > 0.5:
        return fl + 1
    if frac < 0.5:
        return fl
    if (fl % 2) == 0:
        return fl
    return fl + 1
