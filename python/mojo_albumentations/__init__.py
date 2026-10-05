"""Mojo port of the compute-bound inner loops of `albumentations`.

The public surface mirrors upstream's module layout, so a caller that reaches
for `albumentations.augmentations.pixel.functional.solarize` reaches for the
same name here:

    from mojo_albumentations.augmentations.pixel import functional as fpixel
    from mojo_albumentations.augmentations.blur import functional as fblur
    from mojo_albumentations.augmentations.geometric import functional as fgeometric

Every function here keeps upstream's name, argument order and defaults for the
subset listed in the README.
"""

from __future__ import annotations

from .augmentations.blur.functional import (
    box_blur,
    create_gaussian_kernel,
    create_gaussian_kernel_1d,
    create_gaussian_kernel_input_array,
    create_motion_kernel,
)
from .augmentations.geometric.functional import (
    copy_make_border_with_value_extension,
    create_affine_transformation_matrix,
    d4,
    dilate,
    erode,
    extend_value,
    generate_displacement_fields,
    hflip,
    morphology,
    pad,
    remap,
    rot90,
    transpose,
    vflip,
    warp_affine,
)
from .augmentations.pixel.functional import (
    add_noise,
    adjust_brightness_torchvision,
    adjust_contrast_torchvision,
    adjust_saturation_torchvision,
    apply_gaussian_illumination,
    apply_linear_illumination,
    apply_salt_and_pepper,
    auto_contrast,
    channel_shuffle,
    convolve,
    create_contrast_lut,
    create_directional_gradient,
    equalize,
    gamma_transform,
    get_histogram_bounds,
    grayscale_to_multichannel,
    invert,
    linear_transformation_rgb,
    move_tone_curve,
    posterize,
    rgb_to_optical_density,
    separable_convolve,
    sharpen_gaussian,
    solarize,
    to_gray_average,
    to_gray_desaturation,
    to_gray_max,
    to_gray_weighted_average,
    unsharp_mask,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "add_noise",
    "adjust_brightness_torchvision",
    "adjust_contrast_torchvision",
    "adjust_saturation_torchvision",
    "apply_gaussian_illumination",
    "apply_linear_illumination",
    "apply_salt_and_pepper",
    "auto_contrast",
    "box_blur",
    "channel_shuffle",
    "convolve",
    "copy_make_border_with_value_extension",
    "create_affine_transformation_matrix",
    "create_contrast_lut",
    "create_directional_gradient",
    "create_gaussian_kernel",
    "create_gaussian_kernel_1d",
    "create_gaussian_kernel_input_array",
    "create_motion_kernel",
    "d4",
    "dilate",
    "equalize",
    "erode",
    "extend_value",
    "gamma_transform",
    "generate_displacement_fields",
    "get_histogram_bounds",
    "grayscale_to_multichannel",
    "hflip",
    "invert",
    "linear_transformation_rgb",
    "morphology",
    "move_tone_curve",
    "pad",
    "posterize",
    "remap",
    "rgb_to_optical_density",
    "rot90",
    "separable_convolve",
    "sharpen_gaussian",
    "solarize",
    "to_gray_average",
    "to_gray_desaturation",
    "to_gray_max",
    "to_gray_weighted_average",
    "transpose",
    "unsharp_mask",
    "vflip",
    "warp_affine",
]