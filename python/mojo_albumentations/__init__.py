"""mojo-albumentations: the image-transform inner loops of albumentations, in Mojo.

Installable alongside the real ``albumentations``, which it is tested against
for parity entry by entry. The transform classes, parameter samplers, bboxes,
keypoints and composition stay upstream; what is here is the per-pixel work.
"""

from ._lib import (
    BORDER_CONSTANT,
    BORDER_REPLICATE,
    GRAY_AVERAGE,
    GRAY_MAX,
    GRAY_WEIGHTED,
    apply_lut,
    blur_separable,
    channel_stats,
)
from .transforms import (
    channel_shuffle,
    desaturate,
    gaussian_blur,
    gaussian_kernel_1d,
    invert,
    mean_std,
    move_tone_curve,
    posterize,
    remap_nearest,
    solarize,
    to_gray,
    warp_affine_map,
)

__all__ = [
    "BORDER_CONSTANT",
    "BORDER_REPLICATE",
    "GRAY_AVERAGE",
    "GRAY_MAX",
    "GRAY_WEIGHTED",
    "apply_lut",
    "blur_separable",
    "channel_shuffle",
    "channel_stats",
    "desaturate",
    "gaussian_blur",
    "gaussian_kernel_1d",
    "invert",
    "mean_std",
    "move_tone_curve",
    "posterize",
    "remap_nearest",
    "solarize",
    "to_gray",
    "warp_affine_map",
]
__version__ = "0.1.0"
