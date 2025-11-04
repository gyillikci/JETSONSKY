"""
Core image processing functions for JetsonSky.

This package provides essential image processing operations including:
- Color space conversions and channel operations
- Image quality assessment
- Template-based stabilization
- Debayering and HDR processing
"""

from .image_utils import (
    cupy_RGBImage_2_cupy_separateRGB,
    numpy_RGBImage_2_numpy_separateRGB,
    numpy_RGBImage_2_cupy_separateRGB,
    cupy_RGBImage_2_numpy_separateRGB,
    cupy_separateRGB_2_numpy_RGBimage,
    cupy_separateRGB_2_cupy_RGBimage,
    numpy_separateRGB_2_numpy_RGBimage,
    gaussianblur_mono,
    gaussianblur_colour,
    image_negative_colour,
)

from .quality import (
    Image_Quality,
    compute_focus_score,
)

from .stabilization import (
    doe as TemplateStabilizer,  # Main stabilizer class
    Template_tracking,
)

from .debayer import (
    opencv_color_debayer,
    HDR_compute,
    get_bayer_pattern,
)

__all__ = [
    # Image utilities
    'cupy_RGBImage_2_cupy_separateRGB',
    'numpy_RGBImage_2_numpy_separateRGB',
    'numpy_RGBImage_2_cupy_separateRGB',
    'cupy_RGBImage_2_numpy_separateRGB',
    'cupy_separateRGB_2_numpy_RGBimage',
    'cupy_separateRGB_2_cupy_RGBimage',
    'numpy_separateRGB_2_numpy_RGBimage',
    'gaussianblur_mono',
    'gaussianblur_colour',
    'image_negative_colour',
    
    # Quality assessment
    'Image_Quality',
    'compute_focus_score',
    
    # Stabilization
    'TemplateStabilizer',
    'Template_tracking',
    
    # Debayering and HDR
    'opencv_color_debayer',
    'HDR_compute',
    'get_bayer_pattern',
]
