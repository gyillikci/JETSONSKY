# Processing Module

Core image processing functions extracted from the monolithic JetsonSky codebase.

## Overview

This module contains essential image processing operations that were previously embedded in the main application file. Breaking these out improves:
- **Code organization**: Related functions grouped logically
- **Testability**: Individual functions can be unit tested
- **Reusability**: Functions can be imported and used independently
- **Maintainability**: Easier to understand and modify

## Module Structure

```
processing/
├── __init__.py           # Package exports
├── image_utils.py        # Color space conversions (~200 lines)
├── quality.py            # Image quality assessment (~60 lines)
├── stabilization.py      # Template-based stabilization (~250 lines)
├── debayer.py            # Debayering and HDR (~200 lines)
└── README.md            # This file
```

## Modules

### image_utils.py
RGB channel operations and color space conversions.

**Functions:**
- `cupy_RGBImage_2_cupy_separateRGB()` - Split CuPy RGB → separate channels
- `numpy_RGBImage_2_numpy_separateRGB()` - Split NumPy RGB → separate channels
- `numpy_RGBImage_2_cupy_separateRGB()` - Convert and split NumPy → CuPy
- `cupy_RGBImage_2_numpy_separateRGB()` - Convert and split CuPy → NumPy
- `cupy_separateRGB_2_numpy_RGBimage()` - Merge CuPy channels → NumPy RGB
- `cupy_separateRGB_2_cupy_RGBimage()` - Merge CuPy channels → CuPy RGB
- `numpy_separateRGB_2_numpy_RGBimage()` - Merge NumPy channels → NumPy RGB
- `gaussianblur_mono()` - Gaussian blur for monochrome images
- `gaussianblur_colour()` - Gaussian blur for RGB channels
- `image_negative_colour()` - Invert color channels

**Usage:**
```python
from processing.image_utils import cupy_RGBImage_2_cupy_separateRGB

# Split RGB image into separate channels
b, g, r = cupy_RGBImage_2_cupy_separateRGB(image)
```

### quality.py
Image quality and sharpness assessment using edge detection.

**Functions:**
- `Image_Quality()` - Compute quality metric (Laplacian or Sobel variance)
- `compute_focus_score()` - Convenience wrapper for focus assessment

**Usage:**
```python
from processing.quality import Image_Quality

# Assess image sharpness
quality = Image_Quality(frame, "Laplacian")
if quality > threshold:
    save_frame(frame)  # Frame is sharp enough
```

**Methods:**
- **Laplacian**: Second derivative, sensitive to rapid intensity changes
- **Sobel**: First derivative in X and Y directions

### stabilization.py
Template-based image stabilization for video streams.

**Classes:**
- `TemplateStabilizer` - OOP stabilization with state management

**Functions:**
- `Template_tracking()` - Legacy wrapper for backward compatibility

**Usage:**
```python
from processing.stabilization import TemplateStabilizer

# Initialize stabilizer
stabilizer = TemplateStabilizer(res_x=3840, res_y=2160, use_cuda=False)

# Process frames
for frame in video_stream:
    stabilized = stabilizer.process_frame(frame, dim=3)
```

**Features:**
- Template matching with configurable window size
- Keyboard control for manual adjustment
- CUDA acceleration support
- Confidence threshold (0.2) for match quality
- 2x buffer for stabilization in all directions

### debayer.py
Bayer pattern debayering and HDR processing.

**Functions:**
- `opencv_color_debayer()` - Convert Bayer pattern to RGB
- `HDR_compute()` - Create HDR image from 16-bit camera data
- `get_bayer_pattern()` - Get OpenCV constant from camera info

**Usage:**
```python
from processing.debayer import opencv_color_debayer, HDR_compute

# Debayer raw camera image
rgb = opencv_color_debayer(raw_image, cv2.COLOR_BAYER_RG2BGR, cuda_flag=True)

# Compute HDR image
hdr = HDR_compute(
    mono_colour="Colour",
    image_16b=raw_16bit,
    method="Mertens",
    threshold_16b=12,
    BIN_mode=1,
    Hard_BIN=False,
    type_bayer=cv2.COLOR_BAYER_RG2BGR
)
```

**Bayer Patterns:**
- RGGB: `cv2.COLOR_BAYER_RG2BGR`
- BGGR: `cv2.COLOR_BAYER_BG2BGR`
- GRBG: `cv2.COLOR_BAYER_GR2BGR`
- GBRG: `cv2.COLOR_BAYER_GB2BGR`

**HDR Methods:**
- **Mertens**: Exposure fusion without tone mapping (best local contrast)
- **Median**: Robust to outliers
- **Mean**: Smoothest result

## Integration

To use these modules in the main application:

```python
# Old way (monolithic)
from JetsonSky_Linux_Windows_V53_07RC import Image_Quality, Template_tracking

# New way (modular)
from processing import Image_Quality, Template_tracking
from processing.stabilization import TemplateStabilizer
```

## Benefits

1. **Reduced main file size**: ~700 lines extracted from 8,957-line monolith
2. **Improved imports**: Clear dependencies and module boundaries
3. **Better testing**: Each module can be tested independently
4. **Documentation**: Comprehensive docstrings for all functions
5. **Type hints**: Coming soon for better IDE support

## Next Steps

- Add unit tests for each module
- Add type hints for better IDE support
- Consider moving to dataclasses for configuration
- Optimize CUDA operations
- Add performance benchmarks

## Original Code Location

These functions were extracted from:
- Lines 1347-1450: `opencv_color_debayer()`, `HDR_compute()`
- Lines 3505-3730: Image utilities, quality, stabilization
- JetsonSky_Linux_Windows_V53_07RC.py

## Dependencies

- NumPy
- CuPy
- OpenCV (cv2)
- SciPy (ndimage)
