"""
Image utility functions for color space conversions and RGB channel operations.

This module provides functions for:
- Converting between RGB and separate channel formats
- Converting between CuPy and NumPy arrays
- Gaussian blur operations
- Image negation
"""

import numpy as np
import cupy as cp
from cupyx.scipy import ndimage


def cupy_RGBImage_2_cupy_separateRGB(cupyImageRGB):
    """
    Convert CuPy RGB image to separate R, G, B CuPy arrays.
    
    Args:
        cupyImageRGB: CuPy array with shape (H, W, 3)
        
    Returns:
        Tuple of (B, G, R) as separate CuPy uint8 arrays
    """
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:, :, 0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:, :, 1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:, :, 2], dtype=cp.uint8)
    
    return cupy_B, cupy_G, cupy_R


def numpy_RGBImage_2_numpy_separateRGB(numpyImageRGB):
    """
    Convert NumPy RGB image to separate R, G, B NumPy arrays.
    
    Args:
        numpyImageRGB: NumPy array with shape (H, W, 3)
        
    Returns:
        Tuple of (R, G, B) as separate NumPy uint8 arrays
    """
    numpy_R = np.ascontiguousarray(numpyImageRGB[:, :, 0], dtype=np.uint8)
    numpy_G = np.ascontiguousarray(numpyImageRGB[:, :, 1], dtype=np.uint8)
    numpy_B = np.ascontiguousarray(numpyImageRGB[:, :, 2], dtype=np.uint8)
    
    return numpy_R, numpy_G, numpy_B


def numpy_RGBImage_2_cupy_separateRGB(numpyImageRGB):
    """
    Convert NumPy RGB image to separate R, G, B CuPy arrays.
    
    Args:
        numpyImageRGB: NumPy array with shape (H, W, 3)
        
    Returns:
        Tuple of (R, G, B) as separate CuPy uint8 arrays
    """
    cupyImageRGB = cp.asarray(numpyImageRGB)
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:, :, 0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:, :, 1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:, :, 2], dtype=cp.uint8)
    
    return cupy_R, cupy_G, cupy_B


def cupy_RGBImage_2_numpy_separateRGB(cupyImageRGB):
    """
    Convert CuPy RGB image to separate R, G, B NumPy arrays.
    
    Args:
        cupyImageRGB: CuPy array with shape (H, W, 3)
        
    Returns:
        Tuple of (R, G, B) as separate NumPy uint8 arrays
    """
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:, :, 0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:, :, 1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:, :, 2], dtype=cp.uint8)
    numpy_R = cupy_R.get()
    numpy_G = cupy_G.get()
    numpy_B = cupy_B.get()
    
    return numpy_R, numpy_G, numpy_B


def cupy_separateRGB_2_numpy_RGBimage(cupyR, cupyG, cupyB):
    """
    Merge separate CuPy R, G, B arrays into a NumPy RGB image.
    
    Args:
        cupyR: CuPy array for red channel
        cupyG: CuPy array for green channel
        cupyB: CuPy array for blue channel
        
    Returns:
        NumPy RGB image with shape (H, W, 3)
    """
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    numpyRGB = cupyRGB.get()
    
    return numpyRGB


def cupy_separateRGB_2_cupy_RGBimage(cupyR, cupyG, cupyB):
    """
    Merge separate CuPy R, G, B arrays into a CuPy RGB image.
    
    Args:
        cupyR: CuPy array for red channel
        cupyG: CuPy array for green channel
        cupyB: CuPy array for blue channel
        
    Returns:
        CuPy RGB image with shape (H, W, 3)
    """
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    
    return cupyRGB


def numpy_separateRGB_2_numpy_RGBimage(npR, npG, npB):
    """
    Merge separate NumPy R, G, B arrays into a NumPy RGB image.
    
    Args:
        npR: NumPy array for red channel
        npG: NumPy array for green channel
        npB: NumPy array for blue channel
        
    Returns:
        NumPy RGB image with shape (H, W, 3)
    """
    rgb = (npR[..., np.newaxis], npG[..., np.newaxis], npB[..., np.newaxis])
    numpyRGB = np.concatenate(rgb, axis=-1, dtype=np.uint8)
    
    return numpyRGB


def gaussianblur_mono(image_mono, niveau_blur):
    """
    Apply Gaussian blur to monochrome image.
    
    Args:
        image_mono: Monochrome image array
        niveau_blur: Sigma value for Gaussian kernel
        
    Returns:
        Blurred monochrome image
    """
    image_gaussian_blur_mono = ndimage.gaussian_filter(image_mono, sigma=niveau_blur)
    
    return image_gaussian_blur_mono


def gaussianblur_colour(im_r, im_g, im_b, niveau_blur):
    """
    Apply Gaussian blur to separate RGB channels.
    
    Args:
        im_r: Red channel array
        im_g: Green channel array
        im_b: Blue channel array
        niveau_blur: Sigma value for Gaussian kernel
        
    Returns:
        Tuple of (blurred_r, blurred_g, blurred_b)
    """
    im_GB_r = ndimage.gaussian_filter(im_r, sigma=niveau_blur)
    im_GB_g = ndimage.gaussian_filter(im_g, sigma=niveau_blur)
    im_GB_b = ndimage.gaussian_filter(im_b, sigma=niveau_blur)
    
    return im_GB_r, im_GB_g, im_GB_b


def image_negative_colour(red, green, blue):
    """
    Invert color channels (create negative image).
    
    Args:
        red: CuPy array for red channel
        green: CuPy array for green channel
        blue: CuPy array for blue channel
        
    Returns:
        Tuple of (inverted_red, inverted_green, inverted_blue)
    """
    blue = cp.invert(blue, dtype=cp.uint8)
    green = cp.invert(green, dtype=cp.uint8)
    red = cp.invert(red, dtype=cp.uint8)
    
    return red, green, blue
