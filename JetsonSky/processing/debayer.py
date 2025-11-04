"""
Debayering and HDR processing functions for camera raw images.

This module handles:
- Bayer pattern debayering (color reconstruction from raw sensor data)
- HDR (High Dynamic Range) processing using multiple exposure levels
- GPU-accelerated processing when available
"""

import cv2
import numpy as np
import cupy as cp


def opencv_color_debayer(image, debayer_pattern, cuda_flag=False):
    """
    Debayer raw camera image to RGB using OpenCV.
    
    Converts a raw Bayer pattern image to RGB color image. Can use
    CUDA acceleration if available and enabled.
    
    Args:
        image: Raw Bayer pattern image
        debayer_pattern: OpenCV debayer pattern constant
                        (e.g., cv2.COLOR_BAYER_RG2BGR for RGGB pattern)
        cuda_flag: Whether to use CUDA acceleration (default: False)
        
    Returns:
        Debayered RGB image
        
    Bayer Patterns:
        - RGGB: cv2.COLOR_BAYER_RG2BGR
        - BGGR: cv2.COLOR_BAYER_BG2BGR
        - GRBG: cv2.COLOR_BAYER_GR2BGR
        - GBRG: cv2.COLOR_BAYER_GB2BGR
    """
    if cuda_flag:
        try:
            tmpbase = cv2.cuda_GpuMat()
            tmprsz = cv2.cuda_GpuMat()
            tmpbase.upload(image)
            tmprsz = cv2.cuda.cvtColor(tmpbase, debayer_pattern)
            debayer_image = tmprsz.download()
        except:
            # Fallback to CPU if CUDA fails
            debayer_image = cv2.cvtColor(image, debayer_pattern)
    else:
        debayer_image = cv2.cvtColor(image, debayer_pattern)

    return debayer_image


def HDR_compute(mono_colour, image_16b, method, threshold_16b, BIN_mode, Hard_BIN, type_bayer, flag_OpenCvCuda=False):
    """
    Compute HDR image from 16-bit camera data using multiple exposure levels.
    
    Creates multiple virtual exposures from a single 16-bit image by applying
    different thresholds, then merges them using various methods.
    
    Args:
        mono_colour: "Mono" or "Colour" - output format
        image_16b: Input 16-bit image from camera
        method: HDR merge method - "Mertens", "Median", or "Mean"
        threshold_16b: Bit depth threshold (typically 12-16)
        BIN_mode: Binning mode (1 or 2)
        Hard_BIN: Whether using hardware binning
        type_bayer: Bayer pattern for debayering
        flag_OpenCvCuda: Use CUDA acceleration for debayering
        
    Returns:
        8-bit HDR processed image
        
    Methods:
        - Mertens: Exposure fusion without tone mapping (preserves local contrast)
        - Median: Takes median value across exposures (robust to outliers)
        - Mean: Averages exposures (smoothest result)
        
    Process:
        1. Create 4 virtual exposures by clipping at different thresholds
        2. Convert each to 8-bit
        3. Merge using selected method
        4. Debayer to color (if color camera)
        5. Convert to mono if requested
    """
    # Calculate adaptive thresholds based on bit depth
    if (16 - threshold_16b) <= 5:
        delta_th = (16 - threshold_16b) / 3.0
    else:
        delta_th = 5.0 / 3.0
                            
    thres4 = 2 ** threshold_16b - 1
    thres3 = 2 ** (threshold_16b + delta_th) - 1
    thres2 = 2 ** (threshold_16b + delta_th * 2) - 1
    thres1 = 2 ** (threshold_16b + delta_th * 3) - 1
    
    # Create exposure 1 (brightest)
    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres1] = thres1
    if BIN_mode == 2:
        if not Hard_BIN:
            image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
        else:
            image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0)
    else:
        image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8, dtype=cp.uint8)
    image_brute1 = image_brute_cam_tmp.get()

    # Create exposure 2
    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres2] = thres2
    if BIN_mode == 2:
        if not Hard_BIN:
            image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
        else:
            image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0)
    else:
        image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8, dtype=cp.uint8)
    image_brute2 = image_brute_cam_tmp.get()

    # Create exposure 3
    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres3] = thres3
    if BIN_mode == 2:
        if not Hard_BIN:
            image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
        else:
            image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0)
    else:
        image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8, dtype=cp.uint8)
    image_brute3 = image_brute_cam_tmp.get()
    
    # Create exposure 4 (darkest)
    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres4] = thres4
    if BIN_mode == 2:
        if not Hard_BIN:
            image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
        else:
            image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0)
    else:
        image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8, 0, 255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8, dtype=cp.uint8)
    image_brute4 = image_brute_cam_tmp.get()

    img_list = [image_brute1, image_brute2, image_brute3, image_brute4]
    
    # Merge exposures using selected method
    if method == "Mertens":
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        res_mertens_cp = cp.asarray(res_mertens, dtype=cp.float32)
        image_brute_cp = cp.clip(res_mertens_cp * 255, 0, 255).astype('uint8')
    elif method == "Median":
        img_list = cp.asarray(img_list)
        tempo_hdr = cp.median(img_list, axis=0)
        image_brute_cp = cp.asarray(tempo_hdr, dtype=cp.uint8)
    elif method == "Mean":
        img_list = cp.asarray(img_list)
        tempo_hdr = cp.mean(img_list, axis=0)
        image_brute_cp = cp.asarray(tempo_hdr, dtype=cp.uint8)
    else:
        # Default to Mean if unknown method
        img_list = cp.asarray(img_list)
        tempo_hdr = cp.mean(img_list, axis=0)
        image_brute_cp = cp.asarray(tempo_hdr, dtype=cp.uint8)

    # Convert to numpy and debayer
    HDR_image = image_brute_cp.get()
    HDR_image = opencv_color_debayer(HDR_image, type_bayer, flag_OpenCvCuda)
    
    # Convert to mono if requested
    if mono_colour == "Mono":
        HDR_image = cv2.cvtColor(HDR_image, cv2.COLOR_BGR2GRAY)
            
    return HDR_image


def get_bayer_pattern(camera_info):
    """
    Get OpenCV debayer pattern constant from camera info.
    
    Args:
        camera_info: Camera information dict with 'BayerPattern' key
        
    Returns:
        OpenCV color conversion constant for debayering
    """
    bayer_map = {
        0: cv2.COLOR_BAYER_RG2BGR,  # RGGB
        1: cv2.COLOR_BAYER_BG2BGR,  # BGGR
        2: cv2.COLOR_BAYER_GR2BGR,  # GRBG
        3: cv2.COLOR_BAYER_GB2BGR,  # GBRG
    }
    
    pattern = camera_info.get('BayerPattern', 0)
    return bayer_map.get(pattern, cv2.COLOR_BAYER_RG2BGR)
