"""
Image quality assessment functions.

This module provides functions for evaluating image sharpness and quality
using various methods like Laplacian and Sobel variance.
"""

import cv2


def Image_Quality(image, IQ_Method, laplacianksize=3, SobelSize=3):
    """
    Compute image quality metric based on edge detection variance.
    
    Higher values indicate sharper images with more distinct edges.
    This is useful for autofocus, image selection, and quality assessment.
    
    Args:
        image: Input image (grayscale or color)
        IQ_Method: Quality assessment method - "Laplacian" or "Sobel"
        laplacianksize: Kernel size for Laplacian operator (default: 3)
        SobelSize: Kernel size for Sobel operator (default: 3)
        
    Returns:
        Float value representing image quality/sharpness
        
    Methods:
        - Laplacian: Second derivative edge detection, sensitive to rapid intensity changes
        - Sobel: First derivative edge detection in both X and Y directions
        
    Example:
        >>> quality = Image_Quality(frame, "Laplacian")
        >>> if quality > threshold:
        >>>     save_frame(frame)  # Frame is sharp enough
    """
    if IQ_Method == "Laplacian":
        image = cv2.GaussianBlur(image, (3, 3), 0)
        Image_Qual = cv2.Laplacian(image, cv2.CV_64F, ksize=laplacianksize).var()
    elif IQ_Method == "Sobel":
        image = cv2.GaussianBlur(image, (3, 3), 0)
        Image_Qual = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=SobelSize).var()
    else:
        # Default to Laplacian if unknown method
        image = cv2.GaussianBlur(image, (3, 3), 0)
        Image_Qual = cv2.Laplacian(image, cv2.CV_64F, ksize=laplacianksize).var()
        
    return Image_Qual


def compute_focus_score(image, method="Laplacian"):
    """
    Convenience function for computing focus score (alias for Image_Quality).
    
    Args:
        image: Input image
        method: "Laplacian" or "Sobel"
        
    Returns:
        Focus score (higher = sharper)
    """
    return Image_Quality(image, method)
