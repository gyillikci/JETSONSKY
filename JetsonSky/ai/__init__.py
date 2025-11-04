"""
AI-based detection module for satellite and star tracking.

This module provides AI-powered detection algorithms for:
- Satellite tracking using frame differencing
- Star detection using blob detection
- Image reconstruction with enhanced stars
"""

from .detection import (
    satellites_tracking_AI,
    satellites_tracking,
    remove_satellites,
    stars_detection,
    draw_star,
    draw_satellite,
    reconstruction_image
)

__all__ = [
    'satellites_tracking_AI',
    'satellites_tracking',
    'remove_satellites',
    'stars_detection',
    'draw_star',
    'draw_satellite',
    'reconstruction_image'
]
