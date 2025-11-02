"""
Image Stabilization Module

This module provides template-based image stabilization using OpenCV template matching.
Ported from the original JetsonSky monolithic code for clean separation.

Author: Refactored from JetsonSky V53_07RC
Performance: Maintains original stabilization algorithm
"""

import numpy as np
import cv2

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


class ImageStabilizer:
    """
    Template-based image stabilization for astronomy imaging.

    Uses template matching to track and stabilize video frames, compensating
    for atmospheric turbulence, mount drift, and other motion artifacts.

    Features:
    - Automatic template selection from center region
    - Configurable template size
    - Manual offset adjustment
    - Confidence-based tracking
    - Works with color and mono images
    """

    def __init__(self, resolution_x: int, resolution_y: int):
        """
        Initialize stabilizer.

        Args:
            resolution_x: Camera resolution width
            resolution_y: Camera resolution height
        """
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

        # Stabilization state
        self.enabled = False
        self.template_initialized = False
        self.template = None

        # Offset tracking
        self.delta_tx = 0  # X offset
        self.delta_ty = 0  # Y offset
        self.dsw = 0       # Template size adjustment (0-12)

        # Template region coordinates
        self.stab_start_point = (0, 0)
        self.stab_end_point = (0, 0)
        self.flag_new_stab_window = False

        # Template matching confidence threshold
        self.confidence_threshold = 0.2

    def enable(self):
        """Enable stabilization."""
        self.enabled = True
        print("✓ Stabilization enabled")

    def disable(self):
        """Disable stabilization and reset template."""
        self.enabled = False
        self.reset()
        print("✓ Stabilization disabled")

    def reset(self):
        """Reset stabilization template and offsets."""
        self.template_initialized = False
        self.template = None
        self.delta_tx = 0
        self.delta_ty = 0
        self.dsw = 0
        self.flag_new_stab_window = False

    def adjust_offset(self, dx: int, dy: int):
        """
        Manually adjust stabilization offset.

        Args:
            dx: X offset adjustment
            dy: Y offset adjustment
        """
        self.delta_tx += dx
        self.delta_ty += dy
        self.template_initialized = False  # Force template re-initialization

    def adjust_template_size(self, delta: int):
        """
        Adjust template window size.

        Args:
            delta: Size adjustment (-1 to decrease, +1 to increase)
        """
        self.dsw += delta
        self.dsw = max(0, min(12, self.dsw))  # Clamp to 0-12
        self.template_initialized = False  # Force template re-initialization

    def stabilize(self, image: np.ndarray, is_color: bool = True) -> np.ndarray:
        """
        Stabilize image using template matching.

        Args:
            image: Input image (NumPy or CuPy array, uint8)
            is_color: True if color image (3 channels), False if mono

        Returns:
            Stabilized image (same type as input)
        """
        if not self.enabled:
            return image

        # Convert CuPy to NumPy if needed
        is_cupy = HAS_CUPY and isinstance(image, cp.ndarray)
        if is_cupy:
            image = cp.asnumpy(image)

        # First frame or reset: initialize template
        if not self.template_initialized:
            self._initialize_template(image, is_color)
            result = image
        else:
            # Subsequent frames: track and stabilize
            result = self._apply_stabilization(image, is_color)

        # Convert back to CuPy if original was CuPy
        if is_cupy:
            result = cp.asarray(result)

        return result

    def _initialize_template(self, image: np.ndarray, is_color: bool):
        """
        Initialize template from center region of image.

        Args:
            image: Input image (NumPy array)
            is_color: True if color image
        """
        # Save old values for bounds checking
        old_tx = self.delta_tx
        old_ty = self.delta_ty

        # Calculate template region based on resolution
        # (exact logic from V53_07RC)
        if self.resolution_x > 1500:
            # High resolution mode
            rs = self.resolution_y // 2 - self.resolution_y // (8 + self.dsw) + self.delta_ty
            re = self.resolution_y // 2 + self.resolution_y // (8 + self.dsw) + self.delta_ty
            cs = self.resolution_x // 2 - self.resolution_x // (8 + self.dsw) + self.delta_tx
            ce = self.resolution_x // 2 + self.resolution_x // (8 + self.dsw) + self.delta_tx

            # Bounds checking with restoration of old values
            if cs < 30 or ce > (self.resolution_x - 30):
                self.delta_tx = old_tx
                cs = self.resolution_x // 2 - self.resolution_x // (8 + self.dsw) + self.delta_tx
                ce = self.resolution_x // 2 + self.resolution_x // (8 + self.dsw) + self.delta_tx
            if rs < 30 or re > (self.resolution_y - 30):
                self.delta_ty = old_ty
                rs = self.resolution_y // 2 - self.resolution_y // (8 + self.dsw) + self.delta_ty
                re = self.resolution_y // 2 + self.resolution_y // (8 + self.dsw) + self.delta_ty
        else:
            # Lower resolution mode
            rs = self.resolution_y // 2 - self.resolution_y // (3 + self.dsw) + self.delta_ty
            re = self.resolution_y // 2 + self.resolution_y // (3 + self.dsw) + self.delta_ty
            cs = self.resolution_x // 2 - self.resolution_x // (3 + self.dsw) + self.delta_tx
            ce = self.resolution_x // 2 + self.resolution_x // (3 + self.dsw) + self.delta_tx

            # Bounds checking with restoration of old values
            if cs < 30 or ce > (self.resolution_x - 30):
                self.delta_tx = old_tx
                cs = self.resolution_x // 2 - self.resolution_x // (3 + self.dsw) + self.delta_tx
                ce = self.resolution_x // 2 + self.resolution_x // (3 + self.dsw) + self.delta_tx
            if rs < 30 or re > (self.resolution_y - 30):
                self.delta_ty = old_ty
                rs = self.resolution_y // 2 - self.resolution_y // (3 + self.dsw) + self.delta_ty
                re = self.resolution_y // 2 + self.resolution_y // (3 + self.dsw) + self.delta_ty

        # Store template region
        self.stab_start_point = (cs, rs)
        self.stab_end_point = (ce, re)

        # Extract template region
        self.template = image[rs:re, cs:ce].copy()

        # Convert to grayscale if color
        if is_color:
            self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Ensure template is uint8
        if self.template.dtype != np.uint8:
            self.template = np.clip(self.template, 0, 255).astype(np.uint8)

        self.template_initialized = True
        self.flag_new_stab_window = True

        print(f"✓ Template initialized: region ({cs}, {rs}) to ({ce}, {re}), size {ce-cs}x{re-rs}")

    def _apply_stabilization(self, image: np.ndarray, is_color: bool) -> np.ndarray:
        """
        Apply stabilization using template matching.

        Args:
            image: Input image (NumPy array)
            is_color: True if color image

        Returns:
            Stabilized image
        """
        self.flag_new_stab_window = False

        # Convert to grayscale for template matching
        if is_color:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Ensure uint8 format
        if image_gray.dtype != np.uint8:
            image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
        if self.template.dtype != np.uint8:
            self.template = np.clip(self.template, 0, 255).astype(np.uint8)

        # Perform template matching
        result = cv2.matchTemplate(image_gray, self.template, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

        # Check confidence threshold
        if maxVal < self.confidence_threshold:
            # Match confidence too low, return original
            return image

        try:
            # Calculate stabilization offset
            (startX, startY) = maxLoc
            midX = startX + self.template.shape[1] // 2
            midY = startY + self.template.shape[0] // 2

            DeltaX = image.shape[1] // 2 + self.delta_tx - midX
            DeltaY = image.shape[0] // 2 + self.delta_ty - midY

            # Create larger canvas for stabilization
            width = int(image.shape[1] * 2)
            height = int(image.shape[0] * 2)

            if is_color:
                tmp_image = np.zeros((height, width, 3), np.uint8)
            else:
                tmp_image = np.zeros((height, width), np.uint8)

            # Apply stabilization shift
            rs = int(self.resolution_y / 4 + DeltaY)
            re = int(rs + self.resolution_y)
            cs = int(self.resolution_x / 4 + DeltaX)
            ce = int(cs + self.resolution_x)
            tmp_image[rs:re, cs:ce] = image

            # Extract stabilized region
            rs = self.resolution_y // 4
            re = self.resolution_y // 4 + self.resolution_y
            cs = self.resolution_x // 4
            ce = self.resolution_x // 4 + self.resolution_x
            stabilized = tmp_image[rs:re, cs:ce]

            return stabilized

        except Exception as e:
            # Silent failure, return original image
            return image

    def get_template_region(self):
        """
        Get current template region coordinates.

        Returns:
            Tuple of ((x1, y1), (x2, y2)) or None if not initialized
        """
        if self.template_initialized:
            return (self.stab_start_point, self.stab_end_point)
        return None

    def get_status(self) -> dict:
        """
        Get stabilization status information.

        Returns:
            Dictionary with status information
        """
        return {
            'enabled': self.enabled,
            'template_initialized': self.template_initialized,
            'delta_tx': self.delta_tx,
            'delta_ty': self.delta_ty,
            'template_size_adj': self.dsw,
            'confidence_threshold': self.confidence_threshold,
            'template_region': self.get_template_region()
        }
