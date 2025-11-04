"""
Template-based image stabilization using OpenCV template matching.

This module provides real-time image stabilization by tracking a template region
across frames and compensating for motion.
"""

import cv2
import numpy as np


class doe:
    """
    Template-based image stabilization for video streams.
    
    Uses template matching to track a region across frames and applies
    stabilization by shifting the image to compensate for motion.
    
    Attributes:
        flag_template: Whether template has been initialized
        template: Template image for matching
        delta_tx: X offset for template position
        delta_ty: Y offset for template position
        DSW: Template window size adjustment (0-12)
        start_point: Top-left corner of template region
        end_point: Bottom-right corner of template region
        flag_new_stab_window: Whether a new stabilization window was created
        flag_opencv_cuda: Whether to use OpenCV CUDA acceleration
    """
    
    def __init__(self, res_cam_x, res_cam_y, use_cuda=False):
        """
        Initialize stabilizer.
        
        Args:
            res_cam_x: Camera resolution width
            res_cam_y: Camera resolution height
            use_cuda: Whether to use CUDA acceleration (if available)
        """
        self.res_cam_x = res_cam_x
        self.res_cam_y = res_cam_y
        self.use_cuda = use_cuda
        
        # Stabilization state
        self.flag_template = False
        self.template = None
        self.delta_tx = 0
        self.delta_ty = 0
        self.DSW = 0  # Window size adjustment (0-12)
        self.start_point = (0, 0)
        self.end_point = (0, 0)
        self.flag_new_stab_window = False
        
        # Previous match location for incremental search
        self.last_match_x = None
        self.last_match_y = None
        
        # Search parameters
        self.initial_search_radius = 100  # Pixels to search around last location
        self.search_expansion_step = 50   # Pixels to expand if no good match
        
        # OpenCV CUDA objects
        if self.use_cuda:
            try:
                self.gsrc = cv2.cuda_GpuMat()
                self.gtmpl = cv2.cuda_GpuMat()
                self.gresult = cv2.cuda_GpuMat()
                self.matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
            except:
                self.use_cuda = False
        
    def reset_template(self):
        """Reset the template, forcing re-initialization on next frame."""
        self.flag_template = False
        self.template = None
        self.flag_new_stab_window = False
        self.last_match_x = None
        self.last_match_y = None
        
    def adjust_window_position(self, key_pressed):
        """
        Adjust template window position based on keyboard input.
        
        Args:
            key_pressed: String indicating key pressed
                        ("STAB_UP", "STAB_DOWN", "STAB_LEFT", "STAB_RIGHT",
                         "STAB_ZONE_MORE", "STAB_ZONE_LESS")
                         
        Returns:
            True if window was modified
        """
        flag_modif = False
        
        if key_pressed == "STAB_UP":
            self.delta_ty = self.delta_ty - 30
            flag_modif = True
        elif key_pressed == "STAB_DOWN":
            self.delta_ty = self.delta_ty + 30
            flag_modif = True
        elif key_pressed == "STAB_RIGHT":
            self.delta_tx = self.delta_tx + 30
            flag_modif = True
        elif key_pressed == "STAB_LEFT":
            self.delta_tx = self.delta_tx - 30
            flag_modif = True
        elif key_pressed == "STAB_ZONE_MORE":
            self.DSW = max(0, self.DSW - 1)
            flag_modif = True
        elif key_pressed == "STAB_ZONE_LESS":
            self.DSW = min(12, self.DSW + 1)
            flag_modif = True
            
        if flag_modif:
            self.flag_template = False
            
        return flag_modif
    
    def _calculate_template_region(self):
        """
        Calculate template region coordinates based on camera resolution.
        
        Returns:
            Tuple of (rs, re, cs, ce) - row start/end, col start/end
        """
        old_tx = self.delta_tx
        old_ty = self.delta_ty
        
        # Clamp DSW range
        self.DSW = max(0, min(12, self.DSW))
        
        if self.res_cam_x > 1500:
            # High resolution camera
            rs = self.res_cam_y // 2 - self.res_cam_y // (8 + self.DSW) + self.delta_ty
            re = self.res_cam_y // 2 + self.res_cam_y // (8 + self.DSW) + self.delta_ty
            cs = self.res_cam_x // 2 - self.res_cam_x // (8 + self.DSW) + self.delta_tx
            ce = self.res_cam_x // 2 + self.res_cam_x // (8 + self.DSW) + self.delta_tx
            
            # Bounds checking with restoration of old values
            if cs < 30 or ce > (self.res_cam_x - 30):
                self.delta_tx = old_tx
                cs = self.res_cam_x // 2 - self.res_cam_x // (8 + self.DSW) + self.delta_tx
                ce = self.res_cam_x // 2 + self.res_cam_x // (8 + self.DSW) + self.delta_tx
            if rs < 30 or re > (self.res_cam_y - 30):
                self.delta_ty = old_ty
                rs = self.res_cam_y // 2 - self.res_cam_y // (8 + self.DSW) + self.delta_ty
                re = self.res_cam_y // 2 + self.res_cam_y // (8 + self.DSW) + self.delta_ty
        else:
            # Lower resolution camera
            rs = self.res_cam_y // 2 - self.res_cam_y // (3 + self.DSW) + self.delta_ty
            re = self.res_cam_y // 2 + self.res_cam_y // (3 + self.DSW) + self.delta_ty
            cs = self.res_cam_x // 2 - self.res_cam_x // (3 + self.DSW) + self.delta_tx
            ce = self.res_cam_x // 2 + self.res_cam_x // (3 + self.DSW) + self.delta_tx
            
            # Bounds checking with restoration of old values
            if cs < 30 or ce > (self.res_cam_x - 30):
                self.delta_tx = old_tx
                cs = self.res_cam_x // 2 - self.res_cam_x // (3 + self.DSW) + self.delta_tx
                ce = self.res_cam_x // 2 + self.res_cam_x // (3 + self.DSW) + self.delta_tx
            if rs < 30 or re > (self.res_cam_y - 30):
                self.delta_ty = old_ty
                rs = self.res_cam_y // 2 - self.res_cam_y // (3 + self.DSW) + self.delta_ty
                re = self.res_cam_y // 2 + self.res_cam_y // (3 + self.DSW) + self.delta_ty
                
        return rs, re, cs, ce
    
    def _incremental_template_match(self, image_gray):
        """
        Perform incremental template matching starting near previous location.
        
        Searches in expanding regions around the last known match location
        for much faster performance than full-frame search.
        
        Args:
            image_gray: Grayscale image to search
            
        Returns:
            Tuple of (maxVal, maxLoc) - best match confidence and location
        """
        h, w = image_gray.shape
        tmpl_h, tmpl_w = self.template.shape
        
        # If no previous location, search full frame once
        if self.last_match_x is None or self.last_match_y is None:
            if self.use_cuda:
                self.gsrc.upload(image_gray)
                self.gresult = self.matcher.match(self.gsrc, self.gtmpl)
                result = self.gresult.download()
            else:
                result = cv2.matchTemplate(image_gray, self.template, cv2.TM_CCOEFF_NORMED)
            
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            self.last_match_x, self.last_match_y = maxLoc
            return maxVal, maxLoc
        
        # Start with small region around previous location
        search_radius = self.initial_search_radius
        max_radius = max(w, h)  # Maximum possible search
        best_val = 0
        best_loc = (self.last_match_x, self.last_match_y)
        
        while search_radius <= max_radius:
            # Calculate search region bounds
            x1 = max(0, self.last_match_x - search_radius)
            y1 = max(0, self.last_match_y - search_radius)
            x2 = min(w - tmpl_w, self.last_match_x + search_radius)
            y2 = min(h - tmpl_h, self.last_match_y + search_radius)
            
            # Extract search region (add template size to bounds)
            search_x2 = min(w, x2 + tmpl_w)
            search_y2 = min(h, y2 + tmpl_h)
            roi = image_gray[y1:search_y2, x1:search_x2]
            
            # Skip if ROI is too small
            if roi.shape[0] < tmpl_h or roi.shape[1] < tmpl_w:
                search_radius += self.search_expansion_step
                continue
            
            # Template match in this region
            if self.use_cuda:
                groi = cv2.cuda_GpuMat()
                groi.upload(roi)
                gresult_roi = self.matcher.match(groi, self.gtmpl)
                result = gresult_roi.download()
            else:
                result = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
            
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            
            # Found good match?
            if maxVal > 0.2:
                # Adjust coordinates back to full image space
                abs_x = x1 + maxLoc[0]
                abs_y = y1 + maxLoc[1]
                self.last_match_x = abs_x
                self.last_match_y = abs_y
                return maxVal, (abs_x, abs_y)
            
            # Store best so far
            if maxVal > best_val:
                best_val = maxVal
                abs_x = x1 + maxLoc[0]
                abs_y = y1 + maxLoc[1]
                best_loc = (abs_x, abs_y)
            
            # Expand search radius
            search_radius += self.search_expansion_step
            
            # If we've covered most of the image, stop
            if search_radius > min(w, h) // 2:
                break
        
        # Return best match found (even if below threshold)
        self.last_match_x, self.last_match_y = best_loc
        return best_val, best_loc
    
    def process_frame(self, image, dim, key_pressed=""):
        """
        Process a frame for stabilization.
        
        Args:
            image: Input image (color or grayscale)
            dim: Image dimensions (3 for color, 1 for grayscale)
            key_pressed: Optional keyboard command for window adjustment
            
        Returns:
            Stabilized image
        """
        # Handle keyboard commands
        if key_pressed:
            self.adjust_window_position(key_pressed)
        
        if not self.flag_template:
            # Initialize template from center region
            rs, re, cs, ce = self._calculate_template_region()
            
            self.start_point = (cs, rs)
            self.end_point = (ce, re)
            self.template = image[rs:re, cs:ce].copy()
            
            # Convert to grayscale if color image
            if dim == 3:
                self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                
            # Setup CUDA if enabled
            if self.use_cuda:
                self.gtmpl.upload(self.template)
                
            self.flag_template = True
            self.flag_new_stab_window = True
            return image
        else:
            # Template exists, perform tracking
            self.flag_new_stab_window = False
            
            # Convert to grayscale if needed
            if dim == 3:
                imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                imageGray = image
                
            # Incremental template matching (optimized)
            maxVal, maxLoc = self._incremental_template_match(imageGray)
            
            # Apply stabilization if match confidence is high enough
            if maxVal > 0.2:
                try:
                    # Create 2x buffer to allow stabilization in all directions
                    width = int(image.shape[1] * 2)
                    height = int(image.shape[0] * 2)
                    
                    if dim == 3:
                        tmp_image = np.zeros((height, width, 3), np.uint8)
                    else:
                        tmp_image = np.zeros((height, width), np.uint8)
                        
                    # Calculate stabilization offset
                    (startX, startY) = maxLoc
                    midX = startX + self.template.shape[1] // 2
                    midY = startY + self.template.shape[0] // 2
                    DeltaX = image.shape[1] // 2 + self.delta_tx - midX
                    DeltaY = image.shape[0] // 2 + self.delta_ty - midY
                    
                    # Place image with offset
                    rs = int(self.res_cam_y / 4 + DeltaY)  # Y up
                    re = int(rs + self.res_cam_y)  # Y down
                    cs = int(self.res_cam_x / 4 + DeltaX)  # X left
                    ce = int(cs + self.res_cam_x)  # X right
                    tmp_image[rs:re, cs:ce] = image
                    
                    # Extract stabilized region
                    rs = self.res_cam_y // 4  # Y up
                    re = self.res_cam_y // 4 + self.res_cam_y  # Y down
                    cs = self.res_cam_x // 4  # X left
                    ce = self.res_cam_x // 4 + self.res_cam_x  # X right
                    new_image = tmp_image[rs:re, cs:ce]
                    
                    return new_image
                except:
                    # If stabilization fails, return original
                    return image
            else:
                # Match confidence too low, return original
                return image


def Template_tracking(image, dim, stabilizer, key_pressed=""):
    """
    Legacy function wrapper for compatibility with original code.
    
    Args:
        image: Input image
        dim: Image dimensions (3 for color, 1 for grayscale)
        stabilizer: TemplateStabilizer instance
        key_pressed: Keyboard command string
        
    Returns:
        Stabilized image
    """
    return stabilizer.process_frame(image, dim, key_pressed)
