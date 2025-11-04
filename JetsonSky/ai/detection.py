"""
Satellite and star detection algorithms.

This module implements detection and tracking algorithms for celestial objects:
- AI-based satellite tracking using frame differencing
- Traditional blob-based satellite tracking
- Star detection using SimpleBlobDetector
- Satellite removal for cleaner images
- Image reconstruction with enhanced celestial objects
"""

import cv2
import numpy as np
import cupy as cp
import math
import random
from typing import Tuple, Optional


def satellites_tracking_AI(
    image_traitee: np.ndarray,
    img_sat_buf1_AI: np.ndarray,
    img_sat_buf2_AI: np.ndarray,
    img_sat_buf3_AI: np.ndarray,
    img_sat_buf4_AI: np.ndarray,
    img_sat_buf5_AI: Optional[np.ndarray],
    sat_frame_count_AI: int,
    sat_frame_target_AI: int,
    flag_first_sat_pass_AI: bool,
    flag_IsColor: bool,
    cupy_context,
    Dead_Pixels_Remove_Mono_GPU,
    nb_ThreadsX: int,
    nb_ThreadsY: int,
    gaussianblur_mono
) -> Tuple[bool, np.ndarray, dict]:
    """
    AI-based satellite tracking using frame differencing and GPU acceleration.
    
    Uses a 5-frame buffer to detect moving objects (satellites) by computing
    differences between frames. GPU acceleration is used for noise removal.
    
    Args:
        image_traitee: Current processed image
        img_sat_buf1_AI through img_sat_buf5_AI: 5-frame circular buffer
        sat_frame_count_AI: Current frame count in buffer
        sat_frame_target_AI: Target number of frames (typically 5)
        flag_first_sat_pass_AI: Whether this is the first pass
        flag_IsColor: Whether processing color or grayscale
        cupy_context: CuPy context for GPU operations
        Dead_Pixels_Remove_Mono_GPU: GPU kernel for dead pixel removal
        nb_ThreadsX, nb_ThreadsY: GPU thread configuration
        gaussianblur_mono: GPU gaussian blur function
        
    Returns:
        Tuple of (detection_flag, result_image, updated_state):
            - detection_flag: True if satellites detected in this frame
            - result_image: Processed image with satellite mask or original
            - updated_state: Dictionary with updated buffer state
            
    Example:
        >>> flag, result, state = satellites_tracking_AI(
        ...     frame, buf1, buf2, buf3, buf4, buf5, count, 5, True, True,
        ...     ctx, gpu_kernel, 32, 32, blur_fn
        ... )
    """
    # Update buffer state
    state = {
        'img_sat_buf1_AI': img_sat_buf1_AI,
        'img_sat_buf2_AI': img_sat_buf2_AI,
        'img_sat_buf3_AI': img_sat_buf3_AI,
        'img_sat_buf4_AI': img_sat_buf4_AI,
        'img_sat_buf5_AI': img_sat_buf5_AI,
        'sat_frame_count_AI': sat_frame_count_AI,
        'flag_first_sat_pass_AI': flag_first_sat_pass_AI,
        'flag_img_sat_buf5_AI': False
    }
    
    # Fill the buffer
    if sat_frame_count_AI < sat_frame_target_AI and flag_first_sat_pass_AI:
        sat_frame_count_AI += 1
        state['sat_frame_count_AI'] = sat_frame_count_AI
        
        if sat_frame_count_AI == 1:
            state['img_sat_buf1_AI'] = image_traitee.copy()
        elif sat_frame_count_AI == 2:
            state['img_sat_buf2_AI'] = image_traitee.copy()
        elif sat_frame_count_AI == 3:
            state['img_sat_buf3_AI'] = image_traitee.copy()
        elif sat_frame_count_AI == 4:
            state['img_sat_buf4_AI'] = image_traitee.copy()
        elif sat_frame_count_AI == 5:
            state['flag_img_sat_buf5_AI'] = True
            state['flag_first_sat_pass_AI'] = False
            
    # Process when buffer is full
    if state['flag_img_sat_buf5_AI']:
        img_sat_buf5_AI = image_traitee.copy()
        state['img_sat_buf5_AI'] = img_sat_buf5_AI
        
        # Convert to grayscale if needed
        if flag_IsColor:
            imggrey22 = cv2.cvtColor(img_sat_buf5_AI, cv2.COLOR_BGR2GRAY)
            imggrey12 = cv2.cvtColor(img_sat_buf1_AI, cv2.COLOR_BGR2GRAY)
        else:
            imggrey22 = img_sat_buf5_AI
            imggrey12 = img_sat_buf1_AI
            
        height, width = imggrey22.shape
        
        # Compute difference between first and last frame
        diff = cv2.subtract(imggrey22, imggrey12)
        seuilb = np.percentile(diff, 99) + 30
        diff[0:90, 0:width] = 0  # Mask top region
        ret, thresh = cv2.threshold(diff, seuilb, 255, cv2.THRESH_BINARY)
        
        # Shift buffer (rolling window)
        state['img_sat_buf1_AI'] = img_sat_buf2_AI.copy()
        state['img_sat_buf2_AI'] = img_sat_buf3_AI.copy()
        state['img_sat_buf3_AI'] = img_sat_buf4_AI.copy()
        state['img_sat_buf4_AI'] = img_sat_buf5_AI.copy()
        
        # GPU processing for noise removal
        with cupy_context:
            height, width = thresh.shape
            Pixel_threshold = 120
            nb_blocksX = (width // nb_ThreadsX) + 1
            nb_blocksY = (height // nb_ThreadsY) + 1
            res_r = cp.zeros_like(thresh, dtype=cp.uint8)
            img = cp.asarray(thresh, dtype=cp.uint8)
            Dead_Pixels_Remove_Mono_GPU(
                (nb_blocksX, nb_blocksY),
                (nb_ThreadsX, nb_ThreadsY),
                (res_r, img, np.intc(width), np.intc(height), np.intc(Pixel_threshold))
            )
            thresh = res_r.copy()
            thresh_blur = gaussianblur_mono(thresh, 1)
            thresh = thresh_blur.get()
            image_sat = cv2.merge((thresh, thresh, thresh))
            result = image_sat
            
        return True, result, state
    else:
        return False, image_traitee, state


def satellites_tracking(
    image_traitee: np.ndarray,
    img_sat_buf1: np.ndarray,
    img_sat_buf2: np.ndarray,
    img_sat_buf3: np.ndarray,
    img_sat_buf4: np.ndarray,
    img_sat_buf5: Optional[np.ndarray],
    sat_frame_count: int,
    sat_frame_target: int,
    flag_first_sat_pass: bool,
    flag_IsColor: bool,
    sat_x: np.ndarray,
    sat_y: np.ndarray,
    sat_s: np.ndarray,
    sat_id: np.ndarray,
    sat_old_x: np.ndarray,
    sat_old_y: np.ndarray,
    sat_old_id: np.ndarray,
    sat_old_dx: np.ndarray,
    sat_old_dy: np.ndarray,
    sat_speed: np.ndarray,
    nb_trace_sat: int,
    max_sat: int
) -> Tuple[int, dict, np.ndarray, np.ndarray]:
    """
    Traditional blob-based satellite tracking with trajectory prediction.
    
    Detects moving satellites using frame differencing and SimpleBlobDetector,
    then tracks their trajectories across frames with speed and direction prediction.
    
    Args:
        image_traitee: Current processed image
        img_sat_buf1 through img_sat_buf5: 5-frame buffer
        sat_frame_count: Current frame count
        sat_frame_target: Target frames (typically 5)
        flag_first_sat_pass: First pass flag
        flag_IsColor: Color/grayscale flag
        sat_x, sat_y, sat_s: Current satellite positions and sizes
        sat_id: Satellite IDs
        sat_old_x, sat_old_y: Previous positions
        sat_old_id: Previous IDs
        sat_old_dx, sat_old_dy: Previous deltas for velocity
        sat_speed: Satellite speeds
        nb_trace_sat: Number of tracked satellites
        max_sat: Maximum satellites to track
        
    Returns:
        Tuple of (nb_sat, state, calque_satellites, calque_direction_satellites):
            - nb_sat: Number of detected satellites
            - state: Updated tracking state dictionary
            - calque_satellites: Overlay with satellite tracks
            - calque_direction_satellites: Overlay with predicted directions
            
    Example:
        >>> nb, state, tracks, directions = satellites_tracking(
        ...     frame, b1, b2, b3, b4, b5, count, 5, True, True,
        ...     x_arr, y_arr, s_arr, id_arr, old_x, old_y, old_id,
        ...     old_dx, old_dy, speed, 0, 100
        ... )
    """
    # Initialize state
    state = {
        'img_sat_buf1': img_sat_buf1,
        'img_sat_buf2': img_sat_buf2,
        'img_sat_buf3': img_sat_buf3,
        'img_sat_buf4': img_sat_buf4,
        'img_sat_buf5': img_sat_buf5,
        'sat_frame_count': sat_frame_count,
        'flag_first_sat_pass': flag_first_sat_pass,
        'sat_x': sat_x,
        'sat_y': sat_y,
        'sat_s': sat_s,
        'sat_id': sat_id,
        'sat_old_x': sat_old_x,
        'sat_old_y': sat_old_y,
        'sat_old_id': sat_old_id,
        'sat_old_dx': sat_old_dx,
        'sat_old_dy': sat_old_dy,
        'sat_speed': sat_speed,
        'nb_trace_sat': nb_trace_sat,
        'flag_img_sat_buf5': False
    }
    
    calque_satellites = np.zeros_like(image_traitee)
    calque_direction_satellites = np.zeros_like(image_traitee)
    
    # Fill buffer
    if sat_frame_count < sat_frame_target and flag_first_sat_pass:
        sat_frame_count += 1
        state['sat_frame_count'] = sat_frame_count
        
        if sat_frame_count == 1:
            state['img_sat_buf1'] = image_traitee.copy()
        elif sat_frame_count == 2:
            state['img_sat_buf2'] = image_traitee.copy()
        elif sat_frame_count == 3:
            state['img_sat_buf3'] = image_traitee.copy()
        elif sat_frame_count == 4:
            state['img_sat_buf4'] = image_traitee.copy()
        elif sat_frame_count == 5:
            state['flag_img_sat_buf5'] = True
            
    # Process when buffer is full
    if state['flag_img_sat_buf5']:
        img_sat_buf5 = image_traitee.copy()
        state['img_sat_buf5'] = img_sat_buf5
        
        # Convert to grayscale
        if flag_IsColor:
            imggrey2 = cv2.cvtColor(img_sat_buf5, cv2.COLOR_BGR2GRAY)
            imggrey1 = cv2.cvtColor(img_sat_buf1, cv2.COLOR_BGR2GRAY)
        else:
            imggrey2 = img_sat_buf5
            imggrey1 = img_sat_buf1
            
        correspondance = np.zeros(10000, dtype=int)
        height, width = imggrey2.shape
        
        # Frame differencing
        diff = cv2.subtract(imggrey2, imggrey1)
        seuilb = np.percentile(diff, 99) + 30
        diff[0:90, 0:width] = 0
        ret, thresh = cv2.threshold(diff, seuilb, 255, cv2.THRESH_BINARY)
        image_sat = cv2.merge((thresh, thresh, thresh))
        
        # Blob detection
        seuil_min_blob_sat = 20
        params_sat = cv2.SimpleBlobDetector_Params()
        params_sat.minThreshold = seuil_min_blob_sat
        params_sat.maxThreshold = 255
        params_sat.thresholdStep = 10
        params_sat.filterByColor = True
        params_sat.blobColor = 255
        params_sat.minDistBetweenBlobs = 2
        params_sat.filterByArea = True
        params_sat.minArea = 4
        params_sat.maxArea = 2000
        params_sat.minRepeatability = 2
        params_sat.filterByCircularity = False
        params_sat.filterByConvexity = False
        params_sat.filterByInertia = False
        detector_sat = cv2.SimpleBlobDetector_create(params_sat)
        keypoints_sat = detector_sat.detect(image_sat)
        
        # Shift buffer
        state['img_sat_buf1'] = img_sat_buf2.copy()
        state['img_sat_buf2'] = img_sat_buf3.copy()
        state['img_sat_buf3'] = img_sat_buf4.copy()
        state['img_sat_buf4'] = img_sat_buf5.copy()
        state['flag_first_sat_pass'] = False
        
        flag_sat = True
    else:
        flag_sat = False
        
    nb_sat = -1
    
    if flag_sat:
        if not flag_first_sat_pass:
            # Count satellites
            for kp_sat in keypoints_sat:
                nb_sat += 1
                
            if 0 <= nb_sat < max_sat:
                nb_sat = -1
                # Extract positions
                for kp_sat in keypoints_sat:
                    nb_sat += 1
                    sat_x[nb_sat] = int(kp_sat.pt[0])
                    sat_y[nb_sat] = int(kp_sat.pt[1])
                    sat_s[nb_sat] = int(kp_sat.size * 2)
                    
                # Match with previous detections
                for i in range(nb_sat + 1):
                    dist_min = 100000
                    correspondance[i] = -1
                    for j in range(nb_trace_sat + 1):
                        if sat_old_x[j] > 0:
                            distance = int(math.sqrt(
                                (sat_x[i] - sat_old_x[j]) ** 2 +
                                (sat_y[i] - sat_old_y[j]) ** 2
                            ))
                        else:
                            distance = -1
                        if 0 < distance < dist_min:
                            dist_min = distance
                            correspondance[i] = j
                            sat_id[i] = sat_old_id[correspondance[i]]
                            
                    # New satellite if distance too large
                    if dist_min > 50:
                        correspondance[i] = -1
                        nb_trace_sat += 1
                        sat_id[i] = nb_trace_sat
                        sat_old_x[nb_trace_sat] = sat_x[i]
                        sat_old_y[nb_trace_sat] = sat_y[i]
                        sat_old_id[nb_trace_sat] = nb_trace_sat
                        
                state['nb_trace_sat'] = nb_trace_sat
                
                # Clean up inactive traces
                for j in range(nb_trace_sat + 1):
                    flag_active_trace = False
                    for i in range(nb_sat + 1):
                        if sat_old_id[j] == sat_id[i]:
                            flag_active_trace = True
                    if not flag_active_trace:
                        sat_old_x[j] = -1
                        sat_old_y[j] = -1
                        sat_old_id[j] = -1
                        
                # Draw tracks and predict direction
                for i in range(nb_sat + 1):
                    if correspondance[i] >= 0 and sat_old_x[correspondance[i]] > 0:
                        start_point = (sat_old_x[correspondance[i]], sat_old_y[correspondance[i]])
                        end_point = (sat_x[i], sat_y[i])
                        cv2.line(calque_satellites, start_point, end_point, (0, 255, 0), 1)
                        
                        # Calculate velocity and direction
                        delta_x = (sat_x[i] - sat_old_x[correspondance[i]]) * 7
                        delta_x = (delta_x + sat_old_dx[correspondance[i]]) // 2
                        delta_y = (sat_y[i] - sat_old_y[correspondance[i]]) * 7
                        delta_y = (delta_y + sat_old_dy[correspondance[i]]) // 2
                        sat_speed[i] = math.sqrt(delta_x * delta_x + delta_y * delta_y)
                        direction = (sat_x[i] + delta_x, sat_y[i] + delta_y)
                        cv2.line(calque_direction_satellites, end_point, direction, (255, 255, 0), 1)
                        
                        # Update old positions
                        sat_old_x[correspondance[i]] = sat_x[i]
                        sat_old_y[correspondance[i]] = sat_y[i]
                        sat_old_dx[correspondance[i]] = delta_x
                        sat_old_dy[correspondance[i]] = delta_y
                        sat_old_id[correspondance[i]] = sat_id[i]
                    else:
                        if correspondance[i] >= 0:
                            sat_old_x[correspondance[i]] = -1
                            sat_old_y[correspondance[i]] = -1
                            sat_old_id[correspondance[i]] = -1
                            
            # Reset if too many satellites
            if nb_sat >= max_sat:
                # Would call raz_tracking() here
                nb_sat = -1
                
        # First pass initialization
        if flag_first_sat_pass:
            for kp_sat in keypoints_sat:
                nb_sat += 1
            if nb_sat >= 0:
                nb_sat = -1
                for kp_sat in keypoints_sat:
                    nb_sat += 1
                    sat_x[nb_sat] = int(kp_sat.pt[0])
                    sat_y[nb_sat] = int(kp_sat.pt[1])
                    sat_s[nb_sat] = int(kp_sat.size * 2)
                    sat_id[nb_sat] = nb_sat
                    sat_old_x[nb_sat] = sat_x[nb_sat]
                    sat_old_y[nb_sat] = sat_y[nb_sat]
                    sat_old_id[nb_sat] = nb_sat
                nb_trace_sat = nb_sat
                state['nb_trace_sat'] = nb_trace_sat
                state['flag_first_sat_pass'] = False
                
    # Update state arrays
    state['sat_x'] = sat_x
    state['sat_y'] = sat_y
    state['sat_s'] = sat_s
    state['sat_id'] = sat_id
    state['sat_old_x'] = sat_old_x
    state['sat_old_y'] = sat_old_y
    state['sat_old_id'] = sat_old_id
    state['sat_old_dx'] = sat_old_dx
    state['sat_old_dy'] = sat_old_dy
    state['sat_speed'] = sat_speed
    
    return nb_sat, state, calque_satellites, calque_direction_satellites


def remove_satellites(
    image_traitee: np.ndarray,
    nb_sat: int,
    sat_x: np.ndarray,
    sat_y: np.ndarray,
    sat_s: np.ndarray,
    flag_IsColor: bool
) -> np.ndarray:
    """
    Remove detected satellites from image by replacing with local background.
    
    Replaces satellite pixels with values sampled from surrounding background
    based on local percentile statistics with random noise.
    
    Args:
        image_traitee: Image to process
        nb_sat: Number of satellites detected
        sat_x, sat_y: Satellite positions
        sat_s: Satellite sizes
        flag_IsColor: Color/grayscale flag
        
    Returns:
        Image with satellites removed
        
    Example:
        >>> clean_image = remove_satellites(frame, 5, x_arr, y_arr, s_arr, True)
    """
    for i in range(nb_sat + 1):
        try:
            y1 = sat_y[i] - sat_s[i]
            y2 = sat_y[i] + sat_s[i]
            x1 = sat_x[i] - sat_s[i]
            x2 = sat_x[i] + sat_s[i]
            mask_sat = image_traitee[y1:y2, x1:x2]
            
            if flag_IsColor:
                seuilb = abs(np.percentile(mask_sat[:, :, 0], 70))
                seuilg = abs(np.percentile(mask_sat[:, :, 1], 70))
                seuilr = abs(np.percentile(mask_sat[:, :, 2], 70))
                axex = range(x1, x2)
                axey = range(y1, y2)
                for i in axex:
                    for j in axey:
                        if image_traitee[j, i, 0] > seuilb:
                            image_traitee[j, i, 0] = abs(seuilb + random.randrange(0, 40) - 30)
                        if image_traitee[j, i, 1] > seuilg:
                            image_traitee[j, i, 1] = abs(seuilg + random.randrange(0, 40) - 30)
                        if image_traitee[j, i, 2] > seuilr:
                            image_traitee[j, i, 2] = abs(seuilr + random.randrange(0, 40) - 30)
            else:
                seuilb = abs(np.percentile(mask_sat[:, :], 70))
                axex = range(x1, x2)
                axey = range(y1, y2)
                for i in axex:
                    for j in axey:
                        if image_traitee[j, i] > seuilb:
                            image_traitee[j, i] = abs(seuilb + random.randrange(0, 40) - 30)
        except:
            pass
            
    return image_traitee


def stars_detection(
    image_traitee: np.ndarray,
    flag_IsColor: bool,
    draw: bool = True
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect stars using SimpleBlobDetector with adaptive thresholding.
    
    Finds bright point sources (stars) in the image using blob detection
    with automatic threshold calculation based on image percentiles.
    
    Args:
        image_traitee: Input image
        flag_IsColor: Color/grayscale flag
        draw: Whether to draw detection circles (default: True)
        
    Returns:
        Tuple of (nb_stars, calque_stars, stars_x, stars_y, stars_s):
            - nb_stars: Number of stars detected
            - calque_stars: Overlay with star circles
            - stars_x, stars_y: Star positions
            - stars_s: Star sizes
            
    Example:
        >>> n, overlay, x, y, s = stars_detection(frame, True, draw=True)
        >>> print(f"Found {n + 1} stars")
    """
    stars_x = np.zeros(10000, dtype=int)
    stars_y = np.zeros(10000, dtype=int)
    stars_s = np.zeros(10000, dtype=int)
    
    if flag_IsColor:
        calque_stars = np.zeros_like(image_traitee)
        seuilb = np.percentile(image_traitee[:, :, 0], 90)
        seuilg = np.percentile(image_traitee[:, :, 1], 90)
        seuilr = np.percentile(image_traitee[:, :, 2], 90)
        seuil_min_blob = max(seuilb, seuilg, seuilr) + 15
        height, width, layers = image_traitee.shape
    else:
        calque_stars = np.zeros_like(image_traitee)
        seuilb = np.percentile(image_traitee, 90)
        seuil_min_blob = seuilb + 15
        height, width = image_traitee.shape
        
    image_stars = image_traitee.copy()
    image_stars[0:50, 0:width] = 0  # Mask top region
    
    if seuil_min_blob > 160:
        seuil_min_blob = 160
        
    # Configure blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = seuil_min_blob
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.filterByColor = False
    params.blobColor = 255
    params.minDistBetweenBlobs = 3
    params.filterByArea = True
    params.minArea = 2
    params.maxArea = 1000
    params.minRepeatability = 2
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image_stars)
    
    nb_stars = -1
    for kp in keypoints:
        nb_stars += 1
        stars_x[nb_stars] = int(kp.pt[0])
        stars_y[nb_stars] = int(kp.pt[1])
        stars_s[nb_stars] = int(kp.size)
        if draw:
            if flag_IsColor:
                cv2.circle(calque_stars, (int(kp.pt[0]), int(kp.pt[1])),
                          int(kp.size * 1.5), (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(calque_stars, (int(kp.pt[0]), int(kp.pt[1])),
                          int(kp.size * 1.5), (255, 255, 255), 1, cv2.LINE_AA)
                          
    return nb_stars, calque_stars, stars_x, stars_y, stars_s


def draw_satellite(
    image_reconstructed: np.ndarray,
    x: int,
    y: int,
    flag_IsColor: bool
) -> None:
    """
    Draw satellite marker on image (green circle with crosshair).
    
    Args:
        image_reconstructed: Image to draw on (modified in-place)
        x, y: Satellite position
        flag_IsColor: Color/grayscale flag
        
    Example:
        >>> draw_satellite(img, 100, 200, True)
    """
    centercircle = (x, y)
    start = (x - 10, y - 10)
    stop = (x + 10, y + 10)
    
    if flag_IsColor:
        cv2.line(image_reconstructed, start, stop, (0, 255, 0), 5, cv2.LINE_AA)
        cv2.circle(image_reconstructed, centercircle, 7, (0, 255, 0), -1, cv2.LINE_AA)
    else:
        cv2.line(image_reconstructed, start, stop, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.circle(image_reconstructed, centercircle, 7, (255, 255, 255), -1, cv2.LINE_AA)


def draw_star(
    image_traitee: np.ndarray,
    calque_reconstruct: np.ndarray,
    x: int,
    y: int,
    s: int,
    flag_IsColor: bool
) -> None:
    """
    Draw enhanced star with gradient fade effect.
    
    Creates a multi-layered circular gradient mimicking star diffraction.
    Color is sampled from the original star position.
    
    Args:
        image_traitee: Source image for color sampling
        calque_reconstruct: Overlay to draw on (modified in-place)
        x, y: Star position
        s: Star size
        flag_IsColor: Color/grayscale flag
        
    Example:
        >>> draw_star(frame, overlay, 100, 200, 5, True)
    """
    if flag_IsColor:
        red = image_traitee[y, x, 0]
        green = image_traitee[y, x, 1]
        blue = image_traitee[y, x, 2]
    else:
        red = image_traitee[y, x]
        green = red
        blue = red
        
    rayon = 1
    s = int(s / 1.7)
    for i in range(s):
        centercircle = (x, y)
        red = int(red / (s / (s - (0.5 * i))))
        green = int(green / (s / (s - (0.3 * i))))
        blue = int(blue / (s / (s - (0.3 * i))))
        cv2.circle(calque_reconstruct, centercircle, rayon, (red, green, blue), 1, cv2.LINE_AA)
        rayon += 1


def reconstruction_image(
    image_traitee: np.ndarray,
    flag_IsColor: bool,
    flag_TRKSAT: int,
    nb_sat: int,
    sat_x: np.ndarray,
    sat_y: np.ndarray,
    sat_s: np.ndarray
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct image with enhanced stars and optionally satellites.
    
    Creates an aesthetically enhanced version of the image by:
    1. Applying gaussian blur to base image
    2. Detecting and enhancing stars with gradient effects
    3. Optionally tracking and marking satellites
    
    Args:
        image_traitee: Input image
        flag_IsColor: Color/grayscale flag
        flag_TRKSAT: Satellite tracking flag (1=enabled)
        nb_sat: Number of satellites (for tracking)
        sat_x, sat_y: Satellite positions
        sat_s: Satellite sizes
        
    Returns:
        Tuple of (image_reconstructed, nb_stars, stars_x, stars_y, stars_s):
            - image_reconstructed: Enhanced image
            - nb_stars: Number of detected stars
            - stars_x, stars_y: Star positions
            - stars_s: Star sizes
            
    Example:
        >>> enhanced, n_stars, x, y, s = reconstruction_image(
        ...     frame, True, 1, 3, sat_x, sat_y, sat_s
        ... )
    """
    calque_reconstruct = np.zeros_like(image_traitee)
    image_reconstructed = cv2.GaussianBlur(image_traitee, (7, 7), 0)
    
    # Detect stars
    nb_stars, calque_stars, stars_x, stars_y, stars_s = stars_detection(
        image_traitee, flag_IsColor, draw=False
    )
    
    # Draw enhanced stars
    for i in range(nb_stars + 1):
        draw_star(image_traitee, calque_reconstruct, stars_x[i], stars_y[i],
                 stars_s[i], flag_IsColor)
                 
    # Blend with star layer
    image_reconstructed = cv2.addWeighted(image_reconstructed, 1, calque_reconstruct, 1, 0)
    
    # Optionally add satellites
    if flag_TRKSAT == 1:
        for i in range(nb_sat + 1):
            draw_satellite(image_reconstructed, sat_x[i], sat_y[i], flag_IsColor)
            
    return image_reconstructed, nb_stars, stars_x, stars_y, stars_s
