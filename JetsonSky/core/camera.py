"""
CameraController - High-performance camera management with CUDA/CUPY support

This module provides a clean OOP interface for ZWO ASI camera control while maintaining
the performance characteristics of the original implementation.

Author: Refactored from JetsonSky monolithic code
Performance: Optimized for real-time video acquisition with GPU acceleration
"""

import time
import numpy as np
from threading import Thread
from typing import Optional, Tuple, Dict, Any

try:
    import cupy as cp
    FLAG_CUPY = True
except ImportError:
    FLAG_CUPY = False
    print("CuPy not available - camera performance will be degraded")

try:
    import zwoasi_cupy as asi
    FLAG_ASI = True
except ImportError:
    FLAG_ASI = False
    print("ZWO ASI SDK not available")

from .camera_models import get_camera_config, is_camera_supported


class CameraAcquisitionThread(Thread):
    """
    High-performance camera acquisition thread with CUDA context support.

    This thread continuously captures frames from the camera with minimal overhead.
    Performance-critical: Runs in tight loop with CUDA context for zero-copy operations.
    """

    def __init__(self, controller: 'CameraController', cupy_context):
        """
        Initialize acquisition thread.

        Args:
            controller: Parent CameraController instance
            cupy_context: CUDA stream context for GPU operations
        """
        super().__init__(daemon=True)
        self.controller = controller
        self.cupy_context = cupy_context
        self.running = False
        self.active = False

    def run(self):
        """Main acquisition loop - performance critical."""
        self.running = True

        with self.cupy_context:
            while self.running:
                if self.active:
                    try:
                        # Capture frame with minimal overhead
                        if self.controller.use_16bit:
                            if self.controller.my_os == "win32":
                                frame = self.controller.camera.capture_video_frame_RAW16_CUPY(
                                    filename=None,
                                    timeout=self.controller.timeout
                                )
                            else:
                                frame = self.controller.camera.capture_video_frame_RAW16_NUMPY(
                                    filename=None,
                                    timeout=self.controller.timeout
                                )
                        else:
                            if self.controller.my_os == "win32":
                                frame = self.controller.camera.capture_video_frame_RAW8_CUPY(
                                    filename=None,
                                    timeout=self.controller.timeout
                                )
                            else:
                                frame = self.controller.camera.capture_video_frame_RAW8_NUMPY(
                                    filename=None,
                                    timeout=self.controller.timeout
                                )

                        # Update controller state (thread-safe via flags)
                        self.controller._raw_frame = frame.copy()
                        self.controller._frame_captured = True
                        self.controller._new_frame_available = True

                    except Exception as error:
                        # Handle capture errors gracefully
                        self.controller._frame_captured = False
                        self.controller._new_frame_available = False
                        self.controller.error_count += 1

                        print(f"Camera capture error: {error}")
                        print(f"Error count: {self.controller.error_count}")

                        # Attempt recovery
                        try:
                            self.controller.camera.stop_video_capture()
                            self.controller.camera.stop_exposure()
                            time.sleep(0.5)
                            self.controller.camera.start_video_capture()
                            time.sleep(0.5)
                        except:
                            pass
                else:
                    time.sleep(0.5)

    def stop(self):
        """Stop acquisition thread gracefully."""
        self.running = False
        self.active = False


class CameraController:
    """
    High-performance camera controller for ZWO ASI cameras.

    This class encapsulates all camera-related operations while maintaining
    the performance characteristics of the original monolithic implementation.

    Features:
    - Automatic camera detection and configuration
    - High-speed video capture with CUDA/CUPY support
    - Thread-safe frame acquisition
    - Automatic error recovery
    - Support for 8-bit and 16-bit capture modes

    Performance:
    - Zero-copy GPU operations on Windows (CUPY)
    - Minimal overhead on Linux (NumPy with fast GPU transfer)
    - Tight acquisition loop for maximum frame rate
    """

    def __init__(self, lib_path: str, my_os: str, cupy_context, usb_bandwidth: int = 70):
        """
        Initialize camera controller.

        Args:
            lib_path: Path to ZWO ASI SDK library
            my_os: Operating system ("win32" or "linux")
            cupy_context: CUDA stream context for GPU operations
            usb_bandwidth: USB bandwidth setting (40-100)
        """
        self.lib_path = lib_path
        self.my_os = my_os
        self.cupy_context = cupy_context
        self.usb_bandwidth = usb_bandwidth

        # Camera state
        self.camera = None
        self.camera_config = None
        self.camera_name = None
        self.camera_id = 0
        self.is_initialized = False
        self.is_color = False

        # Acquisition state
        self.acquisition_thread: Optional[CameraAcquisitionThread] = None
        self._raw_frame = None
        self._frame_captured = False
        self._new_frame_available = False
        self.error_count = 0

        # Capture settings
        self.use_16bit = True
        self.timeout = 1500  # ms
        self.exposure = 1000  # µs
        self.gain = 100
        self.bin_mode = 1
        self.resolution_x = 3096
        self.resolution_y = 2080

    def initialize(self) -> bool:
        """
        Initialize camera hardware and load configuration.

        Returns:
            bool: True if camera initialized successfully

        Performance: Optimized initialization with minimal overhead.
        """
        if not FLAG_ASI:
            print("ZWO ASI SDK not available")
            return False

        try:
            # Initialize ASI SDK
            asi.init(self.lib_path)
            time.sleep(0.5)

            # Detect cameras
            num_cameras = asi.get_num_cameras()
            if num_cameras == 0:
                print('No cameras found - Video treatment mode activated')
                return False

            # List available cameras
            cameras_found = asi.list_cameras()
            if num_cameras == 1:
                self.camera_id = 0
                print(f'Found one camera: {cameras_found[0]}')
            else:
                print(f'Found {num_cameras} cameras')
                for n in range(num_cameras):
                    print(f'    {n}: {cameras_found[n]}')
                self.camera_id = 0
                print(f'Using #{self.camera_id}: {cameras_found[self.camera_id]}')

            self.camera_name = cameras_found[self.camera_id]

            # Check if camera is supported
            if not is_camera_supported(self.camera_name):
                print(f'WARNING: {self.camera_name} may not be fully supported')

            # Load camera configuration from registry
            self.camera_config = get_camera_config(self.camera_name)
            if self.camera_config:
                self.resolution_x = self.camera_config.resolution_x
                self.resolution_y = self.camera_config.resolution_y
                self.is_color = self.camera_config.is_color
                print(f'Camera config loaded: {self.resolution_x}x{self.resolution_y}, ' +
                      f'Color: {self.is_color}')

            # Open camera
            self.camera = asi.Camera(self.camera_id)
            camera_info = self.camera.get_camera_property()
            print(f'Camera info: {camera_info}')

            # Configure camera
            self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, self.usb_bandwidth)
            self.camera.set_control_value(asi.ASI_EXPOSURE, self.exposure)
            self.camera.set_control_value(asi.ASI_GAIN, self.gain)
            self.camera.set_control_value(asi.ASI_WB_B, 99)
            self.camera.set_control_value(asi.ASI_WB_R, 52)
            self.camera.set_control_value(asi.ASI_GAMMA, 50)
            self.camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
            self.camera.set_control_value(asi.ASI_FLIP, 0)

            # Set capture format
            if self.use_16bit:
                self.camera.set_image_type(asi.ASI_IMG_RAW16)
            else:
                self.camera.set_image_type(asi.ASI_IMG_RAW8)

            # Set ROI (full sensor)
            self.camera.set_roi(
                start_x=0,
                start_y=0,
                width=self.resolution_x,
                height=self.resolution_y,
                bins=self.bin_mode,
                image_type=asi.ASI_IMG_RAW16 if self.use_16bit else asi.ASI_IMG_RAW8
            )

            self.is_initialized = True
            print(f'Camera initialized successfully: {self.camera_name}')
            return True

        except Exception as error:
            print(f'Camera initialization error: {error}')
            self.is_initialized = False
            return False

    def start_acquisition(self) -> bool:
        """
        Start high-speed video capture thread.

        Returns:
            bool: True if acquisition started successfully

        Performance: Launches dedicated thread with CUDA context for maximum throughput.
        """
        if not self.is_initialized:
            print("Camera not initialized")
            return False

        if self.acquisition_thread and self.acquisition_thread.running:
            print("Acquisition already running")
            return False

        try:
            # Start video capture mode on camera
            self.camera.start_video_capture()
            time.sleep(0.1)

            # Create and start acquisition thread
            self.acquisition_thread = CameraAcquisitionThread(self, self.cupy_context)
            self.acquisition_thread.active = True
            self.acquisition_thread.start()

            print("Camera acquisition started")
            return True

        except Exception as error:
            print(f'Failed to start acquisition: {error}')
            return False

    def stop_acquisition(self):
        """Stop video capture thread gracefully."""
        if self.acquisition_thread:
            self.acquisition_thread.stop()
            time.sleep(1)
            self.acquisition_thread = None

        if self.camera and self.is_initialized:
            try:
                self.camera.stop_video_capture()
                self.camera.stop_exposure()
            except:
                pass

        print("Camera acquisition stopped")

    def get_frame(self) -> Optional[Tuple[Any, bool]]:
        """
        Get latest captured frame (non-blocking).

        Returns:
            Tuple of (frame, is_new) or (None, False) if no frame available

        Performance: Zero-copy access to frame buffer.
        """
        if self._new_frame_available and self._frame_captured:
            self._new_frame_available = False
            return self._raw_frame, True
        return None, False

    def set_exposure(self, exposure_us: int):
        """Set camera exposure in microseconds."""
        self.exposure = exposure_us
        if self.is_initialized:
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
            self.timeout = 1 + (exposure_us // 1000) + 500

    def set_gain(self, gain: int):
        """Set camera gain."""
        self.gain = gain
        if self.is_initialized:
            self.camera.set_control_value(asi.ASI_GAIN, gain)

    def set_usb_bandwidth(self, bandwidth: int):
        """Set USB bandwidth (40-100)."""
        self.usb_bandwidth = bandwidth
        if self.is_initialized:
            self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, bandwidth)

    def set_resolution(self, width: int, height: int, bin_mode: int = 1):
        """
        Set camera resolution and binning mode.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            bin_mode: Binning mode (1 or 2)
        """
        self.resolution_x = width
        self.resolution_y = height
        self.bin_mode = bin_mode

        if self.is_initialized:
            # Must stop acquisition to change ROI
            was_running = self.acquisition_thread and self.acquisition_thread.active
            if was_running:
                self.stop_acquisition()

            self.camera.set_roi(
                start_x=0,
                start_y=0,
                width=width,
                height=height,
                bins=bin_mode,
                image_type=asi.ASI_IMG_RAW16 if self.use_16bit else asi.ASI_IMG_RAW8
            )

            if was_running:
                self.start_acquisition()

    def close(self):
        """Cleanup camera resources."""
        self.stop_acquisition()

        if self.camera and self.is_initialized:
            try:
                self.camera.close()
            except:
                pass

        self.is_initialized = False
        print("Camera closed")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.close()
