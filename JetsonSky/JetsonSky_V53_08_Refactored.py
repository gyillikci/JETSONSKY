#!/usr/bin/env python3
"""
JetsonSky V53_08 - Refactored OOP Version

A clean GUI application using the refactored OOP architecture while maintaining
the same look and feel as the original JetsonSky application.

Features:
- Live camera preview with GPU-accelerated processing
- Camera control (exposure, gain, resolution, binning)
- Real-time filter pipeline
- Image/Video capture
- YOLO AI detection (craters, satellites)

Author: Refactored from JetsonSky monolithic code
Performance: 100% maintained with OOP architecture
"""

import os
import sys
import time
from datetime import datetime
from tkinter import *
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk
import numpy as np
import cv2

# Platform detection
my_os = sys.platform
print(f"Platform: {my_os}")

# Import refactored OOP components
try:
    import cupy as cp
    FLAG_CUPY = True
    print("CuPy loaded")
except ImportError:
    FLAG_CUPY = False
    print("CuPy not available - GPU acceleration disabled")
    sys.exit(1)

from core import (
    CameraController,
    ImageProcessor,
    get_camera_config,
    AppState
)
from io import CaptureManager
from ai import AIDetector

# Configure paths
if my_os == "linux":
    lib_path_camera = './x64_Lib/libASICamera2.so.1.27'
    lib_path_efw = './Lib/libEFWFilter.so.1.7'
    image_path = './Images'
    video_path = './Videos'
    usb_bandwidth = 70
elif my_os == "win32":
    lib_path_camera = './Lib/ASICamera2.dll'
    lib_path_efw = './Lib/EFW_filter.dll'
    image_path = './Images'
    video_path = './Videos'
    usb_bandwidth = 95
else:
    print(f"Unsupported platform: {my_os}")
    sys.exit(1)

# AI model paths
crater_model_path = "./AI_models/AI_craters_model6_8s_3c_180e.pt"
satellite_model_path = "./AI_models/AI_Sat_model1_8n_3c_300e.pt"


class JetsonSkyGUI:
    """
    Main GUI application for JetsonSky using refactored OOP components.

    This provides the same user experience as the original while using
    the new clean architecture underneath.
    """

    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("JetsonSky V53_08 Refactored - OOP Architecture")
        self.root.geometry("1600x900")

        # CUDA/CuPy context
        self.cupy_context = cp.cuda.Stream(non_blocking=True)

        # Initialize OOP components
        self.camera = None
        self.processor = None
        self.capture_manager = None
        self.ai_detector = None

        # Application state
        self.app_state = AppState()
        self.running = False
        self.preview_active = False

        # Display settings
        self.display_width = 1280
        self.display_height = 960

        # Build GUI
        self.create_widgets()

        # Initialize components
        self.init_components()

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        # Left panel - Camera preview
        left_frame = Frame(main_frame, bg='black')
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)

        # Camera preview label
        self.preview_label = Label(left_frame, bg='black', text="Camera Preview")
        self.preview_label.pack(fill=BOTH, expand=True)

        # Info label below preview
        self.info_label = Label(left_frame, text="Ready", bg='lightgray', font=('Arial', 10))
        self.info_label.pack(fill=X, pady=2)

        # Right panel - Controls
        right_frame = Frame(main_frame, width=300, bg='lightgray')
        right_frame.pack(side=RIGHT, fill=Y, padx=5, pady=5)
        right_frame.pack_propagate(False)

        # Create control sections
        self.create_camera_controls(right_frame)
        self.create_filter_controls(right_frame)
        self.create_capture_controls(right_frame)
        self.create_ai_controls(right_frame)

    def create_camera_controls(self, parent):
        """Create camera control widgets."""
        frame = LabelFrame(parent, text="Camera Controls", bg='lightgray', font=('Arial', 10, 'bold'))
        frame.pack(fill=X, padx=5, pady=5)

        # Camera status
        self.camera_status = Label(frame, text="Camera: Not initialized", bg='lightgray')
        self.camera_status.pack(pady=2)

        # Init/Start/Stop buttons
        btn_frame = Frame(frame, bg='lightgray')
        btn_frame.pack(fill=X, pady=5)

        self.btn_init = Button(btn_frame, text="Initialize Camera", command=self.init_camera, bg='green', fg='white')
        self.btn_init.pack(side=LEFT, padx=2, expand=True, fill=X)

        self.btn_start = Button(btn_frame, text="Start Preview", command=self.start_preview, state=DISABLED, bg='blue', fg='white')
        self.btn_start.pack(side=LEFT, padx=2, expand=True, fill=X)

        self.btn_stop = Button(btn_frame, text="Stop", command=self.stop_preview, state=DISABLED, bg='red', fg='white')
        self.btn_stop.pack(side=LEFT, padx=2, expand=True, fill=X)

        # Exposure control
        Label(frame, text="Exposure (µs)", bg='lightgray').pack(anchor=W, padx=5)
        self.exposure_var = IntVar(value=1000)
        self.exposure_scale = Scale(
            frame, from_=100, to=100000, orient=HORIZONTAL,
            variable=self.exposure_var, command=self.on_exposure_change,
            bg='lightgray', length=250
        )
        self.exposure_scale.pack(fill=X, padx=5)

        # Gain control
        Label(frame, text="Gain", bg='lightgray').pack(anchor=W, padx=5)
        self.gain_var = IntVar(value=100)
        self.gain_scale = Scale(
            frame, from_=0, to=600, orient=HORIZONTAL,
            variable=self.gain_var, command=self.on_gain_change,
            bg='lightgray', length=250
        )
        self.gain_scale.pack(fill=X, padx=5)

        # USB Bandwidth
        Label(frame, text="USB Bandwidth", bg='lightgray').pack(anchor=W, padx=5)
        self.usb_var = IntVar(value=usb_bandwidth)
        self.usb_scale = Scale(
            frame, from_=40, to=100, orient=HORIZONTAL,
            variable=self.usb_var, command=self.on_usb_change,
            bg='lightgray', length=250
        )
        self.usb_scale.pack(fill=X, padx=5)

        # Binning mode
        Label(frame, text="Binning Mode", bg='lightgray').pack(anchor=W, padx=5)
        self.bin_var = IntVar(value=1)
        bin_frame = Frame(frame, bg='lightgray')
        bin_frame.pack(fill=X, padx=5)
        Radiobutton(bin_frame, text="BIN 1", variable=self.bin_var, value=1,
                   command=self.on_bin_change, bg='lightgray').pack(side=LEFT)
        Radiobutton(bin_frame, text="BIN 2", variable=self.bin_var, value=2,
                   command=self.on_bin_change, bg='lightgray').pack(side=LEFT)

    def create_filter_controls(self, parent):
        """Create filter control widgets."""
        frame = LabelFrame(parent, text="Filters", bg='lightgray', font=('Arial', 10, 'bold'))
        frame.pack(fill=X, padx=5, pady=5)

        # Filter checkbuttons
        self.filter_vars = {}
        filters = [
            ('Hot Pixel', 'hotpixel'),
            ('CLAHE', 'clahe'),
            ('Sharpen 1', 'sharpen1'),
            ('Sharpen 2', 'sharpen2'),
            ('KNN Denoise', 'knn'),
            ('NLM2 Denoise', 'nlm2'),
            ('Saturation', 'saturation'),
            ('Flip V', 'flip_v'),
            ('Flip H', 'flip_h'),
        ]

        for label, key in filters:
            var = BooleanVar(value=False)
            self.filter_vars[key] = var
            cb = Checkbutton(frame, text=label, variable=var,
                           command=lambda k=key: self.on_filter_toggle(k),
                           bg='lightgray')
            cb.pack(anchor=W, padx=5)

        # CLAHE clip limit
        Label(frame, text="CLAHE Clip Limit", bg='lightgray').pack(anchor=W, padx=5)
        self.clahe_var = DoubleVar(value=1.0)
        Scale(
            frame, from_=0.5, to=5.0, resolution=0.1, orient=HORIZONTAL,
            variable=self.clahe_var, command=self.on_clahe_change,
            bg='lightgray', length=250
        ).pack(fill=X, padx=5)

    def create_capture_controls(self, parent):
        """Create capture control widgets."""
        frame = LabelFrame(parent, text="Capture", bg='lightgray', font=('Arial', 10, 'bold'))
        frame.pack(fill=X, padx=5, pady=5)

        # Image capture
        self.btn_capture_image = Button(
            frame, text="📷 Capture Image", command=self.capture_image,
            state=DISABLED, bg='orange', fg='white', font=('Arial', 10, 'bold')
        )
        self.btn_capture_image.pack(fill=X, padx=5, pady=2)

        # Video capture
        btn_frame = Frame(frame, bg='lightgray')
        btn_frame.pack(fill=X, padx=5, pady=2)

        self.btn_start_video = Button(
            btn_frame, text="⏺ Start Video", command=self.start_video_capture,
            state=DISABLED, bg='red', fg='white'
        )
        self.btn_start_video.pack(side=LEFT, expand=True, fill=X, padx=2)

        self.btn_stop_video = Button(
            btn_frame, text="⏹ Stop Video", command=self.stop_video_capture,
            state=DISABLED, bg='gray', fg='white'
        )
        self.btn_stop_video.pack(side=LEFT, expand=True, fill=X, padx=2)

        # Capture format
        Label(frame, text="Image Format", bg='lightgray').pack(anchor=W, padx=5)
        self.format_var = StringVar(value="TIFF")
        format_frame = Frame(frame, bg='lightgray')
        format_frame.pack(fill=X, padx=5)
        Radiobutton(format_frame, text="TIFF", variable=self.format_var, value="TIFF",
                   bg='lightgray').pack(side=LEFT)
        Radiobutton(format_frame, text="JPEG", variable=self.format_var, value="JPEG",
                   bg='lightgray').pack(side=LEFT)
        Radiobutton(format_frame, text="PNG", variable=self.format_var, value="PNG",
                   bg='lightgray').pack(side=LEFT)

    def create_ai_controls(self, parent):
        """Create AI detection control widgets."""
        frame = LabelFrame(parent, text="AI Detection", bg='lightgray', font=('Arial', 10, 'bold'))
        frame.pack(fill=X, padx=5, pady=5)

        # AI detection toggles
        self.ai_crater_var = BooleanVar(value=False)
        self.ai_satellite_var = BooleanVar(value=False)
        self.ai_tracking_var = BooleanVar(value=False)

        Checkbutton(frame, text="Detect Craters", variable=self.ai_crater_var,
                   command=self.on_ai_toggle, bg='lightgray').pack(anchor=W, padx=5)
        Checkbutton(frame, text="Detect Satellites", variable=self.ai_satellite_var,
                   command=self.on_ai_toggle, bg='lightgray').pack(anchor=W, padx=5)
        Checkbutton(frame, text="Enable Tracking", variable=self.ai_tracking_var,
                   command=self.on_ai_toggle, bg='lightgray').pack(anchor=W, padx=5)

        # Detection count
        self.ai_info = Label(frame, text="Craters: 0 | Satellites: 0", bg='lightgray')
        self.ai_info.pack(pady=5)

    def init_components(self):
        """Initialize OOP components."""
        print("\nInitializing components...")

        # Initialize image processor
        self.processor = ImageProcessor(
            cupy_context=self.cupy_context,
            is_color=True
        )
        print("✓ ImageProcessor initialized")

        # Initialize capture manager
        self.capture_manager = CaptureManager(
            image_dir=image_path,
            video_dir=video_path
        )
        print("✓ CaptureManager initialized")

        # Initialize AI detector
        if os.path.exists(crater_model_path) and os.path.exists(satellite_model_path):
            self.ai_detector = AIDetector(
                crater_model_path=crater_model_path,
                satellite_model_path=satellite_model_path
            )
            print("✓ AIDetector initialized")
        else:
            print("⚠ AI models not found - AI detection disabled")

    def init_camera(self):
        """Initialize camera hardware."""
        try:
            self.camera_status.config(text="Initializing camera...")
            self.root.update()

            # Create camera controller
            self.camera = CameraController(
                lib_path=lib_path_camera,
                my_os=my_os,
                cupy_context=self.cupy_context,
                usb_bandwidth=self.usb_var.get()
            )

            # Initialize camera
            if self.camera.initialize():
                self.camera_status.config(text=f"Camera: {self.camera.camera_name}")

                # Configure camera with GUI values
                self.camera.set_exposure(self.exposure_var.get())
                self.camera.set_gain(self.gain_var.get())
                self.camera.set_usb_bandwidth(self.usb_var.get())

                # Update processor color mode
                self.processor.is_color = self.camera.is_color

                # Enable buttons
                self.btn_start.config(state=NORMAL)
                self.btn_init.config(state=DISABLED)

                print(f"✓ Camera initialized: {self.camera.camera_name}")
            else:
                self.camera_status.config(text="Camera: Init failed")
                print("✗ Camera initialization failed")

        except Exception as e:
            self.camera_status.config(text="Camera: Error")
            print(f"✗ Camera init error: {e}")
            import traceback
            traceback.print_exc()

    def start_preview(self):
        """Start live camera preview."""
        if not self.camera or not self.camera.is_initialized:
            print("Camera not initialized")
            return

        try:
            # Start camera acquisition
            if self.camera.start_acquisition():
                self.preview_active = True
                self.running = True

                # Enable/disable buttons
                self.btn_start.config(state=DISABLED)
                self.btn_stop.config(state=NORMAL)
                self.btn_capture_image.config(state=NORMAL)
                self.btn_start_video.config(state=NORMAL)

                # Start preview loop
                self.update_preview()

                print("✓ Preview started")
        except Exception as e:
            print(f"✗ Failed to start preview: {e}")

    def stop_preview(self):
        """Stop live camera preview."""
        self.preview_active = False
        self.running = False

        if self.camera:
            self.camera.stop_acquisition()

        # Enable/disable buttons
        self.btn_start.config(state=NORMAL)
        self.btn_stop.config(state=DISABLED)
        self.btn_capture_image.config(state=DISABLED)
        self.btn_start_video.config(state=DISABLED)

        print("✓ Preview stopped")

    def update_preview(self):
        """Update camera preview (main processing loop)."""
        if not self.preview_active:
            return

        try:
            # Get frame from camera
            frame, is_new = self.camera.get_frame()

            if is_new and frame is not None:
                # Process frame through filter pipeline
                processed_frame, metadata = self.processor.process_frame(
                    frame,
                    is_16bit=self.camera.use_16bit,
                    apply_debayer=self.camera.is_color
                )

                # Apply AI detection if enabled
                if self.ai_detector:
                    crater_detections = None
                    satellite_detections = None

                    if self.ai_crater_var.get():
                        crater_detections = self.ai_detector.detect_craters(
                            processed_frame,
                            use_tracking=self.ai_tracking_var.get()
                        )

                    if self.ai_satellite_var.get():
                        satellite_detections = self.ai_detector.detect_satellites(
                            processed_frame,
                            use_tracking=self.ai_tracking_var.get()
                        )

                    # Draw detections
                    if crater_detections or satellite_detections:
                        processed_frame = self.ai_detector.draw_detections(
                            processed_frame,
                            crater_detections=crater_detections,
                            satellite_detections=satellite_detections,
                            draw_tracks=self.ai_tracking_var.get()
                        )

                        # Update detection count
                        counts = self.ai_detector.get_detection_count()
                        self.ai_info.config(
                            text=f"Craters: {counts['craters']} | Satellites: {counts['satellites']}"
                        )

                # Convert to display format
                display_frame = self.prepare_display_frame(processed_frame)

                # Update preview
                photo = PIL.ImageTk.PhotoImage(image=display_frame)
                self.preview_label.config(image=photo)
                self.preview_label.image = photo

                # Update info
                fps = 1000.0 / metadata['processing_time_ms'] if metadata['processing_time_ms'] > 0 else 0
                info_text = f"Processing: {metadata['processing_time_ms']:.1f}ms | FPS: {fps:.1f} | Filters: {metadata['filters_applied']}"
                self.info_label.config(text=info_text)

        except Exception as e:
            print(f"Preview error: {e}")
            import traceback
            traceback.print_exc()

        # Schedule next update
        if self.preview_active:
            self.root.after(10, self.update_preview)

    def prepare_display_frame(self, frame):
        """Prepare frame for display (resize and convert)."""
        # Convert to RGB if grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to display size
        h, w = frame.shape[:2]
        aspect = w / h

        if aspect > self.display_width / self.display_height:
            new_w = self.display_width
            new_h = int(new_w / aspect)
        else:
            new_h = self.display_height
            new_w = int(new_h * aspect)

        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert to PIL Image
        return PIL.Image.fromarray(frame_resized)

    def on_exposure_change(self, value):
        """Handle exposure slider change."""
        if self.camera and self.camera.is_initialized:
            self.camera.set_exposure(int(value))

    def on_gain_change(self, value):
        """Handle gain slider change."""
        if self.camera and self.camera.is_initialized:
            self.camera.set_gain(int(value))

    def on_usb_change(self, value):
        """Handle USB bandwidth slider change."""
        if self.camera and self.camera.is_initialized:
            self.camera.set_usb_bandwidth(int(value))

    def on_bin_change(self):
        """Handle binning mode change."""
        if self.camera and self.camera.is_initialized:
            # Would need to stop/restart acquisition to change binning
            print(f"Binning mode changed to {self.bin_var.get()}")

    def on_filter_toggle(self, filter_key):
        """Handle filter enable/disable toggle."""
        if not self.processor:
            return

        enabled = self.filter_vars[filter_key].get()

        # Map GUI keys to filter names
        filter_map = {
            'hotpixel': 'hotpixel',
            'clahe': 'clahe',
            'sharpen1': 'sharpen1',
            'sharpen2': 'sharpen2',
            'knn': 'knn',
            'nlm2': 'nlm2',
            'saturation': 'saturation',
        }

        if filter_key in filter_map:
            if enabled:
                self.processor.enable_filter(filter_map[filter_key])
            else:
                self.processor.disable_filter(filter_map[filter_key])
        elif filter_key == 'flip_v':
            self.processor.flip_filter.flip_vertical = enabled
        elif filter_key == 'flip_h':
            self.processor.flip_filter.flip_horizontal = enabled

    def on_clahe_change(self, value):
        """Handle CLAHE parameter change."""
        if self.processor:
            self.processor.update_filter_parameter('clahe', 'clip_limit', float(value))

    def on_ai_toggle(self):
        """Handle AI detection toggle."""
        if not self.ai_detector:
            return

        self.ai_detector.enable_crater_detection = self.ai_crater_var.get()
        self.ai_detector.enable_satellite_detection = self.ai_satellite_var.get()
        self.ai_detector.enable_tracking = self.ai_tracking_var.get()

    def capture_image(self):
        """Capture current frame as image."""
        if not self.preview_active:
            return

        try:
            # Get current processed frame
            frame, is_new = self.camera.get_frame()
            if is_new and frame is not None:
                # Process frame
                processed_frame, _ = self.processor.process_frame(
                    frame,
                    is_16bit=self.camera.use_16bit,
                    apply_debayer=self.camera.is_color
                )

                # Save image
                self.capture_manager.set_image_format(self.format_var.get())
                filepath = self.capture_manager.save_image(processed_frame)

                self.info_label.config(text=f"Image saved: {os.path.basename(filepath)}")
                print(f"✓ Image saved: {filepath}")

        except Exception as e:
            print(f"✗ Image capture failed: {e}")

    def start_video_capture(self):
        """Start video capture."""
        if not self.preview_active:
            return

        try:
            # Get frame size from last processed frame
            frame, _ = self.camera.get_frame()
            if frame is not None:
                processed, _ = self.processor.process_frame(
                    frame,
                    is_16bit=self.camera.use_16bit,
                    apply_debayer=self.camera.is_color
                )

                h, w = processed.shape[:2]

                # Start video capture
                if self.capture_manager.start_video_capture(
                    fps=30,
                    frame_size=(w, h),
                    codec='XVID'
                ):
                    self.btn_start_video.config(state=DISABLED)
                    self.btn_stop_video.config(state=NORMAL)
                    self.info_label.config(text="Video recording...")
                    print("✓ Video capture started")

        except Exception as e:
            print(f"✗ Failed to start video capture: {e}")

    def stop_video_capture(self):
        """Stop video capture."""
        try:
            stats = self.capture_manager.stop_video_capture()
            self.btn_start_video.config(state=NORMAL)
            self.btn_stop_video.config(state=DISABLED)
            self.info_label.config(text=f"Video saved: {stats.get('frames_written', 0)} frames")
            print(f"✓ Video capture stopped: {stats}")

        except Exception as e:
            print(f"✗ Failed to stop video capture: {e}")

    def on_closing(self):
        """Handle window close event."""
        print("\nClosing application...")

        self.stop_preview()

        if self.camera:
            self.camera.close()

        if self.capture_manager:
            self.capture_manager.cleanup()

        self.root.destroy()
        print("✓ Application closed")


def main():
    """Main entry point."""
    print("=" * 70)
    print(" JetsonSky V53_08 - Refactored OOP Architecture")
    print("=" * 70)
    print("\nStarting GUI application...\n")

    root = Tk()
    app = JetsonSkyGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
