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
from tkinter import Canvas, Scrollbar
import PIL.Image
import PIL.ImageTk
import numpy as np
import cv2

# Platform detection
my_os = sys.platform
print(f"Platform: {my_os}")

# Import camera library
import zwoasi_cupy as asi

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
# Import from local io package (avoid built-in io module conflict)
import importlib.util
io_capture_spec = importlib.util.spec_from_file_location(
    "capture_manager",
    os.path.join(os.path.dirname(__file__), "io", "capture_manager.py")
)
if io_capture_spec and io_capture_spec.loader:
    io_capture_module = importlib.util.module_from_spec(io_capture_spec)
    io_capture_spec.loader.exec_module(io_capture_module)
    CaptureManager = io_capture_module.CaptureManager
    print("CaptureManager loaded")
else:
    print("Warning: CaptureManager not available")
    CaptureManager = None
from ai import AIDetector
from utils import ImageStabilizer

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
        self.stabilizer = None  # Will be initialized after camera

        # Application state
        self.app_state = AppState()
        self.running = False
        self.preview_active = False

        # Display settings
        self.display_width = 1920
        self.display_height = 1440

        # Scrollable canvas settings
        self.full_frame = None  # Store full resolution frame
        self.photo_image = None  # Keep reference to PhotoImage
        self.canvas_image_id = None  # Canvas image item id

        # Auto control flags to prevent circular triggers
        self.updating_auto_controls = False

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

        # Create scrollable canvas for preview
        canvas_frame = Frame(left_frame, bg='black')
        canvas_frame.pack(fill=BOTH, expand=True)
        
        # Create canvas with scrollbars
        self.preview_canvas = Canvas(canvas_frame, bg='black', highlightthickness=0)
        h_scrollbar = Scrollbar(canvas_frame, orient=HORIZONTAL, command=self.preview_canvas.xview)
        v_scrollbar = Scrollbar(canvas_frame, orient=VERTICAL, command=self.preview_canvas.yview)
        
        self.preview_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Grid layout for canvas and scrollbars
        self.preview_canvas.grid(row=0, column=0, sticky='nsew')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Bind mouse wheel for scrolling
        self.preview_canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.preview_canvas.bind('<Shift-MouseWheel>', self._on_shift_mousewheel)

        # Info label below preview
        self.info_label = Label(left_frame, text="Ready | Select preview resolution from dropdown", bg='lightgray', font=('Arial', 10))
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
        
        # Auto Exposure checkbox
        self.auto_exposure_var = IntVar(value=0)
        auto_exp_check = Checkbutton(
            frame, text="Auto Exposure", variable=self.auto_exposure_var,
            command=self.on_auto_exposure_toggle, bg='lightgray'
        )
        auto_exp_check.pack(anchor=W, padx=5)

        # Gain control
        Label(frame, text="Gain", bg='lightgray').pack(anchor=W, padx=5)
        self.gain_var = IntVar(value=100)
        self.gain_scale = Scale(
            frame, from_=0, to=600, orient=HORIZONTAL,
            variable=self.gain_var, command=self.on_gain_change,
            bg='lightgray', length=250
        )
        self.gain_scale.pack(fill=X, padx=5)
        
        # Auto Gain checkbox
        self.auto_gain_var = IntVar(value=0)
        auto_gain_check = Checkbutton(
            frame, text="Auto Gain", variable=self.auto_gain_var,
            command=self.on_auto_gain_toggle, bg='lightgray'
        )
        auto_gain_check.pack(anchor=W, padx=5)

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

        # Preview Resolution selector
        Label(frame, text="Preview Resolution", bg='lightgray').pack(anchor=W, padx=5, pady=(10, 0))
        
        from tkinter import ttk
        self.resolution_var = StringVar(value="1920x1440")
        resolution_options = [
            "1280x720 (720p)",
            "1280x960 (4:3)",
            "1600x1200 (UXGA)",
            "1920x1080 (1080p)",
            "1920x1440 (Default)",
            "2560x1440 (1440p)",
            "2560x1920 (4:3)",
            "3840x2160 (4K)",
            "Full Resolution"
        ]
        
        self.resolution_combo = ttk.Combobox(
            frame, 
            textvariable=self.resolution_var,
            values=resolution_options,
            state='readonly',
            width=18
        )
        self.resolution_combo.pack(fill=X, padx=5, pady=2)
        self.resolution_combo.bind('<<ComboboxSelected>>', self.on_resolution_change)

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
            ('Stabilization', 'stabilization'),
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

                # Update processor color mode based on camera config
                is_color = hasattr(self.camera, 'camera_config') and \
                          self.camera.camera_config.bayer_pattern != "MONO"
                self.processor.is_color = is_color

                # Initialize stabilizer with camera resolution
                self.stabilizer = ImageStabilizer(
                    resolution_x=self.camera.resolution_x,
                    resolution_y=self.camera.resolution_y
                )
                print("✓ ImageStabilizer initialized")

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
                # Process frame through filter pipeline first (debayer, etc.)
                processed_frame, metadata = self.processor.process_frame(
                    frame,
                    is_16bit=self.camera.use_16bit,
                    apply_debayer=self.camera.is_color
                )
                
                # Apply stabilization if enabled (after debayering/processing)
                if self.stabilizer and self.stabilizer.enabled and processed_frame is not None:
                    # Determine if color or mono
                    is_color = len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3
                    processed_frame = self.stabilizer.stabilize(processed_frame, is_color=is_color)

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

                # Update preview on canvas
                self.photo_image = PIL.ImageTk.PhotoImage(image=display_frame)
                
                # Create or update canvas image
                if self.canvas_image_id is None:
                    self.canvas_image_id = self.preview_canvas.create_image(
                        0, 0, anchor='nw', image=self.photo_image
                    )
                else:
                    self.preview_canvas.itemconfig(self.canvas_image_id, image=self.photo_image)

                # Update info label with all details
                fps = 1000.0 / metadata['processing_time_ms'] if metadata['processing_time_ms'] > 0 else 0
                
                # Get frame dimensions
                if self.full_frame is not None:
                    orig_h, orig_w = self.full_frame.shape[:2]
                else:
                    orig_h, orig_w = processed_frame.shape[:2]
                
                # Get display dimensions from PhotoImage
                disp_w = display_frame.width
                disp_h = display_frame.height
                
                info_text = (f"Camera: {orig_w}x{orig_h} | Preview: {disp_w}x{disp_h} | "
                            f"{metadata['processing_time_ms']:.1f}ms | FPS: {fps:.1f} | "
                            f"Filters: {metadata['filters_applied']}")
                self.info_label.config(text=info_text)
                
                # Update sliders if auto mode is enabled (read back from camera)
                self.update_auto_controls()

        except Exception as e:
            print(f"Preview error: {e}")
            import traceback
            traceback.print_exc()

        # Schedule next update
        if self.preview_active:
            self.root.after(10, self.update_preview)

    def prepare_display_frame(self, frame):
        """Prepare frame for display - resize to fit selected preview resolution."""
        # Convert to RGB if grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Store full resolution frame
        self.full_frame = frame.copy()
        
        h, w = frame.shape[:2]
        aspect = w / h
        
        # Always resize to fit within the display resolution while maintaining aspect ratio
        if aspect > self.display_width / self.display_height:
            # Width is the limiting factor
            new_w = self.display_width
            new_h = int(new_w / aspect)
        else:
            # Height is the limiting factor
            new_h = self.display_height
            new_w = int(new_h * aspect)
        
        # Resize frame to fit display resolution
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pil_image = PIL.Image.fromarray(frame_resized)
        
        # Update canvas scroll region to match resized frame (no scrolling needed)
        self.preview_canvas.configure(scrollregion=(0, 0, new_w, new_h))
        
        # Update info label with original and display resolutions
        # (Only update resolution info, preserve FPS info if it exists)
        
        return pil_image

    def on_exposure_change(self, value):
        """Handle exposure slider change (user manual adjustment)."""
        if self.camera and self.camera.is_initialized:
            # Disable auto exposure if manual change
            if self.auto_exposure_var.get() == 1:
                self.auto_exposure_var.set(0)
                print("✓ Auto Exposure disabled - manual adjustment")
            self.camera.set_exposure(int(value), auto=False)

    def on_gain_change(self, value):
        """Handle gain slider change (user manual adjustment)."""
        if self.camera and self.camera.is_initialized:
            # Disable auto gain if manual change
            if self.auto_gain_var.get() == 1:
                self.auto_gain_var.set(0)
                print("✓ Auto Gain disabled - manual adjustment")
            self.camera.set_gain(int(value), auto=False)

    def on_auto_exposure_toggle(self):
        """Handle auto exposure checkbox toggle."""
        if not self.camera or not self.camera.is_initialized:
            return
        
        auto_enabled = self.auto_exposure_var.get() == 1
        
        if auto_enabled:
            # Enable auto exposure
            self.camera.set_exposure(self.exposure_var.get(), auto=True)
            # Don't disable slider - let it show the auto-adjusted value
            print("✓ Auto Exposure enabled")
        else:
            # Disable auto exposure, set manual value
            self.camera.set_exposure(self.exposure_var.get(), auto=False)
            print("✓ Auto Exposure disabled - manual control")

    def on_auto_gain_toggle(self):
        """Handle auto gain checkbox toggle."""
        if not self.camera or not self.camera.is_initialized:
            return
        
        auto_enabled = self.auto_gain_var.get() == 1
        
        if auto_enabled:
            # Enable auto gain
            self.camera.set_gain(self.gain_var.get(), auto=True)
            # Don't disable slider - let it show the auto-adjusted value
            print("✓ Auto Gain enabled")
        else:
            # Disable auto gain, set manual value
            self.camera.set_gain(self.gain_var.get(), auto=False)
            print("✓ Auto Gain disabled - manual control")

    def update_auto_controls(self):
        """Update sliders with current camera values when in auto mode."""
        if not self.camera or not self.camera.is_initialized:
            return
        
        try:
            # Update exposure slider if auto exposure is enabled
            if self.auto_exposure_var.get() == 1:
                current_exposure = self.camera.get_control_value(asi.ASI_EXPOSURE)
                if current_exposure:
                    # Convert from microseconds to display value
                    exposure_val = int(current_exposure[0])
                    # Temporarily disable command, update value, then restore
                    old_command = self.exposure_scale.cget('command')
                    self.exposure_scale.config(command='')
                    self.exposure_scale.set(exposure_val)
                    self.exposure_scale.config(command=old_command)
            
            # Update gain slider if auto gain is enabled
            if self.auto_gain_var.get() == 1:
                current_gain = self.camera.get_control_value(asi.ASI_GAIN)
                if current_gain:
                    gain_val = int(current_gain[0])
                    # Temporarily disable command, update value, then restore
                    old_command = self.gain_scale.cget('command')
                    self.gain_scale.config(command='')
                    self.gain_scale.set(gain_val)
                    self.gain_scale.config(command=old_command)
                    
        except Exception as e:
            pass  # Silently ignore errors to avoid spamming console

    def on_usb_change(self, value):
        """Handle USB bandwidth slider change."""
        if self.camera and self.camera.is_initialized:
            self.camera.set_usb_bandwidth(int(value))

    def on_bin_change(self):
        """Handle binning mode change."""
        if self.camera and self.camera.is_initialized:
            # Would need to stop/restart acquisition to change binning
            print(f"Binning mode changed to {self.bin_var.get()}")

    def on_resolution_change(self, event=None):
        """Handle preview resolution change."""
        resolution_str = self.resolution_var.get()
        
        # Parse resolution from string
        if "Full Resolution" in resolution_str:
            # Use camera's full resolution
            if self.camera and hasattr(self.camera, 'resolution_x'):
                self.display_width = self.camera.resolution_x
                self.display_height = self.camera.resolution_y
                print(f"Preview resolution set to Full: {self.display_width}x{self.display_height}")
            else:
                # Default to 4K if camera not initialized
                self.display_width = 3840
                self.display_height = 2160
                print(f"Preview resolution set to Full (default 4K): {self.display_width}x{self.display_height}")
        else:
            # Extract WxH from string like "1920x1080 (1080p)"
            try:
                resolution_part = resolution_str.split()[0]  # Get "1920x1080" part
                width, height = map(int, resolution_part.split('x'))
                self.display_width = width
                self.display_height = height
                print(f"Preview resolution changed to: {self.display_width}x{self.display_height}")
            except Exception as e:
                print(f"Error parsing resolution: {e}")
                return
        
        # Update info label
        self.info_label.config(
            text=f"Preview resolution set to: {self.display_width}x{self.display_height}"
        )
        
        # Force canvas resize to new dimensions
        self.preview_canvas.config(width=self.display_width, height=self.display_height)
        
        # If preview is active, the next frame will use the new dimensions
        # The update_preview loop will automatically resize the video feed

    def on_filter_toggle(self, filter_key):
        """Handle filter enable/disable toggle."""
        if not self.processor:
            return

        enabled = self.filter_vars[filter_key].get()
        
        # Handle stabilization separately
        if filter_key == 'stabilization':
            if self.stabilizer:
                if enabled:
                    self.stabilizer.enable()
                else:
                    self.stabilizer.disable()
            return

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

    def _on_mousewheel(self, event):
        """Handle vertical mouse wheel scrolling."""
        self.preview_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_shift_mousewheel(self, event):
        """Handle horizontal mouse wheel scrolling (Shift + wheel)."""
        self.preview_canvas.xview_scroll(-1 * int(event.delta / 120), "units")

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
