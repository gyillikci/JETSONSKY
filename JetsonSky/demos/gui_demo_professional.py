#!/usr/bin/env python3
"""
JetsonSky Professional GUI - Full Feature Set

Complete astronomy imaging GUI with all original JetsonSky controls:
- Real-time camera preview with Phase 2 filter integration
- Full camera controls (exposition, gain, resolution, USB bandwidth)
- Advanced image processing (denoise, sharpen, contrast, amplification)
- Color management (saturation, white balance, RGB channels)
- Histogram display and statistics
- Real and simulated camera support

Author: JetsonSky Refactoring Project
License: GPL-3.0
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime
import queue

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    AppState,
    get_camera_config,
    get_supported_cameras,
)
from utils.constants import (
    DEFAULT_EXPOSITION,
    DEFAULT_GAIN,
)

# Check dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageTk, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Import Phase 2 filters
try:
    from filters import (
        FilterPipeline,
        HotPixelFilter,
        DenoiseKNNFilter,
        DenoisePaillouFilter,
        SharpenFilter,
        LaplacianSharpenFilter,
        CLAHEFilter,
        SaturationFilter,
        WhiteBalanceFilter,
        FlipFilter,
        GammaCorrectionFilter,
    )
    HAS_FILTERS = True
except ImportError:
    HAS_FILTERS = False

# Real camera support
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'zwoasi_cupy'))
    import zwoasi_cupy as asi
    HAS_REAL_CAMERA = True
except ImportError:
    try:
        import zwoasi as asi
        HAS_REAL_CAMERA = True
    except ImportError:
        HAS_REAL_CAMERA = False

from demos.camera_simulator import create_simulated_camera


class ProfessionalJetsonSkyGUI:
    """
    Professional JetsonSky GUI with complete feature set.

    Includes all original controls plus Phase 2 filter integration.
    """

    def __init__(self, root):
        """Initialize professional GUI."""
        self.root = root
        self.root.title("JetsonSky Professional - Real-Time Astronomy Imaging")

        # Maximize window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width-100}x{screen_height-100}+50+50")

        # Application state
        self.app = AppState()
        self.camera = None
        self.using_real_camera = False
        self.acquisition_thread = None
        self.current_frame = None
        self.photo_image = None
        self.frame_queue = queue.Queue(maxsize=2)

        # Filter pipeline
        self.pipeline = FilterPipeline() if HAS_FILTERS else None

        # Statistics
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.histogram_data = None

        # Control variables
        self.init_control_variables()

        # Build UI
        self.build_ui()

        # Setup filters
        if HAS_FILTERS:
            self.setup_filters()

    def init_control_variables(self):
        """Initialize all control variables."""
        # Camera controls
        self.exposition_var = tk.IntVar(value=DEFAULT_EXPOSITION)
        self.gain_var = tk.IntVar(value=DEFAULT_GAIN)
        self.usb_bandwidth_var = tk.IntVar(value=40)
        self.gamma_var = tk.DoubleVar(value=50.0)
        self.binning_var = tk.IntVar(value=1)

        # Image processing - Denoise
        self.denoise_knn_var = tk.DoubleVar(value=0.2)
        self.denoise_paillou_var = tk.DoubleVar(value=0.4)
        self.denoise_3frame_var = tk.DoubleVar(value=0.5)

        # Image processing - Sharpen
        self.sharpen1_amount_var = tk.DoubleVar(value=1.0)
        self.sharpen1_sigma_var = tk.DoubleVar(value=1.0)
        self.sharpen2_amount_var = tk.DoubleVar(value=1.0)
        self.sharpen2_sigma_var = tk.DoubleVar(value=2.0)

        # Contrast
        self.clahe_clip_var = tk.DoubleVar(value=2.0)
        self.clahe_grid_var = tk.IntVar(value=8)
        self.amplification_var = tk.DoubleVar(value=1.0)

        # Advanced processing
        self.mu_var = tk.DoubleVar(value=0.0)
        self.ro_var = tk.DoubleVar(value=1.0)

        # Color controls
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.wb_red_var = tk.IntVar(value=63)
        self.wb_blue_var = tk.IntVar(value=74)
        self.red_mult_var = tk.DoubleVar(value=1.0)
        self.green_mult_var = tk.DoubleVar(value=1.0)
        self.blue_mult_var = tk.DoubleVar(value=1.0)

        # Filter enable flags
        self.enable_hotpixel_var = tk.BooleanVar(value=False)
        self.enable_denoise_knn_var = tk.BooleanVar(value=False)
        self.enable_denoise_paillou_var = tk.BooleanVar(value=False)
        self.enable_sharpen1_var = tk.BooleanVar(value=False)
        self.enable_sharpen2_var = tk.BooleanVar(value=False)
        self.enable_clahe_var = tk.BooleanVar(value=False)
        self.enable_saturation_var = tk.BooleanVar(value=False)
        self.enable_wb_var = tk.BooleanVar(value=False)
        self.enable_gamma_var = tk.BooleanVar(value=False)

        # Display
        self.flip_v_var = tk.BooleanVar(value=False)
        self.flip_h_var = tk.BooleanVar(value=False)
        self.show_histogram_var = tk.BooleanVar(value=True)

    def build_ui(self):
        """Build comprehensive user interface."""
        # Main container with notebook (tabs)
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create main layout: left panel, center (preview), right panel
        left_panel = tk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        left_panel.pack_propagate(False)

        center_panel = tk.Frame(main_container)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        right_panel = tk.Frame(main_container, width=350)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        right_panel.pack_propagate(False)

        # Build each section
        self.build_left_panel(left_panel)
        self.build_center_panel(center_panel)
        self.build_right_panel(right_panel)

        # Status bar at bottom
        self.build_status_bar()

    def build_left_panel(self, parent):
        """Build left panel with camera and basic controls."""
        # Camera selection
        camera_frame = tk.LabelFrame(parent, text="Camera Selection", padx=10, pady=10)
        camera_frame.pack(fill=tk.X, pady=5)

        tk.Label(camera_frame, text="Model:").pack(anchor=tk.W)
        self.camera_var = tk.StringVar()
        cameras = get_supported_cameras()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, width=28)
        self.camera_combo['values'] = cameras
        self.camera_combo.current(cameras.index("ZWO ASI178MC"))
        self.camera_combo.pack(fill=tk.X, pady=5)

        tk.Button(camera_frame, text="🎥 Load Camera", command=self.load_camera,
                 bg="#3498db", fg="white", height=2).pack(fill=tk.X, pady=5)

        # Camera controls with notebook
        controls_notebook = ttk.Notebook(parent)
        controls_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Tab 1: Basic Controls
        basic_tab = tk.Frame(controls_notebook)
        controls_notebook.add(basic_tab, text="Basic")
        self.build_basic_controls(basic_tab)

        # Tab 2: Advanced Controls
        advanced_tab = tk.Frame(controls_notebook)
        controls_notebook.add(advanced_tab, text="Advanced")
        self.build_advanced_controls(advanced_tab)

        # Tab 3: Color Controls
        color_tab = tk.Frame(controls_notebook)
        controls_notebook.add(color_tab, text="Color")
        self.build_color_controls(color_tab)

        # Acquisition buttons
        acq_frame = tk.Frame(parent)
        acq_frame.pack(fill=tk.X, pady=5)

        tk.Button(acq_frame, text="▶ Start", command=self.start_acquisition,
                 bg="#27ae60", fg="white", height=2).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(acq_frame, text="⏹ Stop", command=self.stop_acquisition,
                 bg="#e74c3c", fg="white", height=2).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Button(parent, text="💾 Save Image", command=self.save_image,
                 bg="#9b59b6", fg="white", height=2).pack(fill=tk.X, pady=5)

    def build_basic_controls(self, parent):
        """Build basic camera controls."""
        scroll_canvas = tk.Canvas(parent)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tk.Frame(scroll_canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )

        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Exposition
        self.create_slider(scroll_frame, "Exposition (µs)", self.exposition_var,
                          100, 100000, 100, self.update_exposition, row=0)

        # Gain
        self.create_slider(scroll_frame, "Gain", self.gain_var,
                          0, 600, 1, self.update_gain, row=1)

        # USB Bandwidth
        self.create_slider(scroll_frame, "USB Bandwidth", self.usb_bandwidth_var,
                          40, 100, 1, self.update_usb, row=2)

        # Binning
        bin_frame = tk.LabelFrame(scroll_frame, text="Binning", padx=5, pady=5)
        bin_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Radiobutton(bin_frame, text="1x1", variable=self.binning_var, value=1,
                      command=self.update_binning).pack(side=tk.LEFT)
        tk.Radiobutton(bin_frame, text="2x2", variable=self.binning_var, value=2,
                      command=self.update_binning).pack(side=tk.LEFT)

        # Flip controls
        flip_frame = tk.LabelFrame(scroll_frame, text="Flip", padx=5, pady=5)
        flip_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Checkbutton(flip_frame, text="Vertical", variable=self.flip_v_var,
                      command=self.update_flip).pack(anchor=tk.W)
        tk.Checkbutton(flip_frame, text="Horizontal", variable=self.flip_h_var,
                      command=self.update_flip).pack(anchor=tk.W)

    def build_advanced_controls(self, parent):
        """Build advanced processing controls."""
        scroll_canvas = tk.Canvas(parent)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tk.Frame(scroll_canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )

        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Gamma
        self.create_slider_with_checkbox(scroll_frame, "Gamma", self.gamma_var,
                                         self.enable_gamma_var, 0, 100, 1,
                                         self.update_gamma, row)
        row += 1

        # Amplification
        self.create_slider(scroll_frame, "Amplification", self.amplification_var,
                          0.0, 20.0, 0.1, self.update_amplification, row)
        row += 1

        # Mu parameter
        self.create_slider(scroll_frame, "Mu (Star Amp)", self.mu_var,
                          -5.0, 5.0, 0.1, self.update_mu, row)
        row += 1

        # Ro parameter
        self.create_slider(scroll_frame, "Ro (Star Amp)", self.ro_var,
                          0.2, 5.0, 0.1, self.update_ro, row)
        row += 1

    def build_color_controls(self, parent):
        """Build color adjustment controls."""
        scroll_canvas = tk.Canvas(parent)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tk.Frame(scroll_canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )

        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Saturation
        self.create_slider_with_checkbox(scroll_frame, "Saturation", self.saturation_var,
                                         self.enable_saturation_var, 0.0, 2.0, 0.01,
                                         self.update_saturation, row)
        row += 1

        # White Balance section
        wb_frame = tk.LabelFrame(scroll_frame, text="White Balance", padx=5, pady=5)
        wb_frame.pack(fill=tk.X, padx=5, pady=5)
        row += 1

        tk.Checkbutton(wb_frame, text="Enable WB", variable=self.enable_wb_var,
                      command=self.update_white_balance).grid(row=0, column=0, columnspan=2, sticky=tk.W)

        tk.Label(wb_frame, text="Red:").grid(row=1, column=0, sticky=tk.W)
        tk.Scale(wb_frame, from_=1, to=99, orient=tk.HORIZONTAL,
                variable=self.wb_red_var, command=lambda v: self.update_white_balance(),
                length=180).grid(row=1, column=1, sticky=tk.W)

        tk.Label(wb_frame, text="Blue:").grid(row=2, column=0, sticky=tk.W)
        tk.Scale(wb_frame, from_=1, to=99, orient=tk.HORIZONTAL,
                variable=self.wb_blue_var, command=lambda v: self.update_white_balance(),
                length=180).grid(row=2, column=1, sticky=tk.W)

        # RGB Channel Multipliers
        rgb_frame = tk.LabelFrame(scroll_frame, text="RGB Multipliers", padx=5, pady=5)
        rgb_frame.pack(fill=tk.X, padx=5, pady=5)
        row += 1

        tk.Label(rgb_frame, text="Red:").grid(row=0, column=0, sticky=tk.W)
        tk.Scale(rgb_frame, from_=0, to=2, orient=tk.HORIZONTAL, resolution=0.01,
                variable=self.red_mult_var, command=lambda v: self.update_rgb_multipliers(),
                length=180).grid(row=0, column=1, sticky=tk.W)

        tk.Label(rgb_frame, text="Green:").grid(row=1, column=0, sticky=tk.W)
        tk.Scale(rgb_frame, from_=0, to=2, orient=tk.HORIZONTAL, resolution=0.01,
                variable=self.green_mult_var, command=lambda v: self.update_rgb_multipliers(),
                length=180).grid(row=1, column=1, sticky=tk.W)

        tk.Label(rgb_frame, text="Blue:").grid(row=2, column=0, sticky=tk.W)
        tk.Scale(rgb_frame, from_=0, to=2, orient=tk.HORIZONTAL, resolution=0.01,
                variable=self.blue_mult_var, command=lambda v: self.update_rgb_multipliers(),
                length=180).grid(row=2, column=1, sticky=tk.W)

    def build_center_panel(self, parent):
        """Build center preview panel."""
        # Preview frame
        preview_frame = tk.LabelFrame(parent, text="Camera Preview", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for image
        self.canvas = tk.Canvas(preview_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Image info
        info_frame = tk.Frame(preview_frame)
        info_frame.pack(fill=tk.X, pady=5)

        self.image_info_label = tk.Label(info_frame, text="No image", font=("Arial", 9))
        self.image_info_label.pack(side=tk.LEFT, padx=10)

        self.fps_label = tk.Label(info_frame, text="FPS: 0.0", font=("Arial", 9))
        self.fps_label.pack(side=tk.LEFT, padx=10)

        self.frame_count_label = tk.Label(info_frame, text="Frame: 0", font=("Arial", 9))
        self.frame_count_label.pack(side=tk.LEFT, padx=10)

        # Histogram frame (optional)
        hist_frame = tk.LabelFrame(parent, text="Histogram", padx=5, pady=5)
        hist_frame.pack(fill=tk.X, pady=5)

        tk.Checkbutton(hist_frame, text="Show Histogram", variable=self.show_histogram_var).pack(anchor=tk.W)

        self.histogram_canvas = tk.Canvas(hist_frame, height=100, bg="white")
        self.histogram_canvas.pack(fill=tk.X)

    def build_right_panel(self, parent):
        """Build right panel with filters."""
        # Filter controls with notebook
        filter_notebook = ttk.Notebook(parent)
        filter_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Noise Reduction
        denoise_tab = tk.Frame(filter_notebook)
        filter_notebook.add(denoise_tab, text="Denoise")
        self.build_denoise_filters(denoise_tab)

        # Tab 2: Sharpening
        sharpen_tab = tk.Frame(filter_notebook)
        filter_notebook.add(sharpen_tab, text="Sharpen")
        self.build_sharpen_filters(sharpen_tab)

        # Tab 3: Contrast
        contrast_tab = tk.Frame(filter_notebook)
        filter_notebook.add(contrast_tab, text="Contrast")
        self.build_contrast_filters(contrast_tab)

        # Tab 4: Status & Stats
        stats_tab = tk.Frame(filter_notebook)
        filter_notebook.add(stats_tab, text="Stats")
        self.build_stats_panel(stats_tab)

    def build_denoise_filters(self, parent):
        """Build denoise filter controls."""
        # Hot Pixel Removal
        tk.Checkbutton(parent, text="Hot Pixel Removal", variable=self.enable_hotpixel_var,
                      command=self.update_filters).pack(anchor=tk.W, padx=10, pady=5)

        # KNN Denoise
        self.create_slider_with_checkbox(parent, "KNN Denoise", self.denoise_knn_var,
                                         self.enable_denoise_knn_var, 0.05, 1.2, 0.05,
                                         self.update_denoise_knn, row=0)

        # Paillou Denoise
        self.create_slider_with_checkbox(parent, "Paillou Denoise", self.denoise_paillou_var,
                                         self.enable_denoise_paillou_var, 0.1, 1.2, 0.1,
                                         self.update_denoise_paillou, row=1)

        # 3-Frame NR
        self.create_slider(parent, "3-Frame NR Threshold", self.denoise_3frame_var,
                          0.2, 0.8, 0.05, self.update_3frame, row=2)

    def build_sharpen_filters(self, parent):
        """Build sharpen filter controls."""
        # Sharpen 1
        sharpen1_frame = tk.LabelFrame(parent, text="Sharpen 1 (Unsharp Mask)", padx=10, pady=10)
        sharpen1_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(sharpen1_frame, text="Enable", variable=self.enable_sharpen1_var,
                      command=self.update_sharpen).pack(anchor=tk.W)

        tk.Label(sharpen1_frame, text="Amount:").pack(anchor=tk.W)
        tk.Scale(sharpen1_frame, from_=0, to=10, orient=tk.HORIZONTAL, resolution=0.2,
                variable=self.sharpen1_amount_var, command=lambda v: self.update_sharpen(),
                length=250).pack(fill=tk.X)

        tk.Label(sharpen1_frame, text="Sigma:").pack(anchor=tk.W)
        tk.Scale(sharpen1_frame, from_=1, to=9, orient=tk.HORIZONTAL, resolution=1,
                variable=self.sharpen1_sigma_var, command=lambda v: self.update_sharpen(),
                length=250).pack(fill=tk.X)

        # Sharpen 2
        sharpen2_frame = tk.LabelFrame(parent, text="Sharpen 2 (Laplacian)", padx=10, pady=10)
        sharpen2_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(sharpen2_frame, text="Enable", variable=self.enable_sharpen2_var,
                      command=self.update_sharpen).pack(anchor=tk.W)

        tk.Label(sharpen2_frame, text="Amount:").pack(anchor=tk.W)
        tk.Scale(sharpen2_frame, from_=0, to=10, orient=tk.HORIZONTAL, resolution=0.2,
                variable=self.sharpen2_amount_var, command=lambda v: self.update_sharpen(),
                length=250).pack(fill=tk.X)

        tk.Label(sharpen2_frame, text="Sigma:").pack(anchor=tk.W)
        tk.Scale(sharpen2_frame, from_=1, to=9, orient=tk.HORIZONTAL, resolution=1,
                variable=self.sharpen2_sigma_var, command=lambda v: self.update_sharpen(),
                length=250).pack(fill=tk.X)

    def build_contrast_filters(self, parent):
        """Build contrast filter controls."""
        # CLAHE
        clahe_frame = tk.LabelFrame(parent, text="CLAHE", padx=10, pady=10)
        clahe_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(clahe_frame, text="Enable CLAHE", variable=self.enable_clahe_var,
                      command=self.update_clahe).pack(anchor=tk.W)

        tk.Label(clahe_frame, text="Clip Limit:").pack(anchor=tk.W)
        tk.Scale(clahe_frame, from_=0.5, to=10.0, orient=tk.HORIZONTAL, resolution=0.1,
                variable=self.clahe_clip_var, command=lambda v: self.update_clahe(),
                length=250).pack(fill=tk.X)

        tk.Label(clahe_frame, text="Grid Size:").pack(anchor=tk.W)
        tk.Scale(clahe_frame, from_=4, to=16, orient=tk.HORIZONTAL, resolution=2,
                variable=self.clahe_grid_var, command=lambda v: self.update_clahe(),
                length=250).pack(fill=tk.X)

    def build_stats_panel(self, parent):
        """Build statistics panel."""
        # Pipeline info
        pipeline_frame = tk.LabelFrame(parent, text="Filter Pipeline", padx=10, pady=10)
        pipeline_frame.pack(fill=tk.X, padx=5, pady=5)

        self.pipeline_info_text = tk.Text(pipeline_frame, height=10, width=35, state=tk.DISABLED,
                                          font=("Courier", 9))
        self.pipeline_info_text.pack(fill=tk.BOTH, expand=True)

        # Image statistics
        stats_frame = tk.LabelFrame(parent, text="Image Statistics", padx=10, pady=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_text = tk.Text(stats_frame, height=8, width=35, state=tk.DISABLED,
                                  font=("Courier", 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Status log
        log_frame = tk.LabelFrame(parent, text="Status Log", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, height=10, width=35, state=tk.DISABLED,
                               font=("Courier", 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

    def build_status_bar(self):
        """Build status bar at bottom."""
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Camera type indicator
        self.camera_type_label = tk.Label(status_frame, text="No camera", anchor=tk.E)
        self.camera_type_label.pack(side=tk.RIGHT, padx=5)

    def create_slider(self, parent, label, variable, from_, to, resolution, command, row):
        """Create labeled slider control."""
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(frame, text=label).pack(anchor=tk.W)
        tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, resolution=resolution,
                variable=variable, command=command, length=220).pack(fill=tk.X)

    def create_slider_with_checkbox(self, parent, label, variable, checkbox_var, from_, to, resolution, command, row):
        """Create slider with enable checkbox."""
        frame = tk.LabelFrame(parent, text=label, padx=5, pady=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(frame, text="Enable", variable=checkbox_var,
                      command=command).pack(anchor=tk.W)

        tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, resolution=resolution,
                variable=variable, command=lambda v: command(), length=220).pack(fill=tk.X)

    # Control update methods (will implement filter pipeline integration)
    def update_exposition(self, value=None):
        """Update exposition."""
        if self.camera and hasattr(self.camera, 'set_exposition'):
            self.camera.set_exposition(self.exposition_var.get())

    def update_gain(self, value=None):
        """Update gain."""
        if self.camera and hasattr(self.camera, 'set_gain'):
            self.camera.set_gain(self.gain_var.get())

    def update_usb(self, value=None):
        """Update USB bandwidth."""
        pass  # Implement for real camera

    def update_binning(self):
        """Update binning mode."""
        self.app.binning_mode = self.binning_var.get()

    def update_flip(self):
        """Update flip settings."""
        if self.pipeline:
            flip_filter = self.pipeline.get_filter("FlipFilter")
            if flip_filter:
                flip_filter.set_vertical(self.flip_v_var.get())
                flip_filter.set_horizontal(self.flip_h_var.get())
                if self.flip_v_var.get() or self.flip_h_var.get():
                    flip_filter.enable()
                else:
                    flip_filter.disable()

    def update_gamma(self):
        """Update gamma correction."""
        if self.pipeline:
            gamma_filter = self.pipeline.get_filter("GammaCorrectionFilter")
            if gamma_filter:
                # Convert 0-100 slider to gamma value (0.5-2.0)
                gamma_value = 0.5 + (self.gamma_var.get() / 100.0) * 1.5
                gamma_filter.set_gamma(gamma_value)
                if self.enable_gamma_var.get():
                    gamma_filter.enable()
                else:
                    gamma_filter.disable()

    def update_amplification(self, value=None):
        """Update amplification."""
        pass  # For future implementation

    def update_mu(self, value=None):
        """Update Mu parameter."""
        pass  # For star amplification

    def update_ro(self, value=None):
        """Update Ro parameter."""
        pass  # For star amplification

    def update_saturation(self):
        """Update saturation."""
        if self.pipeline:
            sat_filter = self.pipeline.get_filter("SaturationFilter")
            if sat_filter:
                sat_filter.set_saturation(self.saturation_var.get())
                if self.enable_saturation_var.get():
                    sat_filter.enable()
                else:
                    sat_filter.disable()

    def update_white_balance(self):
        """Update white balance."""
        if self.pipeline:
            wb_filter = self.pipeline.get_filter("WhiteBalanceFilter")
            if wb_filter:
                wb_filter.set_red_balance(self.wb_red_var.get())
                wb_filter.set_blue_balance(self.wb_blue_var.get())
                if self.enable_wb_var.get():
                    wb_filter.enable()
                else:
                    wb_filter.disable()

    def update_rgb_multipliers(self):
        """Update RGB channel multipliers."""
        pass  # For future custom filter

    def update_filters(self):
        """Update general filters."""
        if self.pipeline:
            if self.enable_hotpixel_var.get():
                self.pipeline.enable_filter("HotPixelFilter")
            else:
                self.pipeline.disable_filter("HotPixelFilter")
            self.update_pipeline_info()

    def update_denoise_knn(self):
        """Update KNN denoise."""
        if self.pipeline:
            knn_filter = self.pipeline.get_filter("DenoiseKNNFilter")
            if knn_filter:
                knn_filter.set_strength(self.denoise_knn_var.get())
                if self.enable_denoise_knn_var.get():
                    knn_filter.enable()
                else:
                    knn_filter.disable()
            self.update_pipeline_info()

    def update_denoise_paillou(self):
        """Update Paillou denoise."""
        if self.pipeline:
            paillou_filter = self.pipeline.get_filter("DenoisePaillouFilter")
            if paillou_filter:
                paillou_filter.set_strength(self.denoise_paillou_var.get())
                if self.enable_denoise_paillou_var.get():
                    paillou_filter.enable()
                else:
                    paillou_filter.disable()
            self.update_pipeline_info()

    def update_3frame(self, value=None):
        """Update 3-frame noise reduction."""
        pass  # For future implementation

    def update_sharpen(self):
        """Update sharpen filters."""
        if self.pipeline:
            # Sharpen 1
            sharpen1 = self.pipeline.get_filter("SharpenFilter")
            if sharpen1:
                sharpen1.set_amount(self.sharpen1_amount_var.get())
                sharpen1.set_sigma(self.sharpen1_sigma_var.get())
                if self.enable_sharpen1_var.get():
                    sharpen1.enable()
                else:
                    sharpen1.disable()

            # Sharpen 2
            sharpen2 = self.pipeline.get_filter("LaplacianSharpenFilter")
            if sharpen2:
                sharpen2.set_strength(self.sharpen2_amount_var.get())
                if self.enable_sharpen2_var.get():
                    sharpen2.enable()
                else:
                    sharpen2.disable()

            self.update_pipeline_info()

    def update_clahe(self):
        """Update CLAHE contrast."""
        if self.pipeline:
            clahe_filter = self.pipeline.get_filter("CLAHEFilter")
            if clahe_filter:
                clahe_filter.set_clip_limit(self.clahe_clip_var.get())
                clahe_filter.set_grid_size(int(self.clahe_grid_var.get()))
                if self.enable_clahe_var.get():
                    clahe_filter.enable()
                else:
                    clahe_filter.disable()
            self.update_pipeline_info()

    def setup_filters(self):
        """Setup Phase 2 filter pipeline."""
        if not HAS_OPENCV:
            return

        # Add all filters (disabled by default)
        self.pipeline.add_filter(FlipFilter(enabled=False))
        self.pipeline.add_filter(HotPixelFilter(enabled=False))
        self.pipeline.add_filter(DenoiseKNNFilter(strength=0.2, enabled=False))
        self.pipeline.add_filter(DenoisePaillouFilter(strength=0.4, enabled=False))
        self.pipeline.add_filter(SharpenFilter(amount=1.0, sigma=1.0, enabled=False))
        self.pipeline.add_filter(LaplacianSharpenFilter(strength=1.0, enabled=False))
        self.pipeline.add_filter(CLAHEFilter(clip_limit=2.0, grid_size=8, enabled=False))
        self.pipeline.add_filter(SaturationFilter(saturation=1.0, enabled=False))
        self.pipeline.add_filter(WhiteBalanceFilter(red_balance=63, blue_balance=74, enabled=False))
        self.pipeline.add_filter(GammaCorrectionFilter(gamma=1.0, enabled=False))

        self.log("✓ Filter pipeline initialized with 10 filters")
        self.update_pipeline_info()

    def update_pipeline_info(self):
        """Update pipeline info display."""
        if not self.pipeline:
            return

        info = f"Filter Pipeline Status:\n"
        info += f"{'='*30}\n"
        info += f"Total filters: {len(self.pipeline)}\n"
        info += f"Active filters: {self.pipeline.get_enabled_count()}\n\n"

        for f in self.pipeline.filters:
            status = "✓" if f.is_enabled() else "✗"
            info += f"{status} {f.get_name()}\n"

        self.pipeline_info_text.config(state=tk.NORMAL)
        self.pipeline_info_text.delete(1.0, tk.END)
        self.pipeline_info_text.insert(1.0, info)
        self.pipeline_info_text.config(state=tk.DISABLED)

    def load_camera(self):
        """Load selected camera."""
        camera_model = self.camera_var.get()

        try:
            self.app.camera_config = get_camera_config(camera_model)
            self.app.camera_connected = True
            self.app.resolution_mode = 1
            self.app.binning_mode = self.binning_var.get()

            resolution = self.app.get_current_resolution()
            self.log(f"✓ Camera loaded: {camera_model}")
            self.log(f"  Resolution: {resolution[0]}x{resolution[1]}")
            self.status_label.config(text=f"Camera: {camera_model}")

        except Exception as e:
            self.log(f"✗ Failed to load camera: {e}")
            messagebox.showerror("Error", f"Failed to load camera: {e}")

    def start_acquisition(self):
        """Start camera acquisition."""
        if not HAS_NUMPY:
            messagebox.showerror("Error", "NumPy required")
            return

        if self.app.camera_config is None:
            messagebox.showwarning("Warning", "Please load a camera first!")
            return

        if self.app.acquisition_running:
            return

        resolution = self.app.get_current_resolution()

        # Try to use real camera if available
        if HAS_REAL_CAMERA:
            try:
                # Initialize ASI library
                lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Lib')
                if sys.platform == "win32":
                    asi_lib = os.path.join(lib_path, "ASICamera2.dll")
                else:
                    asi_lib = os.path.join(lib_path, "libASICamera2.so.1.27")
                
                if os.path.exists(asi_lib):
                    asi.init(asi_lib)
                    num_cameras = asi.get_num_cameras()
                    
                    if num_cameras > 0:
                        # Use real camera
                        self.camera = asi.Camera(0)
                        camera_info = self.camera.get_camera_property()
                        
                        # Configure camera
                        self.camera.set_control_value(asi.ASI_GAIN, int(self.gain_var.get()))
                        self.camera.set_control_value(asi.ASI_EXPOSURE, int(self.exposition_var.get()))
                        self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, int(self.usb_bandwidth_var.get()))
                        
                        # Set ROI - use RAW8 for faster capture
                        self.camera.set_roi(
                            width=resolution[0],
                            height=resolution[1],
                            bins=self.binning_var.get(),
                            image_type=asi.ASI_IMG_RAW8
                        )
                        
                        self.camera.start_video_capture()
                        self.log(f"▶ Acquisition started with REAL camera: {camera_info['Name']}")
                        self.camera_type_label.config(text=f"Real: {camera_info['Name']}", fg="green")
                        self.using_real_camera = True
                    else:
                        raise Exception("No real camera detected")
                else:
                    raise Exception("Camera library not found")
                    
            except Exception as e:
                self.log(f"⚠ Real camera failed: {e}, using simulator")
                self.using_real_camera = False
                self.camera = create_simulated_camera(
                    self.app.camera_config.model,
                    resolution,
                    self.app.camera_config.sensor_bits
                )
                self.camera.set_exposition(self.exposition_var.get())
                self.camera.set_gain(self.gain_var.get())
                self.camera.start_capture()
                self.camera_type_label.config(text="Simulator", fg="blue")
        else:
            # Use simulator
            self.camera = create_simulated_camera(
                self.app.camera_config.model,
                resolution,
                self.app.camera_config.sensor_bits
            )
            self.camera.set_exposition(self.exposition_var.get())
            self.camera.set_gain(self.gain_var.get())
            self.camera.start_capture()
            self.log("▶ Acquisition started with simulated camera")
            self.camera_type_label.config(text="Simulator", fg="blue")
            self.using_real_camera = False

        self.app.acquisition_running = True
        self.frame_count = 0
        self.last_time = time.time()

        self.log("▶ Acquisition started")
        self.status_label.config(text="Acquiring...")

        self.acquisition_thread = threading.Thread(target=self.acquisition_loop, daemon=True)
        self.acquisition_thread.start()

    def stop_acquisition(self):
        """Stop acquisition."""
        if not self.app.acquisition_running:
            return

        self.app.acquisition_running = False

        if self.camera:
            try:
                # Stop real camera video capture
                if hasattr(self, 'using_real_camera') and self.using_real_camera:
                    if hasattr(self.camera, 'stop_video_capture'):
                        self.camera.stop_video_capture()
                    if hasattr(self.camera, 'close'):
                        self.camera.close()
                else:
                    # Stop simulator
                    if hasattr(self.camera, 'stop_capture'):
                        self.camera.stop_capture()
                    if hasattr(self.camera, 'close'):
                        self.camera.close()
            except Exception as e:
                self.log(f"⚠ Error stopping camera: {e}")

        self.log("⏹ Acquisition stopped")
        self.status_label.config(text="Stopped")

    def acquisition_loop(self):
        """Main acquisition loop."""
        while self.app.acquisition_running:
            try:
                if self.camera:
                    # Get frame based on camera type
                    if hasattr(self, 'using_real_camera') and self.using_real_camera:
                        # Real camera - use optimized RAW8 capture with CuPy
                        try:
                            if sys.platform == "win32":
                                frame = self.camera.capture_video_frame_RAW8_CUPY(filename=None, timeout=1000)
                            else:
                                frame = self.camera.capture_video_frame_RAW8_NUMPY(filename=None, timeout=1000)
                            
                            # Debayer to color if needed
                            if frame is not None and HAS_OPENCV:
                                # Convert CuPy to NumPy if needed
                                if hasattr(frame, 'get'):
                                    frame = frame.get()
                                # Debayer using OpenCV
                                frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
                        except Exception as e:
                            frame = None
                    else:
                        # Simulator
                        frame = self.camera.capture_frame()

                    if frame is not None and HAS_NUMPY:
                        self.frame_count += 1

                        # Calculate FPS
                        current_time = time.time()
                        if current_time - self.last_time > 0:
                            self.fps = 1.0 / (current_time - self.last_time)
                        self.last_time = current_time

                        # Apply filters
                        if self.pipeline and HAS_OPENCV:
                            filtered_frame = self.pipeline.apply(frame)
                        else:
                            filtered_frame = frame

                        self.current_frame = filtered_frame

                        # Update display (use queue to avoid blocking)
                        if not self.frame_queue.full():
                            self.frame_queue.put(filtered_frame)
                            self.root.after(0, self.process_frame_queue)

                time.sleep(0.01)  # Reduced sleep for faster frame rate
            except Exception as e:
                self.log(f"✗ Acquisition error: {e}")
                break

    def process_frame_queue(self):
        """Process frame from queue and update display."""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                self.update_display(frame)
                self.update_histogram(frame)
                self.update_statistics(frame)
        except:
            pass

    def update_display(self, frame):
        """Update image display."""
        if not HAS_PIL or frame is None:
            return

        try:
            # Convert to PIL
            if len(frame.shape) == 2:
                if frame.dtype == np.uint16:
                    frame_8bit = (frame / 256).astype(np.uint8)
                else:
                    frame_8bit = frame
                pil_image = Image.fromarray(frame_8bit, mode='L')
            else:
                if frame.dtype == np.uint16:
                    frame_8bit = (frame / 256).astype(np.uint8)
                else:
                    frame_8bit = frame
                if HAS_OPENCV:
                    frame_rgb = cv2.cvtColor(frame_8bit, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame_8bit
                pil_image = Image.fromarray(frame_rgb)

            # Resize to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(pil_image)

            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo_image)

            # Update labels
            mean_val = frame.mean()
            max_val = frame.max()
            min_val = frame.min()

            self.image_info_label.config(
                text=f"{frame.shape[1]}x{frame.shape[0]} | Min:{min_val:.0f} Mean:{mean_val:.0f} Max:{max_val:.0f}"
            )
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.frame_count_label.config(text=f"Frame: {self.frame_count}")

        except Exception as e:
            self.log(f"✗ Display error: {e}")

    def update_histogram(self, frame):
        """Update histogram display."""
        if not self.show_histogram_var.get() or frame is None:
            return

        try:
            # Calculate histogram
            if len(frame.shape) == 2:
                hist, _ = np.histogram(frame.flatten(), bins=256, range=(0, 255))
            else:
                # Use luminance for color images
                if HAS_OPENCV:
                    gray = cv2.cvtColor(frame if frame.dtype == np.uint8 else (frame/256).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
                else:
                    return

            # Draw histogram
            self.histogram_canvas.delete("all")
            canvas_width = self.histogram_canvas.winfo_width()
            canvas_height = self.histogram_canvas.winfo_height()

            if canvas_width > 1:
                max_hist = hist.max()
                if max_hist > 0:
                    bin_width = canvas_width / 256
                    for i, count in enumerate(hist):
                        height = (count / max_hist) * (canvas_height - 10)
                        x0 = i * bin_width
                        y0 = canvas_height - height
                        x1 = (i + 1) * bin_width
                        y1 = canvas_height
                        self.histogram_canvas.create_rectangle(x0, y0, x1, y1, fill="blue", outline="")

        except Exception as e:
            pass

    def update_statistics(self, frame):
        """Update image statistics."""
        if frame is None:
            return

        try:
            stats = f"Image Statistics:\n"
            stats += f"{'='*25}\n"
            stats += f"Shape: {frame.shape}\n"
            stats += f"Dtype: {frame.dtype}\n"
            stats += f"Min: {frame.min()}\n"
            stats += f"Max: {frame.max()}\n"
            stats += f"Mean: {frame.mean():.2f}\n"
            stats += f"Std: {frame.std():.2f}\n"

            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats)
            self.stats_text.config(state=tk.DISABLED)

        except:
            pass

    def save_image(self):
        """Save current frame."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to save!")
            return

        if not HAS_OPENCV:
            messagebox.showerror("Error", "OpenCV required to save images")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"jetsonsky_pro_{timestamp}.png"

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if filename:
            try:
                cv2.imwrite(filename, self.current_frame)
                self.log(f"✓ Saved: {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Image saved:\n{filename}")
            except Exception as e:
                self.log(f"✗ Save failed: {e}")
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def log(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


def main():
    """Main entry point."""
    print("=" * 70)
    print("JetsonSky Professional GUI")
    print("=" * 70)
    print()
    print("Dependencies:")
    print(f"  NumPy: {'✓' if HAS_NUMPY else '✗ (required)'}")
    print(f"  PIL/Pillow: {'✓' if HAS_PIL else '✗ (required)'}")
    print(f"  OpenCV: {'✓' if HAS_OPENCV else '✗ (required)'}")
    print(f"  Phase 2 Filters: {'✓' if HAS_FILTERS else '✗'}")
    print(f"  Real Camera Support: {'✓' if HAS_REAL_CAMERA else '✗'}")
    print()

    if not HAS_PIL:
        print("⚠ Install Pillow: pip install Pillow")
    if not HAS_OPENCV:
        print("⚠ Install OpenCV: pip install opencv-python")
    print()

    root = tk.Tk()
    app = ProfessionalJetsonSkyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
