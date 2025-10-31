#!/usr/bin/env python3
"""
JetsonSky Enhanced GUI Demo with Real-Time Frame Display

Features:
- Real-time camera frame display
- Phase 2 filter pipeline integration
- Visual preview of filtered images
- Save filtered images
- Performance statistics

Usage:
    python3 gui_demo_enhanced.py
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime

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

# Try to import required packages
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠ NumPy not available - limited functionality")

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠ PIL/Pillow not available - image display disabled")
    print("  Install with: pip install Pillow")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠ OpenCV not available - filters disabled")

# Import Phase 2 filters
try:
    from filters import (
        FilterPipeline,
        HotPixelFilter,
        DenoiseKNNFilter,
        SharpenFilter,
        CLAHEFilter,
        SaturationFilter,
        FlipFilter,
    )
    HAS_FILTERS = True
except ImportError:
    HAS_FILTERS = False
    print("⚠ Phase 2 filters not available")

from demos.camera_simulator import create_simulated_camera


class EnhancedJetsonSkyGUI:
    """Enhanced GUI with real-time frame display and filters."""

    def __init__(self, root):
        """Initialize enhanced GUI."""
        self.root = root
        self.root.title("JetsonSky Enhanced Demo - Real-Time Filtering")
        self.root.geometry("1200x800")

        # Application state
        self.app = AppState()
        self.camera = None
        self.acquisition_thread = None
        self.current_frame = None
        self.photo_image = None

        # Filter pipeline
        self.pipeline = FilterPipeline() if HAS_FILTERS else None

        # Statistics
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()

        # Build UI
        self.build_ui()

        # Initialize filters
        if HAS_FILTERS:
            self.setup_filters()

    def setup_filters(self):
        """Setup initial filter pipeline."""
        # Add all available filters (disabled by default)
        if HAS_OPENCV:
            self.pipeline.add_filter(FlipFilter(vertical=False, horizontal=False, enabled=False))
            self.pipeline.add_filter(HotPixelFilter(threshold=0.9, enabled=False))
            self.pipeline.add_filter(DenoiseKNNFilter(strength=0.3, enabled=False))
            self.pipeline.add_filter(SharpenFilter(amount=1.5, enabled=False))
            self.pipeline.add_filter(CLAHEFilter(clip_limit=2.0, enabled=False))
            self.pipeline.add_filter(SaturationFilter(saturation=1.3, enabled=False))

    def build_ui(self):
        """Build enhanced user interface."""
        # Main container with 3 columns
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT PANEL - Camera & Settings
        left_panel = tk.LabelFrame(main, text="Camera & Settings", padx=10, pady=10)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, rowspan=2)

        # Camera selection
        tk.Label(left_panel, text="Camera Model:").grid(row=0, column=0, sticky="w", pady=5)
        self.camera_var = tk.StringVar()
        cameras = get_supported_cameras()
        self.camera_combo = ttk.Combobox(left_panel, textvariable=self.camera_var, width=25)
        self.camera_combo['values'] = cameras
        self.camera_combo.current(cameras.index("ZWO ASI178MC"))
        self.camera_combo.grid(row=0, column=1, pady=5)

        tk.Button(
            left_panel,
            text="🎥 Load Camera",
            command=self.load_camera,
            bg="#3498db",
            fg="white",
            width=20
        ).grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # Exposition
        tk.Label(left_panel, text="Exposition (µs):").grid(row=2, column=0, sticky="w", pady=5)
        self.exposition_var = tk.IntVar(value=DEFAULT_EXPOSITION)
        tk.Scale(
            left_panel,
            from_=100,
            to=50000,
            orient=tk.HORIZONTAL,
            variable=self.exposition_var,
            command=self.update_exposition,
            length=200
        ).grid(row=2, column=1, sticky="ew")
        self.exposition_label = tk.Label(left_panel, text=f"{DEFAULT_EXPOSITION} µs")
        self.exposition_label.grid(row=3, column=1, sticky="w")

        # Gain
        tk.Label(left_panel, text="Gain:").grid(row=4, column=0, sticky="w", pady=5)
        self.gain_var = tk.IntVar(value=DEFAULT_GAIN)
        tk.Scale(
            left_panel,
            from_=0,
            to=400,
            orient=tk.HORIZONTAL,
            variable=self.gain_var,
            command=self.update_gain,
            length=200
        ).grid(row=4, column=1, sticky="ew")
        self.gain_label = tk.Label(left_panel, text=f"{DEFAULT_GAIN}")
        self.gain_label.grid(row=5, column=1, sticky="w")

        # Binning
        tk.Label(left_panel, text="Binning:").grid(row=6, column=0, sticky="w", pady=5)
        self.binning_var = tk.IntVar(value=1)
        ttk.Radiobutton(left_panel, text="1x1", variable=self.binning_var, value=1,
                       command=self.update_binning).grid(row=6, column=1, sticky="w")
        ttk.Radiobutton(left_panel, text="2x2", variable=self.binning_var, value=2,
                       command=self.update_binning).grid(row=7, column=1, sticky="w")

        # Control buttons
        tk.Button(
            left_panel,
            text="▶ Start Preview",
            command=self.start_acquisition,
            bg="#27ae60",
            fg="white",
            height=2,
            width=20
        ).grid(row=8, column=0, columnspan=2, pady=5, sticky="ew")

        tk.Button(
            left_panel,
            text="⏹ Stop Preview",
            command=self.stop_acquisition,
            bg="#e74c3c",
            fg="white",
            height=2,
            width=20
        ).grid(row=9, column=0, columnspan=2, pady=5, sticky="ew")

        tk.Button(
            left_panel,
            text="💾 Save Image",
            command=self.save_image,
            bg="#9b59b6",
            fg="white",
            height=2,
            width=20
        ).grid(row=10, column=0, columnspan=2, pady=5, sticky="ew")

        # CENTER PANEL - Image Display
        center_panel = tk.LabelFrame(main, text="Camera Preview", padx=5, pady=5)
        center_panel.grid(row=0, column=1, sticky="nsew", padx=5)

        # Canvas for image display
        self.canvas = tk.Canvas(center_panel, width=640, height=480, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status label below canvas
        self.image_status = tk.Label(center_panel, text="No image", font=("Arial", 9))
        self.image_status.pack(pady=5)

        # Statistics panel
        stats_frame = tk.Frame(center_panel)
        stats_frame.pack(fill=tk.X, pady=5)

        self.fps_label = tk.Label(stats_frame, text="FPS: 0.0", font=("Arial", 9))
        self.fps_label.pack(side=tk.LEFT, padx=10)

        self.frame_label = tk.Label(stats_frame, text="Frame: 0", font=("Arial", 9))
        self.frame_label.pack(side=tk.LEFT, padx=10)

        # RIGHT PANEL - Filters
        right_panel = tk.LabelFrame(main, text="Phase 2 Filters", padx=10, pady=10)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=5, rowspan=2)

        if HAS_FILTERS and HAS_OPENCV:
            # Flip filter
            tk.Label(right_panel, text="Transform:", font=("Arial", 10, "bold")).grid(
                row=0, column=0, sticky="w", pady=(0,5))

            self.flip_v_var = tk.BooleanVar()
            tk.Checkbutton(right_panel, text="Flip Vertical", variable=self.flip_v_var,
                          command=self.update_flip_filter).grid(row=1, column=0, sticky="w")

            self.flip_h_var = tk.BooleanVar()
            tk.Checkbutton(right_panel, text="Flip Horizontal", variable=self.flip_h_var,
                          command=self.update_flip_filter).grid(row=2, column=0, sticky="w")

            # Hot pixel filter
            tk.Label(right_panel, text="Cleanup:", font=("Arial", 10, "bold")).grid(
                row=3, column=0, sticky="w", pady=(10,5))

            self.filter_hotpixel_var = tk.BooleanVar()
            tk.Checkbutton(right_panel, text="Hot Pixel Removal",
                          variable=self.filter_hotpixel_var,
                          command=self.update_filter_pipeline).grid(row=4, column=0, sticky="w")

            # Denoise filter
            tk.Label(right_panel, text="Enhancement:", font=("Arial", 10, "bold")).grid(
                row=5, column=0, sticky="w", pady=(10,5))

            self.filter_denoise_var = tk.BooleanVar()
            tk.Checkbutton(right_panel, text="Denoise (KNN)",
                          variable=self.filter_denoise_var,
                          command=self.update_filter_pipeline).grid(row=6, column=0, sticky="w")

            self.filter_sharpen_var = tk.BooleanVar()
            tk.Checkbutton(right_panel, text="Sharpen",
                          variable=self.filter_sharpen_var,
                          command=self.update_filter_pipeline).grid(row=7, column=0, sticky="w")

            self.filter_clahe_var = tk.BooleanVar()
            tk.Checkbutton(right_panel, text="CLAHE Contrast",
                          variable=self.filter_clahe_var,
                          command=self.update_filter_pipeline).grid(row=8, column=0, sticky="w")

            self.filter_saturation_var = tk.BooleanVar()
            tk.Checkbutton(right_panel, text="Saturation Boost",
                          variable=self.filter_saturation_var,
                          command=self.update_filter_pipeline).grid(row=9, column=0, sticky="w")

            # Filter info
            tk.Label(right_panel, text="Pipeline Info:", font=("Arial", 10, "bold")).grid(
                row=10, column=0, sticky="w", pady=(10,5))

            self.filter_info = tk.Text(right_panel, height=8, width=25, state=tk.DISABLED,
                                      font=("Courier", 8))
            self.filter_info.grid(row=11, column=0, sticky="nsew", pady=5)
        else:
            tk.Label(right_panel, text="Filters require:\n- NumPy\n- OpenCV\n- Phase 2 filters\n\n"
                    "Install with:\npip install numpy\npip install opencv-python",
                    justify=tk.LEFT, fg="red").grid(row=0, column=0, padx=10, pady=10)

        # BOTTOM PANEL - Status Log
        status_panel = tk.LabelFrame(main, text="Status Log", padx=5, pady=5)
        status_panel.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.status_text = tk.Text(status_panel, height=8, width=60, state=tk.DISABLED,
                                   font=("Courier", 9))
        self.status_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for status
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)

        # Configure grid weights
        main.columnconfigure(1, weight=3)  # Center panel gets more space
        main.columnconfigure(0, weight=1)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=1)

        # Initial status
        self.log_status("✓ GUI initialized")
        if not HAS_PIL:
            self.log_status("✗ PIL not available - install Pillow for image display")
        if not HAS_OPENCV:
            self.log_status("✗ OpenCV not available - filters disabled")
        if HAS_FILTERS:
            self.log_status(f"✓ Filter pipeline ready ({len(self.pipeline)} filters)")

    def update_flip_filter(self):
        """Update flip filter settings."""
        if self.pipeline:
            flip_filter = self.pipeline.get_filter("FlipFilter")
            if flip_filter:
                flip_filter.set_vertical(self.flip_v_var.get())
                flip_filter.set_horizontal(self.flip_h_var.get())
                # Enable if either flip is active
                if self.flip_v_var.get() or self.flip_h_var.get():
                    flip_filter.enable()
                else:
                    flip_filter.disable()
                self.update_filter_info()

    def update_filter_pipeline(self):
        """Update filter pipeline based on checkboxes."""
        if not self.pipeline:
            return

        # Update each filter's enabled state
        filter_map = {
            "HotPixelFilter": self.filter_hotpixel_var.get(),
            "DenoiseKNNFilter": self.filter_denoise_var.get(),
            "SharpenFilter": self.filter_sharpen_var.get(),
            "CLAHEFilter": self.filter_clahe_var.get(),
            "SaturationFilter": self.filter_saturation_var.get(),
        }

        for filter_name, enabled in filter_map.items():
            if enabled:
                self.pipeline.enable_filter(filter_name)
            else:
                self.pipeline.disable_filter(filter_name)

        self.update_filter_info()
        self.log_status(f"Pipeline updated: {self.pipeline.get_enabled_count()}/{len(self.pipeline)} filters active")

    def update_filter_info(self):
        """Update filter info display."""
        if not self.pipeline:
            return

        info = f"Pipeline Status:\n"
        info += f"Total: {len(self.pipeline)} filters\n"
        info += f"Active: {self.pipeline.get_enabled_count()}\n\n"

        for f in self.pipeline.filters:
            status = "✓" if f.is_enabled() else "✗"
            info += f"{status} {f.get_name()}\n"

        self.filter_info.config(state=tk.NORMAL)
        self.filter_info.delete(1.0, tk.END)
        self.filter_info.insert(1.0, info)
        self.filter_info.config(state=tk.DISABLED)

    def load_camera(self):
        """Load selected camera."""
        camera_model = self.camera_var.get()

        try:
            self.app.camera_config = get_camera_config(camera_model)
            self.app.camera_connected = True
            self.app.processing.exposition = self.exposition_var.get()
            self.app.processing.gain = self.gain_var.get()
            self.app.resolution_mode = 1
            self.app.binning_mode = self.binning_var.get()

            resolution = self.app.get_current_resolution()

            self.log_status(f"✓ Camera loaded: {camera_model}")
            self.log_status(f"  Resolution: {resolution[0]}x{resolution[1]}")
            self.log_status(f"  Sensor: {self.app.camera_config.sensor_factor} ({self.app.camera_config.sensor_bits}-bit)")

        except Exception as e:
            self.log_status(f"✗ Failed to load camera: {e}")
            messagebox.showerror("Error", f"Failed to load camera: {e}")

    def update_exposition(self, value):
        """Update exposition."""
        self.app.processing.exposition = int(float(value))
        self.exposition_label.config(text=f"{self.app.processing.exposition} µs")
        if self.camera:
            self.camera.set_exposition(self.app.processing.exposition)

    def update_gain(self, value):
        """Update gain."""
        self.app.processing.gain = int(float(value))
        self.gain_label.config(text=f"{self.app.processing.gain}")
        if self.camera:
            self.camera.set_gain(self.app.processing.gain)

    def update_binning(self):
        """Update binning mode."""
        self.app.binning_mode = self.binning_var.get()
        if self.app.camera_config:
            resolution = self.app.get_current_resolution()
            self.log_status(f"Binning changed to {self.app.binning_mode}x{self.app.binning_mode}: {resolution[0]}x{resolution[1]}")

    def start_acquisition(self):
        """Start acquisition and preview."""
        if not HAS_NUMPY:
            messagebox.showerror("Error", "NumPy required for acquisition")
            return

        if self.app.camera_config is None:
            messagebox.showwarning("Warning", "Please load a camera first!")
            return

        if self.app.acquisition_running:
            return

        resolution = self.app.get_current_resolution()

        self.camera = create_simulated_camera(
            self.app.camera_config.model,
            resolution,
            self.app.camera_config.sensor_bits
        )

        self.camera.set_exposition(self.app.processing.exposition)
        self.camera.set_gain(self.app.processing.gain)
        self.camera.start_capture()

        self.app.acquisition_running = True
        self.frame_count = 0
        self.last_time = time.time()

        self.log_status("▶ Acquisition started")

        self.acquisition_thread = threading.Thread(target=self.acquisition_loop, daemon=True)
        self.acquisition_thread.start()

    def stop_acquisition(self):
        """Stop acquisition."""
        if not self.app.acquisition_running:
            return

        self.app.acquisition_running = False

        if self.camera:
            self.camera.stop_capture()

        self.log_status("⏹ Acquisition stopped")

    def acquisition_loop(self):
        """Main acquisition loop."""
        while self.app.acquisition_running:
            if self.camera:
                frame = self.camera.capture_frame()

                if frame is not None and HAS_NUMPY:
                    self.frame_count += 1

                    # Calculate FPS
                    current_time = time.time()
                    if current_time - self.last_time > 0:
                        self.fps = 1.0 / (current_time - self.last_time)
                    self.last_time = current_time

                    # Apply filters if available
                    if self.pipeline and HAS_OPENCV:
                        filtered_frame = self.pipeline.apply(frame)
                    else:
                        filtered_frame = frame

                    # Store current frame
                    self.current_frame = filtered_frame

                    # Update display
                    self.root.after(0, self.update_display, filtered_frame)

            time.sleep(0.033)  # ~30 FPS max

    def update_display(self, frame):
        """Update image display."""
        if not HAS_PIL or frame is None:
            return

        try:
            # Convert numpy array to PIL Image
            if len(frame.shape) == 2:
                # Grayscale
                if frame.dtype == np.uint16:
                    frame_8bit = (frame / 256).astype(np.uint8)
                else:
                    frame_8bit = frame
                pil_image = Image.fromarray(frame_8bit, mode='L')
            else:
                # Color (assume BGR from camera)
                if frame.dtype == np.uint16:
                    frame_8bit = (frame / 256).astype(np.uint8)
                else:
                    frame_8bit = frame
                # Convert BGR to RGB for PIL
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

            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(pil_image)

            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.photo_image
            )

            # Update statistics
            mean_val = frame.mean()
            max_val = frame.max()
            min_val = frame.min()

            self.image_status.config(
                text=f"Frame {self.frame_count} | {frame.shape[1]}x{frame.shape[0]} | "
                     f"Min:{min_val:.0f} Mean:{mean_val:.0f} Max:{max_val:.0f}"
            )

            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.frame_label.config(text=f"Frame: {self.frame_count}")

        except Exception as e:
            self.log_status(f"✗ Display error: {e}")

    def save_image(self):
        """Save current frame to disk."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to save!")
            return

        if not HAS_OPENCV:
            messagebox.showerror("Error", "OpenCV required to save images")
            return

        # Ask for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"jetsonsky_frame_{timestamp}.png"

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if filename:
            try:
                # Save with OpenCV
                if len(self.current_frame.shape) == 2:
                    # Grayscale
                    cv2.imwrite(filename, self.current_frame)
                else:
                    # Color (already in BGR format for OpenCV)
                    cv2.imwrite(filename, self.current_frame)

                self.log_status(f"✓ Saved: {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Image saved:\n{filename}")
            except Exception as e:
                self.log_status(f"✗ Save failed: {e}")
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def log_status(self, message):
        """Add message to status log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, log_message)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)


def main():
    """Main entry point."""
    print("=" * 60)
    print("JetsonSky Enhanced GUI Demo")
    print("=" * 60)
    print()
    print("Dependencies:")
    print(f"  NumPy: {'✓' if HAS_NUMPY else '✗ (required)'}")
    print(f"  PIL/Pillow: {'✓' if HAS_PIL else '✗ (required for display)'}")
    print(f"  OpenCV: {'✓' if HAS_OPENCV else '✗ (required for filters)'}")
    print(f"  Phase 2 Filters: {'✓' if HAS_FILTERS else '✗'}")
    print()

    if not HAS_PIL:
        print("⚠ Install Pillow for image display:")
        print("  pip install Pillow")
        print()

    root = tk.Tk()
    app = EnhancedJetsonSkyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
