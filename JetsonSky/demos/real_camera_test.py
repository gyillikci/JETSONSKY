#!/usr/bin/env python3
"""
Real Camera Detection and Test Script
Attempts to connect to actual ZWO ASI camera hardware
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("JetsonSky - Real Camera Detection")
print("=" * 60)
print()

# Check for required libraries
try:
    import numpy as np
    print("✓ NumPy available")
    HAS_NUMPY = True
except ImportError:
    print("✗ NumPy not available")
    HAS_NUMPY = False

try:
    import cv2
    print("✓ OpenCV available")
    HAS_OPENCV = True
except ImportError:
    print("✗ OpenCV not available")
    HAS_OPENCV = False

# Try to import ZWO ASI library
print()
print("Checking for ZWO ASI SDK...")
try:
    # Try the cupy version first
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'zwoasi_cupy'))
    import zwoasi_cupy as asi
    print("✓ Found zwoasi_cupy library")
    HAS_ASI = True
except ImportError:
    try:
        # Try standard zwoasi
        import zwoasi as asi
        print("✓ Found zwoasi library")
        HAS_ASI = True
    except ImportError:
        print("✗ ZWO ASI library not available")
        print("  The demo uses simulated cameras only")
        HAS_ASI = False

print()

if HAS_ASI:
    print("Attempting to detect ZWO ASI cameras...")
    print()
    
    try:
        # Initialize ASI library
        # Look for library files
        lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Lib')
        
        if sys.platform == "win32":
            asi_lib = os.path.join(lib_path, "ASICamera2.dll")
        elif sys.platform == "linux":
            asi_lib = os.path.join(lib_path, "libASICamera2.so.1.27")
        else:
            asi_lib = None
            
        if asi_lib and os.path.exists(asi_lib):
            print(f"Found ASI library: {asi_lib}")
            asi.init(asi_lib)
        else:
            print(f"ASI library not found at: {asi_lib}")
            print("Attempting to use system-installed library...")
            asi.init()
        
        # Get number of cameras
        num_cameras = asi.get_num_cameras()
        print(f"Number of connected cameras: {num_cameras}")
        print()
        
        if num_cameras == 0:
            print("⚠ No ZWO ASI cameras detected!")
            print()
            print("Possible reasons:")
            print("  1. Camera not physically connected")
            print("  2. Camera drivers not installed")
            print("  3. USB connection issue")
            print("  4. Camera in use by another application")
            print()
            print("Please check your camera connection and try again.")
        else:
            # List all detected cameras
            for i in range(num_cameras):
                camera_info = asi._get_camera_property(i)
                print(f"Camera {i}:")
                print(f"  Name: {camera_info['Name']}")
                print(f"  Camera ID: {camera_info['CameraID']}")
                print(f"  Max Height: {camera_info['MaxHeight']}")
                print(f"  Max Width: {camera_info['MaxWidth']}")
                print(f"  Is Color: {camera_info['IsColorCam']}")
                print(f"  Bayer Pattern: {camera_info['BayerPattern']}")
                print(f"  Pixel Size: {camera_info['PixelSize']} µm")
                print(f"  Bit Depth: {camera_info.get('BitDepth', 'N/A')}")
                print()
            
            # Try to open first camera and capture a test frame
            print("Attempting to open first camera...")
            camera = asi.Camera(0)
            camera_info = camera.get_camera_property()
            print(f"✓ Opened: {camera_info['Name']}")
            print()
            
            # Get camera controls
            print("Available camera controls:")
            controls = camera.get_controls()
            for ctrl_name, ctrl_info in controls.items():
                print(f"  {ctrl_name}: Min={ctrl_info['MinValue']}, Max={ctrl_info['MaxValue']}, Default={ctrl_info['DefaultValue']}")
            print()
            
            # Set basic parameters
            print("Configuring camera for test capture...")
            camera.set_control_value(asi.ASI_GAIN, 100)
            camera.set_control_value(asi.ASI_EXPOSURE, 10000)  # 10ms
            camera.set_control_value(asi.ASI_WB_B, 90)
            camera.set_control_value(asi.ASI_WB_R, 48)
            camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 40)
            
            # Set ROI to full resolution
            camera.set_roi(
                width=camera_info['MaxWidth'],
                height=camera_info['MaxHeight'],
                bins=1,
                image_type=asi.ASI_IMG_RAW8
            )
            
            print("Starting video capture...")
            camera.start_video_capture()
            
            # Capture a few test frames
            print("Capturing test frames...")
            for i in range(5):
                try:
                    frame = camera.capture_video_frame(timeout=5000)
                    if frame is not None:
                        if HAS_NUMPY:
                            frame_array = np.frombuffer(frame, dtype=np.uint8)
                            frame_2d = frame_array.reshape((camera_info['MaxHeight'], camera_info['MaxWidth']))
                            print(f"  Frame {i+1}: Shape={frame_2d.shape}, Mean={frame_2d.mean():.1f}, Min={frame_2d.min()}, Max={frame_2d.max()}")
                            
                            # Save first frame if OpenCV available
                            if i == 0 and HAS_OPENCV:
                                output_file = "test_frame.png"
                                cv2.imwrite(output_file, frame_2d)
                                print(f"  ✓ Saved test frame to: {output_file}")
                        else:
                            print(f"  Frame {i+1}: Captured ({len(frame)} bytes)")
                    else:
                        print(f"  Frame {i+1}: Failed to capture")
                except Exception as e:
                    print(f"  Frame {i+1}: Error - {e}")
                
                time.sleep(0.1)
            
            print()
            print("Stopping capture...")
            camera.stop_video_capture()
            camera.close()
            print("✓ Camera test complete!")
            
    except Exception as e:
        print(f"✗ Error during camera detection: {e}")
        import traceback
        traceback.print_exc()

else:
    print("=" * 60)
    print("CAMERA SDK NOT AVAILABLE")
    print("=" * 60)
    print()
    print("To use real ZWO ASI cameras, you need:")
    print("  1. ZWO ASI SDK installed")
    print("  2. Python zwoasi library")
    print()
    print("The GUI demo currently uses simulated cameras only.")
    print("For real camera support, you need to run the main")
    print("JetsonSky application with CUDA/CuPy installed.")
    print()

print()
print("=" * 60)
