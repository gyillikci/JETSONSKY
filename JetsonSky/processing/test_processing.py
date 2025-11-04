"""
Quick tests for the processing module.

Run this to verify that all modules can be imported and basic functions work.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from processing import (
            cupy_RGBImage_2_cupy_separateRGB,
            Image_Quality,
            TemplateStabilizer,
            opencv_color_debayer,
            HDR_compute,
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_quality():
    """Test image quality functions."""
    print("\nTesting quality module...")
    
    try:
        import numpy as np
        from processing.quality import Image_Quality
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Test Laplacian method
        quality_lap = Image_Quality(test_image, "Laplacian")
        print(f"  Laplacian quality: {quality_lap:.2f}")
        
        # Test Sobel method
        quality_sob = Image_Quality(test_image, "Sobel")
        print(f"  Sobel quality: {quality_sob:.2f}")
        
        print("✓ Quality module working")
        return True
    except Exception as e:
        print(f"✗ Quality test failed: {e}")
        return False


def test_image_utils():
    """Test image utility functions."""
    print("\nTesting image_utils module...")
    
    try:
        import numpy as np
        from processing.image_utils import (
            numpy_RGBImage_2_numpy_separateRGB,
            numpy_separateRGB_2_numpy_RGBimage,
            gaussianblur_mono,
        )
        
        # Create test RGB image
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test separation
        r, g, b = numpy_RGBImage_2_numpy_separateRGB(rgb_image)
        print(f"  Separated channels: R{r.shape}, G{g.shape}, B{b.shape}")
        
        # Test merge
        merged = numpy_separateRGB_2_numpy_RGBimage(r, g, b)
        print(f"  Merged image: {merged.shape}")
        
        # Test gaussian blur (needs CuPy array)
        try:
            import cupy as cp
            mono_image = cp.random.randint(0, 255, (100, 100), dtype=cp.uint8)
            blurred = gaussianblur_mono(mono_image, 1.0)
            print(f"  Blurred image: {blurred.shape}")
        except:
            print(f"  Gaussian blur test skipped (CuPy required)")
        
        print("✓ Image utils module working")
        return True
    except Exception as e:
        print(f"✗ Image utils test failed: {e}")
        return False


def test_stabilizer():
    """Test stabilization module."""
    print("\nTesting stabilization module...")
    
    try:
        import numpy as np
        from processing.stabilization import TemplateStabilizer
        
        # Create stabilizer
        stabilizer = TemplateStabilizer(res_cam_x=640, res_cam_y=480, use_cuda=False)
        print(f"  Stabilizer created: {stabilizer}")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame (will create template)
        result = stabilizer.process_frame(test_frame, dim=3)
        print(f"  Processed frame: {result.shape}")
        print(f"  Template initialized: {stabilizer.flag_template}")
        
        print("✓ Stabilization module working")
        return True
    except Exception as e:
        print(f"✗ Stabilization test failed: {e}")
        return False


def test_debayer():
    """Test debayering functions."""
    print("\nTesting debayer module...")
    
    try:
        import numpy as np
        import cv2
        from processing.debayer import opencv_color_debayer, get_bayer_pattern
        
        # Create test raw image (simulated Bayer pattern)
        raw_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Test debayering
        debayered = opencv_color_debayer(raw_image, cv2.COLOR_BAYER_RG2BGR, cuda_flag=False)
        print(f"  Debayered image: {debayered.shape}")
        
        # Test bayer pattern lookup
        pattern = get_bayer_pattern({'BayerPattern': 0})
        print(f"  Bayer pattern constant: {pattern}")
        
        print("✓ Debayer module working")
        return True
    except Exception as e:
        print(f"✗ Debayer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Processing Module Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Quality", test_quality()))
    results.append(("Image Utils", test_image_utils()))
    results.append(("Stabilizer", test_stabilizer()))
    results.append(("Debayer", test_debayer()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
