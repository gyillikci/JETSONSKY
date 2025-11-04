"""
Template Matching Performance Stress Test

Compares performance of:
1. OpenCV CPU template matching
2. OpenCV CUDA template matching
3. Custom CUDA cross-correlation kernel
"""

import cv2
import numpy as np
import time
import cupy as cp
from cuda_kernels import cross_correlation_kernel

def generate_test_data(image_size=(1920, 1080), template_size=(160, 90)):
    """Generate synthetic test image and template"""
    # Create random noise image
    image = np.random.randint(0, 255, image_size, dtype=np.uint8)
    
    # Add some features (circles and rectangles) to make it more realistic
    for _ in range(20):
        x = np.random.randint(50, image_size[0] - 50)
        y = np.random.randint(50, image_size[1] - 50)
        radius = np.random.randint(10, 30)
        cv2.circle(image, (x, y), radius, int(np.random.randint(100, 255)), -1)
    
    # Extract template from image
    tx = image_size[0] // 2 - template_size[0] // 2
    ty = image_size[1] // 2 - template_size[1] // 2
    template = image[ty:ty+template_size[1], tx:tx+template_size[0]].copy()
    
    return image, template

def test_opencv_cpu(image, template, roi_bounds, iterations=100):
    """Test OpenCV CPU template matching"""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return times

def test_opencv_cuda(image, template, roi_bounds, iterations=100):
    """Test OpenCV CUDA template matching"""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Check if CUDA is available
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA not available for OpenCV")
        return None
    
    # Create GPU mats
    gsrc = cv2.cuda_GpuMat()
    gtmpl = cv2.cuda_GpuMat()
    gresult = cv2.cuda_GpuMat()
    gtmpl.upload(template)
    matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        gsrc.upload(roi)
        gresult = matcher.match(gsrc, gtmpl)
        result = gresult.download()
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return times

def test_custom_cuda(image, template, roi_bounds, iterations=100):
    """Test custom CUDA cross-correlation kernel"""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
    result_width = roi_width - template.shape[1] + 1
    result_height = roi_height - template.shape[0] + 1
    
    # Upload to GPU once
    image_gpu = cp.asarray(image, dtype=cp.uint8)
    template_gpu = cp.asarray(template, dtype=cp.uint8)
    result_gpu = cp.zeros((result_height, result_width), dtype=cp.float32)
    
    # Configure kernel
    threads_per_block = (16, 16)
    blocks_x = (result_width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (result_height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        
        # Launch kernel
        cross_correlation_kernel(
            blocks_per_grid, threads_per_block,
            (image_gpu, template_gpu, result_gpu,
             image.shape[1], image.shape[0],
             template.shape[1], template.shape[0],
             roi_x1, roi_y1)
        )
        
        # Download result and find max
        result = cp.asnumpy(result_gpu)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return times

def print_statistics(name, times):
    """Print performance statistics"""
    if times is None:
        print(f"\n{name}: NOT AVAILABLE")
        return
    
    times = np.array(times)
    print(f"\n{name}:")
    print(f"  Iterations: {len(times)}")
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")
    print(f"  95th percentile: {np.percentile(times, 95):.3f} ms")

def run_stress_test():
    """Run comprehensive stress test"""
    print("=" * 70)
    print("TEMPLATE MATCHING PERFORMANCE STRESS TEST")
    print("=" * 70)
    
    # Test configurations
    image_sizes = [(1920, 1080), (1280, 720)]
    template_sizes = [(160, 90), (80, 45)]
    iterations = 100
    
    for img_size in image_sizes:
        for tpl_size in template_sizes:
            print(f"\n{'='*70}")
            print(f"Configuration: Image {img_size[0]}x{img_size[1]}, Template {tpl_size[0]}x{tpl_size[1]}")
            print(f"{'='*70}")
            
            # Generate test data
            image, template = generate_test_data(img_size, tpl_size)
            
            # Define ROI (1.25x template size around center)
            padding = max(tpl_size[0], tpl_size[1]) // 8
            center_x = img_size[0] // 2
            center_y = img_size[1] // 2
            roi_x1 = max(0, center_x - padding - tpl_size[0] // 2)
            roi_y1 = max(0, center_y - padding - tpl_size[1] // 2)
            roi_x2 = min(img_size[0], center_x + padding + tpl_size[0] // 2)
            roi_y2 = min(img_size[1], center_y + padding + tpl_size[1] // 2)
            roi_bounds = (roi_x1, roi_y1, roi_x2, roi_y2)
            
            roi_width = roi_x2 - roi_x1
            roi_height = roi_y2 - roi_y1
            print(f"ROI size: {roi_width}x{roi_height} pixels")
            print(f"Search area: {roi_width * roi_height:,} pixels")
            
            # Test 1: OpenCV CPU
            print("\nTesting OpenCV CPU...")
            cpu_times = test_opencv_cpu(image, template, roi_bounds, iterations)
            print_statistics("OpenCV CPU", cpu_times)
            
            # Test 2: OpenCV CUDA
            print("\nTesting OpenCV CUDA...")
            try:
                opencv_cuda_times = test_opencv_cuda(image, template, roi_bounds, iterations)
                print_statistics("OpenCV CUDA", opencv_cuda_times)
            except Exception as e:
                print(f"OpenCV CUDA failed: {e}")
                opencv_cuda_times = None
            
            # Test 3: Custom CUDA
            print("\nTesting Custom CUDA Kernel...")
            try:
                custom_cuda_times = test_custom_cuda(image, template, roi_bounds, iterations)
                print_statistics("Custom CUDA", custom_cuda_times)
            except Exception as e:
                print(f"Custom CUDA failed: {e}")
                custom_cuda_times = None
            
            # Comparison
            print(f"\n{'='*70}")
            print("PERFORMANCE COMPARISON:")
            print(f"{'='*70}")
            
            if cpu_times:
                cpu_mean = np.mean(cpu_times)
                print(f"OpenCV CPU:        {cpu_mean:.3f} ms (baseline)")
                
                if opencv_cuda_times:
                    cuda_mean = np.mean(opencv_cuda_times)
                    speedup = cpu_mean / cuda_mean
                    print(f"OpenCV CUDA:       {cuda_mean:.3f} ms ({speedup:.2f}x faster)")
                
                if custom_cuda_times:
                    custom_mean = np.mean(custom_cuda_times)
                    speedup = cpu_mean / custom_mean
                    print(f"Custom CUDA:       {custom_mean:.3f} ms ({speedup:.2f}x faster)")
                
                if opencv_cuda_times and custom_cuda_times:
                    custom_vs_opencv = np.mean(opencv_cuda_times) / np.mean(custom_cuda_times)
                    print(f"\nCustom vs OpenCV CUDA: {custom_vs_opencv:.2f}x")

if __name__ == "__main__":
    run_stress_test()
