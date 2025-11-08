# JetsonSky Installation Guide

## Quick Start

### For Windows Users:
1. **Check dependencies**: `python check_dependencies.py`
2. **Install missing packages**: See manual installation below
3. **Run JetsonSky**: `run_jetsonsky.bat` or `cd JetsonSky && python JetsonSky_Linux_Windows_V53_07RC.py`

### For Linux Users:
1. **Check dependencies**: `python3 check_dependencies.py`
2. **Install missing packages**: See manual installation below  
3. **Run JetsonSky**: `cd JetsonSky && python3 JetsonSky_Linux_Windows_V53_07RC.py`

## System Requirements

- **Python**: 3.12.x (recommended) or 3.10+
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA**: 11.x or 12.x
- **RAM**: 8GB minimum, 16GB recommended
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)

## Step-by-Step Installation

### 1. Install Python 3.12

**Windows**: Download from https://www.python.org/downloads/
**Linux**: `sudo apt install python3.12 python3-pip`

### 2. Install CUDA Toolkit

Download from https://developer.nvidia.com/cuda-downloads

Verify: `nvcc --version`

### 3. Clone Repository

```bash
git clone https://github.com/gyillikci/JETSONSKY.git
cd JETSONSKY
```

### 4. Create Virtual Environment (Recommended)

**Windows**:
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Install Dependencies

#### Install PyTorch (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Install CuPy (CUDA 12.x):
```bash
pip install cupy-cuda12x
```

#### Install Other Packages:
```bash
pip install numpy pillow opencv-python ultralytics psutil matplotlib
```

#### Platform-Specific:
- **Windows**: `pip install keyboard`
- **Linux**: `pip install pynput`

### 6. Verify Installation

```bash
python check_dependencies.py
```

## Supported Cameras

- ZWO ASI120/178/183/224/290/294/385/462/482/485/533/585/662/676/678/715/1600 series
- Requires ZWO camera drivers

## Running Without Camera

JetsonSky works in **Video Treatment Mode** without a connected camera for testing filters and processing existing videos.

## Troubleshooting

### "CUDA not available"
- Verify: `nvcc --version`
- Reinstall PyTorch with CUDA support

### "CuPy not found"  
- Match CuPy to your CUDA version
- CUDA 12.x: `pip install cupy-cuda12x`
- CUDA 11.x: `pip install cupy-cuda11x`

### "Module not found"
- Activate virtual environment
- Reinstall: `pip install -r requirements.txt`

## Features

- Real-time GPU-accelerated image processing
- AI crater and satellite detection (YOLOv8)
- Multiple stabilization methods (Template, Optical Flow, Hybrid)
- 20+ image processing filters
- SER file format support

## License

Free for personal/non-commercial use.
Copyright Alain Paillou 2018-2025

