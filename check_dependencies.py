#!/usr/bin/env python3
import sys, platform

print("="*70)
print("JetsonSky Dependency Checker".center(70))
print("="*70)
print(f"\nPython: {sys.version}")
print(f"OS: {platform.system()} {platform.release()}\n")
print("Checking dependencies...\n")

def check(name, imp=None):
    try:
        m = __import__(imp or name)
        v = getattr(m, "__version__", "OK")
        print(f" {name:20} {v}")
        return True
    except:
        print(f" {name:20} NOT INSTALLED")
        return False

r = {}
r["numpy"] = check("numpy")
r["PIL"] = check("Pillow", "PIL")
r["opencv"] = check("opencv-python", "cv2")
r["cupy"] = check("cupy", "cp")
r["torch"] = check("torch")
r["ultralytics"] = check("ultralytics")
r["psutil"] = check("psutil")
if platform.system() == "Windows":
    r["keyboard"] = check("keyboard")
else:
    r["pynput"] = check("pynput")

print("\n" + "="*70)
if all(r.values()):
    print(" All required dependencies installed!")
    sys.exit(0)
else:
    print(" Some dependencies missing. Run: install.bat")
    sys.exit(1)
