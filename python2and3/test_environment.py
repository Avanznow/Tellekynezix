#!/usr/bin/env python3
"""
Avatar Python 3 Environment Verification Script
Tests that all required dependencies for Tello and Avatar GUI are available
"""

import sys
import importlib.util

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def check_python_version():
    """Verify Python version is 3.x"""
    print(f"\n{Colors.BLUE}=== Python Version Check ==={Colors.NC}\n")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3:
        print(f"{Colors.RED}✗ ERROR: Python 3 required, but Python {version.major}.{version.minor} detected{Colors.NC}")
        return False
    
    print(f"{Colors.GREEN}✓ Python 3.{version.minor} detected{Colors.NC}")
    return True

def check_package(package_name, import_name=None, version_check=None):
    """
    Check if a package is installed
    
    Args:
        package_name: Display name of the package
        import_name: Actual import name (defaults to package_name)
        version_check: Optional function to get version string
    """
    if import_name is None:
        import_name = package_name.lower().replace('-', '_')
    
    try:
        module = importlib.import_module(import_name)
        
        # Try to get version
        version_str = ""
        if version_check:
            try:
                version_str = f" ({version_check(module)})"
            except:
                pass
        elif hasattr(module, '__version__'):
            version_str = f" ({module.__version__})"
        
        print(f"{Colors.GREEN}✓{Colors.NC} {package_name}{version_str}")
        return True
    except ImportError:
        print(f"{Colors.RED}✗{Colors.NC} {package_name} (not installed)")
        return False

def check_critical_packages():
    """Check all critical packages for Avatar + Tello"""
    print(f"\n{Colors.BLUE}=== Critical Dependencies ==={Colors.NC}\n")
    
    packages = [
        ("PySide6", "PySide6"),
        ("djitellopy", "djitellopy"),
        ("BrainFlow", "brainflow"),
        ("PyTorch", "torch"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
    ]
    
    all_good = True
    for display_name, import_name in packages:
        if not check_package(display_name, import_name):
            all_good = False
    
    return all_good

def check_optional_packages():
    """Check optional packages"""
    print(f"\n{Colors.BLUE}=== Optional Dependencies ==={Colors.NC}\n")
    
    packages = [
        ("OpenCV", "cv2"),
        ("Matplotlib", "matplotlib"),
        ("Pillow", "PIL"),
        ("SciPy", "scipy"),
        ("scikit-learn", "sklearn"),
    ]
    
    for display_name, import_name in packages:
        check_package(display_name, import_name)

def check_qt_backend():
    """Check Qt GUI backend"""
    print(f"\n{Colors.BLUE}=== Qt Backend Check ==={Colors.NC}\n")
    
    try:
        from PySide6.QtCore import QT_VERSION_STR, PYSIDE_VERSION_STR
        from PySide6.QtWidgets import QApplication
        
        print(f"{Colors.GREEN}✓{Colors.NC} Qt version: {QT_VERSION_STR}")
        print(f"{Colors.GREEN}✓{Colors.NC} PySide6 version: {PYSIDE_VERSION_STR}")
        
        # Test QApplication creation
        try:
            app = QApplication.instance() or QApplication([])
            print(f"{Colors.GREEN}✓{Colors.NC} QApplication can be created")
            return True
        except Exception as e:
            print(f"{Colors.YELLOW}⚠{Colors.NC}  Warning: QApplication creation issue: {e}")
            return False
            
    except ImportError as e:
        print(f"{Colors.RED}✗{Colors.NC} Qt backend not available: {e}")
        return False

def check_torch_cuda():
    """Check PyTorch CUDA availability"""
    print(f"\n{Colors.BLUE}=== PyTorch CUDA Check ==={Colors.NC}\n")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"{Colors.GREEN}✓{Colors.NC} CUDA available")
            print(f"{Colors.GREEN}✓{Colors.NC} GPU devices: {device_count}")
            print(f"{Colors.GREEN}✓{Colors.NC} Primary device: {device_name}")
        else:
            print(f"{Colors.YELLOW}ℹ{Colors.NC}  CUDA not available (CPU mode will be used)")
            print(f"{Colors.YELLOW}   {Colors.NC}This is OK - PyTorch will run on CPU")
        
        return True
    except ImportError:
        print(f"{Colors.RED}✗{Colors.NC} PyTorch not available")
        return False

def display_installation_instructions():
    """Show installation instructions for missing packages"""
    print(f"\n{Colors.YELLOW}=== Installation Instructions ==={Colors.NC}\n")
    print("To install missing packages, run:")
    print("")
    print("  pip3 install PySide6 djitellopy brainflow torch pandas numpy")
    print("")
    print("Optional packages:")
    print("  pip3 install opencv-python matplotlib pillow scipy scikit-learn")
    print("")

def main():
    """Main verification routine"""
    print(f"{Colors.BLUE}╔═══════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║                                                   ║{Colors.NC}")
    print(f"{Colors.BLUE}║   Avatar Python 3 Environment Verification       ║{Colors.NC}")
    print(f"{Colors.BLUE}║                                                   ║{Colors.NC}")
    print(f"{Colors.BLUE}╚═══════════════════════════════════════════════════╝{Colors.NC}")
    
    # Run checks
    python_ok = check_python_version()
    critical_ok = check_critical_packages()
    check_optional_packages()
    qt_ok = check_qt_backend()
    torch_ok = check_torch_cuda()
    
    # Summary
    print(f"\n{Colors.BLUE}=== Summary ==={Colors.NC}\n")
    
    if python_ok and critical_ok and qt_ok:
        print(f"{Colors.GREEN}✓ All critical dependencies are installed{Colors.NC}")
        print(f"{Colors.GREEN}✓ Avatar environment is ready{Colors.NC}")
        print("")
        print("You can now run:")
        print("  ./start_tello_environment.sh    # Tello + Avatar only")
        print("  ./start_avatar_full.sh          # Full environment (NAO + Tello)")
        return 0
    else:
        print(f"{Colors.RED}✗ Some critical dependencies are missing{Colors.NC}")
        display_installation_instructions()
        return 1

if __name__ == "__main__":
    sys.exit(main())
