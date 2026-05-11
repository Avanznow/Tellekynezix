#!/bin/bash
# Tello/Avatar Environment Launcher (Python 3)
# Starts the main Avatar GUI with Tello drone support
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Avatar + Tello Environment Launcher    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}\n"

# 1. Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # Go up one level from python_2and3
echo -e "${GREEN}➤${NC} Project root: $PROJECT_ROOT"

# 2. Find Python 3
PYTHON3=""
if command -v python3 &> /dev/null; then
    PYTHON3=$(command -v python3)
elif command -v python &> /dev/null; then
    VERSION=$(python --version 2>&1 | grep -oP '3\.\d+\.\d+' || echo "")
    if [ ! -z "$VERSION" ]; then
        PYTHON3=$(command -v python)
    fi
fi

if [ -z "$PYTHON3" ]; then
    echo -e "${RED}✗ ERROR: Python 3 not found!${NC}"
    echo "Install instructions:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  macOS:         brew install python3"
    exit 1
fi

PYTHON_VERSION=$($PYTHON3 --version 2>&1)
echo -e "${GREEN}✓${NC} Found Python: $PYTHON_VERSION ($PYTHON3)"

# 3. Check for virtual environment
VENV_PATH="$PROJECT_ROOT/venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment found: $VENV_PATH"
    
    # Activate virtual environment
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        echo -e "${GREEN}✓${NC} Virtual environment activated"
    else
        echo -e "${YELLOW}⚠${NC}  Warning: venv activation script not found"
    fi
else
    echo -e "${YELLOW}⚠${NC}  No virtual environment found at $VENV_PATH"
    echo "   Using system Python 3"
    echo "   Recommended: Create venv with 'python3 -m venv $VENV_PATH'"
fi

# 4. Verify main.py exists
MAIN_FILE="$PROJECT_ROOT/GUI5.py"
if [ ! -f "$MAIN_FILE" ]; then
    echo -e "${RED}✗ ERROR: main.py not found!${NC}"
    echo "Expected location: $MAIN_FILE"
    exit 1
fi
echo -e "${GREEN}✓${NC} Found main.py: $MAIN_FILE"

# 5. Check critical dependencies
echo ""
echo "Checking Python 3 dependencies..."

check_package() {
    PACKAGE=$1
    DISPLAY_NAME=${2:-$1}
    
    if $PYTHON3 -c "import $PACKAGE" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $DISPLAY_NAME"
        return 0
    else
        echo -e "${RED}✗${NC} $DISPLAY_NAME (missing)"
        return 1
    fi
}

MISSING=0

check_package "PySide6" "PySide6 (Qt GUI)" || MISSING=1
check_package "djitellopy" "djitellopy (Tello SDK)" || MISSING=1
check_package "brainflow" "BrainFlow" || MISSING=1
check_package "torch" "PyTorch" || MISSING=1
check_package "pandas" "Pandas" || MISSING=1
check_package "numpy" "NumPy" || MISSING=1

if [ $MISSING -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}⚠  Some dependencies are missing!${NC}"
    echo "Install them with:"
    echo "  pip3 install PySide6 djitellopy brainflow torch pandas numpy"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 6. Check if NAO service is running (optional but recommended)
echo ""
echo "Checking NAO service status..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} NAO service is running (port 5000)"
    NAO_STATUS=$(curl -s http://localhost:5000/health | grep -o '"status":"ok"' || echo "")
    if [ ! -z "$NAO_STATUS" ]; then
        echo -e "${GREEN}✓${NC} NAO service health check passed"
    fi
else
    echo -e "${YELLOW}⚠${NC}  NAO service not detected (this is OK if not using NAO robot)"
    echo "   To start NAO service: ./start_nao_service.sh"
fi

# 7. Check for Tello drone connectivity (optional)
echo ""
echo "Tello drone status:"
echo -e "${YELLOW}ℹ${NC}  Make sure Tello is powered on and WiFi connected before starting"
echo "   Tello WiFi SSID usually: TELLO-XXXXXX"

# 8. Display configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Python Interpreter: $PYTHON3"
echo "  Python Version:     $PYTHON_VERSION"
echo "  Project Root:       $PROJECT_ROOT"
echo "  Main Script:        $MAIN_FILE"
echo "  Virtual Env:        ${VENV_PATH:-Not using venv}"

# 9. Set environment variables for optimal performance
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_QUICK_CONTROLS_STYLE=Fusion
export PYTHONUNBUFFERED=1

# 10. Optional: Check display server (for GUI)
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo ""
    echo -e "${YELLOW}⚠  Warning: No display server detected${NC}"
    echo "   GUI applications may not work"
    echo "   Make sure you're running this in a graphical environment"
fi

# 11. Start the application
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Starting Avatar GUI Application      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Launch main.py
exec $PYTHON3 "$MAIN_FILE"
