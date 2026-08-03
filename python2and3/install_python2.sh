#!/bin/bash
# Python 2.7 Installation Script for Ubuntu/Debian
# Installs Python 2.7 and pip for NAO6 service

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Python 2.7 Installer (Ubuntu/Debian)   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}\n"

# Check if running on Ubuntu/Debian
if [ ! -f /etc/debian_version ]; then
    echo -e "${RED}✗ ERROR: This script is for Ubuntu/Debian systems only${NC}"
    echo "For macOS, use: install_python2_macos.sh"
    exit 1
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠  Warning: Running as root${NC}"
    echo "This script will install system-wide packages"
else
    echo -e "${GREEN}➤${NC} This script will use sudo for installations"
    echo "You may be prompted for your password"
fi

echo ""
echo -e "${BLUE}═══ Step 1: Update Package Lists ═══${NC}\n"
sudo apt update
echo -e "${GREEN}✓${NC} Package lists updated"

echo ""
echo -e "${BLUE}═══ Step 2: Install Python 2.7 ═══${NC}\n"

# Check if Python 2.7 is already installed
if command -v python2.7 &> /dev/null; then
    CURRENT_VERSION=$(python2.7 --version 2>&1)
    echo -e "${GREEN}✓${NC} Python 2.7 is already installed: $CURRENT_VERSION"
else
    echo -e "${YELLOW}➤${NC} Installing Python 2.7..."
    sudo apt install -y python2.7 python2.7-dev
    echo -e "${GREEN}✓${NC} Python 2.7 installed successfully"
fi

echo ""
echo -e "${BLUE}═══ Step 3: Install pip for Python 2.7 ═══${NC}\n"

# Check if pip is already installed for Python 2.7
if python2.7 -m pip --version &> /dev/null; then
    PIP_VERSION=$(python2.7 -m pip --version 2>&1)
    echo -e "${GREEN}✓${NC} pip is already installed: $PIP_VERSION"
else
    echo -e "${YELLOW}➤${NC} Installing pip for Python 2.7..."
    
    # Download get-pip.py for Python 2.7
    curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o /tmp/get-pip.py
    
    # Install pip
    sudo python2.7 /tmp/get-pip.py
    
    # Cleanup
    rm /tmp/get-pip.py
    
    echo -e "${GREEN}✓${NC} pip installed successfully"
fi

echo ""
echo -e "${BLUE}═══ Step 4: Install Essential Build Tools ═══${NC}\n"

echo -e "${YELLOW}➤${NC} Installing build-essential and development headers..."
sudo apt install -y build-essential python2.7-dev libffi-dev libssl-dev

echo -e "${GREEN}✓${NC} Build tools installed"

echo ""
echo -e "${BLUE}═══ Step 5: Install Python 2.7 Dependencies for NAO ═══${NC}\n"

echo -e "${YELLOW}➤${NC} Installing Flask and Werkzeug (compatible versions)..."

# Install specific versions compatible with Python 2.7
python2.7 -m pip install --user flask==1.1.4 werkzeug==1.0.1 markupsafe==1.1.1 jinja2==2.11.3

echo -e "${GREEN}✓${NC} Flask and dependencies installed"

echo ""
echo -e "${BLUE}═══ Installation Complete! ═══${NC}\n"

# Verify installation
echo "Verification:"
echo ""
echo -e "${GREEN}Python 2.7:${NC}"
python2.7 --version
echo ""
echo -e "${GREEN}pip for Python 2.7:${NC}"
python2.7 -m pip --version
echo ""
echo -e "${GREEN}Flask version:${NC}"
python2.7 -c "import flask; print('Flask', flask.__version__)"
echo ""

echo -e "${BLUE}═══ Next Steps ═══${NC}\n"
echo "1. Extract NAOqi SDK to your Avatar project directory"
echo "2. Run: ./start_nao_service.sh to test NAO service"
echo "3. Run: ./start_avatar_full.sh to launch complete environment"
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
