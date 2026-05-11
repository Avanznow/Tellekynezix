#!/bin/bash
# Python 2.7 Installation Script for macOS
# Installs Python 2.7 and pip for NAO6 service

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      Python 2.7 Installer (macOS)        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}\n"

# Check if running on macOS
if [ "$(uname)" != "Darwin" ]; then
    echo -e "${RED}✗ ERROR: This script is for macOS only${NC}"
    echo "For Ubuntu/Debian, use: install_python2.sh"
    exit 1
fi

echo -e "${GREEN}➤${NC} Detected macOS system"
echo ""

# Check for Homebrew
echo -e "${BLUE}═══ Step 1: Check for Homebrew ═══${NC}\n"

if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}⚠${NC}  Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo -e "${GREEN}✓${NC} Homebrew installed"
else
    echo -e "${GREEN}✓${NC} Homebrew is already installed"
    echo -e "${YELLOW}➤${NC} Updating Homebrew..."
    brew update
fi

echo ""
echo -e "${BLUE}═══ Step 2: Install Python 2.7 ═══${NC}\n"

# Check if Python 2.7 is already installed
if command -v python2.7 &> /dev/null; then
    CURRENT_VERSION=$(python2.7 --version 2>&1)
    echo -e "${GREEN}✓${NC} Python 2.7 is already installed: $CURRENT_VERSION"
else
    echo -e "${YELLOW}➤${NC} Installing Python 2.7 via Homebrew..."
    
    # Python@2 is deprecated, so we may need to use pyenv or manual installation
    echo -e "${YELLOW}ℹ${NC}  Note: Python 2.7 is no longer available in Homebrew"
    echo "      Installing via alternative method..."
    
    # Install pyenv for Python version management
    if ! command -v pyenv &> /dev/null; then
        echo -e "${YELLOW}➤${NC} Installing pyenv..."
        brew install pyenv
    fi
    
    # Install Python 2.7.18 (last Python 2.7 release) via pyenv
    echo -e "${YELLOW}➤${NC} Installing Python 2.7.18 via pyenv..."
    pyenv install 2.7.18
    
    # Set Python 2.7.18 as a version available system-wide
    pyenv global 2.7.18
    
    # Add pyenv to PATH
    if ! grep -q 'pyenv init' ~/.zshrc 2>/dev/null; then
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    fi
    
    if ! grep -q 'pyenv init' ~/.bash_profile 2>/dev/null; then
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
        echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
    fi
    
    # Source the configuration
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    echo -e "${GREEN}✓${NC} Python 2.7 installed successfully via pyenv"
fi

echo ""
echo -e "${BLUE}═══ Step 3: Install pip for Python 2.7 ═══${NC}\n"

# Find Python 2.7 executable
PYTHON27=""
if command -v python2.7 &> /dev/null; then
    PYTHON27=$(command -v python2.7)
elif [ -f "$HOME/.pyenv/versions/2.7.18/bin/python2.7" ]; then
    PYTHON27="$HOME/.pyenv/versions/2.7.18/bin/python2.7"
fi

if [ -z "$PYTHON27" ]; then
    echo -e "${RED}✗ ERROR: Python 2.7 not found in PATH${NC}"
    echo "Try sourcing your shell configuration:"
    echo "  source ~/.zshrc    # for zsh"
    echo "  source ~/.bash_profile  # for bash"
    exit 1
fi

# Check if pip is already installed for Python 2.7
if $PYTHON27 -m pip --version &> /dev/null; then
    PIP_VERSION=$($PYTHON27 -m pip --version 2>&1)
    echo -e "${GREEN}✓${NC} pip is already installed: $PIP_VERSION"
else
    echo -e "${YELLOW}➤${NC} Installing pip for Python 2.7..."
    
    # Download get-pip.py for Python 2.7
    curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o /tmp/get-pip.py
    
    # Install pip
    $PYTHON27 /tmp/get-pip.py --user
    
    # Cleanup
    rm /tmp/get-pip.py
    
    echo -e "${GREEN}✓${NC} pip installed successfully"
fi

echo ""
echo -e "${BLUE}═══ Step 4: Install Python 2.7 Dependencies for NAO ═══${NC}\n"

echo -e "${YELLOW}➤${NC} Installing Flask and Werkzeug (compatible versions)..."

# Install specific versions compatible with Python 2.7
$PYTHON27 -m pip install --user flask==1.1.4 werkzeug==1.0.1 markupsafe==1.1.1 jinja2==2.11.3

echo -e "${GREEN}✓${NC} Flask and dependencies installed"

echo ""
echo -e "${BLUE}═══ Installation Complete! ═══${NC}\n"

# Verify installation
echo "Verification:"
echo ""
echo -e "${GREEN}Python 2.7:${NC}"
$PYTHON27 --version
echo ""
echo -e "${GREEN}pip for Python 2.7:${NC}"
$PYTHON27 -m pip --version
echo ""
echo -e "${GREEN}Flask version:${NC}"
$PYTHON27 -c "import flask; print('Flask', flask.__version__)"
echo ""

echo -e "${BLUE}═══ Next Steps ═══${NC}\n"
echo "1. Restart your terminal or source your shell config:"
echo "   source ~/.zshrc    # for zsh"
echo "   source ~/.bash_profile  # for bash"
echo ""
echo "2. Extract NAOqi SDK to your Avatar project directory"
echo "3. Run: ./start_nao_service.sh to test NAO service"
echo "4. Run: ./start_avatar_full.sh to launch complete environment"
echo ""
echo -e "${YELLOW}NOTE for macOS:${NC} NAOqi SDK binaries may not work on newer macOS versions (ARM/M1/M2)"
echo "If NAOqi doesn't work, consider using a Linux VM or Docker for the NAO service."
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
