#!/bin/bash
# Full Avatar Environment Launcher
# Starts both NAO service (Python 2.7) and Avatar GUI (Python 3)
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                       ║${NC}"
echo -e "${CYAN}║      Avatar Full Environment Launcher v1.0           ║${NC}"
echo -e "${CYAN}║      NAO6 (Python 2.7) + Tello (Python 3)            ║${NC}"
echo -e "${CYAN}║                                                       ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════╝${NC}\n"

# Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}═══ Environment Setup ═══${NC}\n"
echo -e "${GREEN}➤${NC} Project root: $PROJECT_ROOT"
echo -e "${GREEN}➤${NC} Script directory: $SCRIPT_DIR"

# Log files
NAO_LOG="$PROJECT_ROOT/nao_service.log"
AVATAR_LOG="$PROJECT_ROOT/avatar_gui.log"

# PID files for cleanup
NAO_PID_FILE="/tmp/avatar_nao_service.pid"
AVATAR_PID_FILE="/tmp/avatar_gui.pid"

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}═══ Shutting Down Services ═══${NC}"
    
    # Kill NAO service
    if [ -f "$NAO_PID_FILE" ]; then
        NAO_PID=$(cat "$NAO_PID_FILE")
        if ps -p $NAO_PID > /dev/null 2>&1; then
            echo -e "${YELLOW}➤${NC} Stopping NAO service (PID: $NAO_PID)..."
            kill $NAO_PID 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if ps -p $NAO_PID > /dev/null 2>&1; then
                kill -9 $NAO_PID 2>/dev/null || true
            fi
        fi
        rm -f "$NAO_PID_FILE"
    fi
    
    # Kill Avatar GUI
    if [ -f "$AVATAR_PID_FILE" ]; then
        AVATAR_PID=$(cat "$AVATAR_PID_FILE")
        if ps -p $AVATAR_PID > /dev/null 2>&1; then
            echo -e "${YELLOW}➤${NC} Stopping Avatar GUI (PID: $AVATAR_PID)..."
            kill $AVATAR_PID 2>/dev/null || true
            sleep 1
            if ps -p $AVATAR_PID > /dev/null 2>&1; then
                kill -9 $AVATAR_PID 2>/dev/null || true
            fi
        fi
        rm -f "$AVATAR_PID_FILE"
    fi
    
    echo -e "${GREEN}✓${NC} Cleanup complete"
    echo ""
    echo -e "${CYAN}═══ Session Logs ═══${NC}"
    echo "NAO Service Log: $NAO_LOG"
    echo "Avatar GUI Log:  $AVATAR_LOG"
    echo ""
    exit 0
}

# Register cleanup on script exit
trap cleanup EXIT INT TERM

# Check if required launch scripts exist
if [ ! -f "$SCRIPT_DIR/start_nao_service.sh" ]; then
    echo -e "${RED}✗ ERROR: start_nao_service.sh not found!${NC}"
    echo "Expected: $SCRIPT_DIR/start_nao_service.sh"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/start_tello_environment.sh" ]; then
    echo -e "${RED}✗ ERROR: start_tello_environment.sh not found!${NC}"
    echo "Expected: $SCRIPT_DIR/start_tello_environment.sh"
    exit 1
fi

# Make scripts executable
chmod +x "$SCRIPT_DIR/start_nao_service.sh"
chmod +x "$SCRIPT_DIR/start_tello_environment.sh"

echo ""
echo -e "${BLUE}═══ Phase 1: Starting NAO Service (Python 2.7) ═══${NC}\n"

# Start NAO service in background
echo -e "${GREEN}➤${NC} Launching NAO service on port 5000..."
"$SCRIPT_DIR/start_nao_service.sh" > "$NAO_LOG" 2>&1 &
NAO_PID=$!
echo $NAO_PID > "$NAO_PID_FILE"
echo -e "${GREEN}✓${NC} NAO service started (PID: $NAO_PID)"
echo -e "${CYAN}ℹ${NC}  Log file: $NAO_LOG"

# Wait for NAO service to be ready
echo -e "${YELLOW}➤${NC} Waiting for NAO service to be ready..."
MAX_WAIT=30
COUNTER=0
while [ $COUNTER -lt $MAX_WAIT ]; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} NAO service is ready!"
        break
    fi
    sleep 1
    COUNTER=$((COUNTER + 1))
    echo -n "."
done
echo ""

if [ $COUNTER -eq $MAX_WAIT ]; then
    echo -e "${RED}✗ ERROR: NAO service failed to start within ${MAX_WAIT}s${NC}"
    echo "Check log file: $NAO_LOG"
    echo ""
    echo "Last 20 lines of NAO service log:"
    tail -n 20 "$NAO_LOG"
    exit 1
fi

# Verify NAO service health
echo -e "${GREEN}➤${NC} Verifying NAO service health..."
HEALTH_RESPONSE=$(curl -s http://localhost:5000/health)
if echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✓${NC} NAO service health check passed"
    echo -e "${CYAN}   Response: $HEALTH_RESPONSE${NC}"
else
    echo -e "${YELLOW}⚠${NC}  Warning: Unexpected health response"
    echo -e "${CYAN}   Response: $HEALTH_RESPONSE${NC}"
fi

echo ""
echo -e "${BLUE}═══ Phase 2: Starting Avatar GUI (Python 3) ═══${NC}\n"

# Small delay to ensure NAO service is stable
sleep 2

echo -e "${GREEN}➤${NC} Launching Avatar GUI with Tello support..."
echo -e "${CYAN}ℹ${NC}  This will open the main application window"
echo ""

# Start Avatar GUI (this will run in foreground)
# We redirect output to log file but also show it on screen using tee
"$SCRIPT_DIR/start_tello_environment.sh" 2>&1 | tee "$AVATAR_LOG" &
AVATAR_PID=$!
echo $AVATAR_PID > "$AVATAR_PID_FILE"

# Wait for Avatar GUI process
wait $AVATAR_PID

# Cleanup will be handled by trap
