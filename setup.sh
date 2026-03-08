#!/usr/bin/env bash
# =============================================================================
# Viral Clip Extractor — Setup Script
# =============================================================================
# Installs dependencies and verifies the environment for viral_clip_extractor.
#
# Usage:
#   git clone <repo> && cd yacg && ./setup.sh
#
# What this does:
#   1. Detects your OS and package manager
#   2. Checks Python version (3.10+ required)
#   3. Installs system dependencies (ffmpeg)
#   4. Installs Python package in editable mode with dev dependencies
#   5. Optionally pulls Ollama model for semantic analysis / captions
#   6. Verifies everything works
#
# Safe to run multiple times (idempotent).
# =============================================================================

set -euo pipefail

# -- Colors & formatting ------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[i]${NC}  $*"; }
success() { echo -e "${GREEN}[+]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC}  $*"; }
error()   { echo -e "${RED}[-]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${NC}"; }

# -- Header -------------------------------------------------------------------
header "==========================================="
header "  Viral Clip Extractor — Setup"
header "==========================================="
echo ""

# -- Python check -------------------------------------------------------------
header "Checking Python..."

if ! command -v python3 &>/dev/null; then
    error "Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi
success "Python $PYTHON_VERSION"

# pip
if ! python3 -m pip --version &>/dev/null; then
    error "pip not found. Install it with: python3 -m ensurepip --upgrade"
    exit 1
fi
success "pip available"

# -- OS Detection -------------------------------------------------------------
header "Detecting system..."

OS="$(uname -s)"
ARCH="$(uname -m)"
DISTRO="unknown"
PKG_MANAGER="unknown"
WSL=false
SUDO_CMD="sudo"

case "$OS" in
    Linux)
        # WSL check
        if grep -qi microsoft /proc/version 2>/dev/null; then
            WSL=true
        fi

        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case "$ID" in
                ubuntu|debian|pop|mint|elementary|zorin)
                    DISTRO="debian"
                    PKG_MANAGER="apt"
                    ;;
                fedora|rhel|centos|rocky|alma)
                    DISTRO="redhat"
                    if command -v dnf &>/dev/null; then
                        PKG_MANAGER="dnf"
                    else
                        PKG_MANAGER="yum"
                    fi
                    ;;
                arch|manjaro|endeavouros)
                    DISTRO="arch"
                    PKG_MANAGER="pacman"
                    ;;
                opensuse*|suse*)
                    DISTRO="suse"
                    PKG_MANAGER="zypper"
                    ;;
                *)
                    DISTRO="$ID"
                    ;;
            esac
        fi
        ;;
    Darwin)
        DISTRO="macos"
        if command -v brew &>/dev/null; then
            PKG_MANAGER="brew"
            SUDO_CMD=""
        else
            PKG_MANAGER="none"
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        DISTRO="windows"
        PKG_MANAGER="manual"
        ;;
    *)
        warn "Unknown OS: $OS"
        ;;
esac

WSL_TAG=""
if $WSL; then WSL_TAG=" (WSL)"; fi

info "OS: $OS / $DISTRO$WSL_TAG ($ARCH)"
info "Package manager: $PKG_MANAGER"

# -- Root check ---------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    warn "Running as root is not recommended. The script will use sudo when needed."
    SUDO_CMD=""
fi

# -- Install ffmpeg -----------------------------------------------------------
header "Checking ffmpeg..."

if command -v ffmpeg &>/dev/null && command -v ffprobe &>/dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
    success "ffmpeg $FFMPEG_VERSION already installed"
else
    info "ffmpeg not found -- installing..."

    install_ffmpeg() {
        case "$PKG_MANAGER" in
            apt)
                $SUDO_CMD apt-get update -qq
                $SUDO_CMD apt-get install -y -qq ffmpeg
                ;;
            dnf)
                if ! $SUDO_CMD dnf install -y ffmpeg 2>/dev/null; then
                    warn "ffmpeg not in default repos. Trying RPM Fusion..."
                    $SUDO_CMD dnf install -y \
                        "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
                        2>/dev/null || true
                    $SUDO_CMD dnf install -y ffmpeg
                fi
                ;;
            yum)
                $SUDO_CMD yum install -y ffmpeg || {
                    error "ffmpeg not available via yum. You may need EPEL + RPM Fusion."
                    error "See: https://rpmfusion.org/Configuration"
                    exit 1
                }
                ;;
            pacman)
                $SUDO_CMD pacman -Sy --noconfirm ffmpeg
                ;;
            zypper)
                $SUDO_CMD zypper install -y ffmpeg
                ;;
            brew)
                brew install ffmpeg
                ;;
            *)
                error "Don't know how to install ffmpeg on $DISTRO."
                error "Please install ffmpeg manually and re-run this script."
                exit 1
                ;;
        esac
    }

    if [ -n "$SUDO_CMD" ] && ! sudo -n true 2>/dev/null; then
        info "sudo access needed for ffmpeg installation."
    fi

    install_ffmpeg
    success "ffmpeg installed"
fi

# -- Install Python package ----------------------------------------------------
header "Installing Python package..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    error "Cannot find pyproject.toml"
    error "Make sure you're running setup.sh from the project root."
    exit 1
fi

# Determine pip install flags
PIP_FLAGS=()

# If we're in a virtualenv, no special flags needed
if [ -n "${VIRTUAL_ENV:-}" ]; then
    info "Virtual environment detected: $VIRTUAL_ENV"
else
    # Try --user first; if externally managed, add --break-system-packages
    PIP_FLAGS=("--user")
    if python3 -m pip install --help 2>/dev/null | grep -q "break-system-packages"; then
        # Check if we're on an externally-managed system (PEP 668)
        if python3 -c "import sysconfig; p=sysconfig.get_path('stdlib'); exit(0)" 2>/dev/null; then
            # Test if pip actually needs the flag
            if ! python3 -m pip install --dry-run --user pip 2>/dev/null; then
                PIP_FLAGS+=("--break-system-packages")
                warn "Externally managed environment detected -- using --break-system-packages"
            fi
        fi
    fi
fi

info "Installing from pyproject.toml (editable + dev deps)..."
if python3 -m pip install "${PIP_FLAGS[@]}" -e "$SCRIPT_DIR[dev]" 2>&1; then
    success "Python package installed"
else
    warn "Install with flags failed, retrying without --user..."
    PIP_FLAGS_RETRY=()
    if [[ " ${PIP_FLAGS[*]} " =~ " --break-system-packages " ]]; then
        PIP_FLAGS_RETRY=("--break-system-packages")
    fi
    python3 -m pip install "${PIP_FLAGS_RETRY[@]}" -e "$SCRIPT_DIR[dev]" || {
        error "Failed to install Python package."
        error "Try creating a virtual environment first:"
        echo -e "    ${BOLD}python3 -m venv .venv && source .venv/bin/activate && ./setup.sh${NC}"
        exit 1
    }
    success "Python package installed (without --user)"
fi

# -- Check for dual opencv conflict -------------------------------------------
if python3 -c "import importlib.metadata; importlib.metadata.version('opencv-python')" 2>/dev/null; then
    warn "Both opencv-python and opencv-python-headless are installed."
    warn "This can cause conflicts. Removing opencv-python (keeping headless)..."
    python3 -m pip uninstall -y opencv-python 2>/dev/null || true
    success "Removed opencv-python (headless version retained)"
fi

# -- Verify installation ------------------------------------------------------
header "Verifying installation..."

VERIFY_OK=true

# Suppress known third-party warnings during verification
if python3 -W ignore::FutureWarning -W ignore::UserWarning -m viral_clip_extractor check 2>/dev/null; then
    success "viral_clip_extractor check passed"
else
    warn "viral_clip_extractor check reported issues (may be OK for optional deps)"
    VERIFY_OK=false
fi

# Verify individual core imports
for mod in scenedetect librosa cv2 numpy requests yt_dlp; do
    if python3 -c "import $mod" 2>/dev/null; then
        success "$mod importable"
    else
        error "$mod failed to import"
        VERIFY_OK=false
    fi
done

if command -v ffmpeg &>/dev/null; then
    success "ffmpeg on PATH"
else
    error "ffmpeg not on PATH after install"
    VERIFY_OK=false
fi

if command -v ffprobe &>/dev/null; then
    success "ffprobe on PATH"
else
    error "ffprobe not on PATH after install"
    VERIFY_OK=false
fi

# -- Ollama (optional) -------------------------------------------------------
header "Checking Ollama (optional)..."

if command -v ollama &>/dev/null; then
    success "Ollama found"
    if ollama list 2>/dev/null | grep -q "qwen2.5-vl"; then
        success "qwen2.5-vl model already available"
    else
        info "Pulling qwen2.5-vl:7b model (this may take a while)..."
        if ollama pull qwen2.5vl:7b 2>/dev/null; then
            success "qwen2.5-vl:7b model pulled"
        else
            warn "Failed to pull model. You can do it manually later:"
            echo -e "    ${BOLD}ollama pull qwen2.5vl:7b${NC}"
        fi
    fi
else
    info "Ollama not installed (optional -- needed for semantic analysis and captions)"
    info "Install from: https://ollama.com/download"
    echo -e "    Then run: ${BOLD}ollama pull qwen2.5vl:7b${NC}"
fi

# -- GPU info -----------------------------------------------------------------
header "GPU detection..."

HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        success "NVIDIA GPU: $GPU_NAME ($GPU_MEM)"
        HAS_CUDA=true
    fi
fi

if ! $HAS_CUDA; then
    if [ "$DISTRO" = "macos" ] && [ "$ARCH" = "arm64" ]; then
        success "Apple Silicon detected -- MPS acceleration may be available"
    else
        info "No NVIDIA GPU detected -- CPU mode (works fine, visual analysis will be slower)"
    fi
fi

# -- Done! --------------------------------------------------------------------
echo ""
header "==========================================="
header "  Setup complete!"
header "==========================================="
echo ""

if $VERIFY_OK; then
    echo -e "  ${GREEN}All core dependencies verified.${NC}"
else
    echo -e "  ${YELLOW}Some checks had warnings -- see above.${NC}"
fi

echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo -e "    python -m viral_clip_extractor check"
echo -e "    python -m viral_clip_extractor process --video ${CYAN}video.mp4${NC} --title ${CYAN}\"My Video\"${NC}"
echo -e "    python -m viral_clip_extractor youtube --url ${CYAN}https://youtube.com/watch?v=XXXXX${NC}"
echo -e "    python -m viral_clip_extractor batch --videos-dir ${CYAN}/path/to/videos/${NC}"
echo ""
echo -e "  ${BOLD}Useful flags:${NC}"
echo -e "    --min-score 0     Accept all clips regardless of score"
echo -e "    --top-n 5         Limit to top 5 clips"
echo -e "    -v                Verbose/debug output"
echo ""
echo -e "  ${BOLD}Optional (for semantic analysis & captions):${NC}"
echo -e "    1. Install Ollama: ${CYAN}https://ollama.com/download${NC}"
echo -e "    2. Pull model:    ${BOLD}ollama pull qwen2.5vl:7b${NC}"
echo -e "    3. Start Ollama:  ${BOLD}ollama serve${NC}"
echo ""
echo ""
