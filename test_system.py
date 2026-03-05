#!/usr/bin/env python3
"""
Test script for Instagram Caption Generator
Verifies installation and configuration before running main pipeline
"""

import sys
import subprocess
import os
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def print_test(name, passed, details=""):
    status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
    print(f"{status} - {name}")
    if details:
        print(f"       {details}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 10)
    passed = version >= required
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    print_test("Python Version (>=3.10)", passed, details)
    return passed

def check_command(cmd, name):
    """Check if a command exists"""
    try:
        result = subprocess.run([cmd, '--version'], capture_output=True, timeout=5)
        passed = result.returncode == 0
        details = result.stdout.decode().split('\n')[0] if passed else "Not installed"
    except FileNotFoundError:
        passed = False
        details = "Not found in PATH"
    except Exception as e:
        passed = False
        details = str(e)
    
    print_test(f"{name} Installed", passed, details)
    return passed

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        passed = response.status_code == 200
        
        if passed:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            # Check for recommended model
            has_qwen = any('qwen' in name.lower() for name in model_names)
            
            if has_qwen:
                details = f"Running with {len(models)} model(s) - Qwen models detected ✓"
            else:
                details = f"Running with {len(models)} model(s) - No Qwen models found"
        else:
            details = f"HTTP {response.status_code}"
            
    except ImportError:
        passed = False
        details = "requests module not installed - run: pip install requests --break-system-packages"
    except requests.exceptions.ConnectionError:
        passed = False
        details = "Ollama not running - start with: ollama serve"
    except Exception as e:
        passed = False
        details = str(e)
    
    print_test("Ollama Service", passed, details)
    return passed

def check_qwen_model():
    """Check if Qwen2.5-VL model is available"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            # Look for any Qwen VL model
            qwen_models = [name for name in model_names if 'qwen' in name.lower() and 'vl' in name.lower()]
            
            if qwen_models:
                passed = True
                details = f"Found: {', '.join(qwen_models)}"
            else:
                passed = False
                details = "Not found - run: ollama pull qwen2.5-vl:7b"
        else:
            passed = False
            details = "Cannot connect to Ollama"
            
    except Exception as e:
        passed = False
        details = str(e)
    
    print_test("Qwen2.5-VL Model", passed, details)
    return passed

def check_directories():
    """Check if required directories exist"""
    base_dir = Path('/home/claude/instagram-caption-generator')
    required_dirs = ['videos', 'output', 'logs']
    
    all_exist = True
    details_list = []
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        exists = dir_path.exists() and dir_path.is_dir()
        all_exist = all_exist and exists
        
        if exists:
            details_list.append(f"{dir_name}/ ✓")
        else:
            details_list.append(f"{dir_name}/ ✗")
    
    details = " | ".join(details_list)
    print_test("Directory Structure", all_exist, details)
    return all_exist

def check_video_files():
    """Check if there are video files to process"""
    video_dir = Path('/home/claude/instagram-caption-generator/videos')
    
    if not video_dir.exists():
        passed = False
        details = "Video directory doesn't exist"
    else:
        video_files = list(video_dir.glob('*.mp4')) + \
                     list(video_dir.glob('*.mov')) + \
                     list(video_dir.glob('*.avi'))
        
        passed = len(video_files) > 0
        
        if passed:
            total_size = sum(f.stat().st_size for f in video_files)
            size_mb = total_size / (1024 * 1024)
            details = f"{len(video_files)} video(s) found ({size_mb:.1f} MB)"
        else:
            details = "No videos found - add .mp4/.mov/.avi files to videos/ directory"
    
    print_test("Video Files Present", passed, details)
    return passed

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        stat = shutil.disk_usage('/home/claude/instagram-caption-generator')
        free_gb = stat.free / (1024**3)
        passed = free_gb > 5.0  # At least 5GB free
        details = f"{free_gb:.1f} GB free"
    except Exception as e:
        passed = False
        details = str(e)
    
    print_test("Disk Space (>5GB)", passed, details)
    return passed

def check_permissions():
    """Check write permissions"""
    test_file = Path('/home/claude/instagram-caption-generator/output/.test_write')
    
    try:
        test_file.write_text('test')
        test_file.unlink()
        passed = True
        details = "Write permissions OK"
    except Exception as e:
        passed = False
        details = f"Cannot write to output/ - {str(e)}"
    
    print_test("Write Permissions", passed, details)
    return passed

def run_all_tests():
    """Run all system checks"""
    print_header("Instagram Caption Generator - System Check")
    
    results = {
        'python': check_python_version(),
        'ffmpeg': check_command('ffmpeg', 'FFmpeg'),
        'ffprobe': check_command('ffprobe', 'FFprobe'),
        'ollama': check_command('ollama', 'Ollama'),
        'ollama_running': check_ollama_running(),
        'qwen_model': check_qwen_model(),
        'directories': check_directories(),
        'videos': check_video_files(),
        'disk': check_disk_space(),
        'permissions': check_permissions(),
    }
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'SUMMARY':^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED ({passed_count}/{total_count}){Colors.ENDC}")
        print(f"\n{Colors.GREEN}System is ready! Run: python caption_generator.py{Colors.ENDC}\n")
        return True
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ SOME TESTS FAILED ({passed_count}/{total_count} passed){Colors.ENDC}\n")
        
        # Provide specific guidance
        if not results['python']:
            print(f"{Colors.YELLOW}→ Upgrade Python to 3.10 or higher{Colors.ENDC}")
        
        if not results['ffmpeg'] or not results['ffprobe']:
            print(f"{Colors.YELLOW}→ Install FFmpeg: sudo apt install ffmpeg{Colors.ENDC}")
        
        if not results['ollama']:
            print(f"{Colors.YELLOW}→ Install Ollama: curl -fsSL https://ollama.com/install.sh | sh{Colors.ENDC}")
        
        if not results['ollama_running']:
            print(f"{Colors.YELLOW}→ Start Ollama service: ollama serve{Colors.ENDC}")
        
        if not results['qwen_model']:
            print(f"{Colors.YELLOW}→ Pull model: ollama pull qwen2.5-vl:7b{Colors.ENDC}")
        
        if not results['videos']:
            print(f"{Colors.YELLOW}→ Add videos to: /home/claude/instagram-caption-generator/videos/{Colors.ENDC}")
        
        if not results['disk']:
            print(f"{Colors.YELLOW}→ Free up disk space (need >5GB){Colors.ENDC}")
        
        if not results['permissions']:
            print(f"{Colors.YELLOW}→ Fix permissions: chmod 755 /home/claude/instagram-caption-generator/output/{Colors.ENDC}")
        
        print()
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
