#!/usr/bin/env python3
"""
Helper Script for Instagram Caption Generator
Provides easy menu-driven interface for common operations
"""

import sys
import subprocess
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║      INSTAGRAM CAPTION GENERATOR                          ║
║      Automated AI-Powered Caption Creation                ║
║      Optimized for 2026 Algorithm                         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)

def print_menu():
    menu = f"""
{Colors.BOLD}Main Menu:{Colors.ENDC}

{Colors.GREEN}1.{Colors.ENDC} Run System Test (check installation)
{Colors.GREEN}2.{Colors.ENDC} Generate Captions (process all videos)
{Colors.GREEN}3.{Colors.ENDC} Generate Captions (force reprocess)
{Colors.GREEN}4.{Colors.ENDC} View Output CSV
{Colors.GREEN}5.{Colors.ENDC} Check Video Count
{Colors.GREEN}6.{Colors.ENDC} Check Model Status
{Colors.GREEN}7.{Colors.ENDC} Pull/Update Model
{Colors.GREEN}8.{Colors.ENDC} View Recent Logs
{Colors.GREEN}9.{Colors.ENDC} Clean Output (remove CSV)
{Colors.RED}0.{Colors.ENDC} Exit

"""
    print(menu)

def run_command(cmd, description):
    """Run a command and display output"""
    print(f"\n{Colors.CYAN}→ {description}...{Colors.ENDC}\n")
    try:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        return False

def get_video_count():
    """Count video files in videos directory"""
    video_dir = Path('/home/claude/instagram-caption-generator/videos')
    
    if not video_dir.exists():
        print(f"{Colors.RED}Videos directory not found{Colors.ENDC}")
        return
    
    mp4_files = list(video_dir.glob('*.mp4'))
    mov_files = list(video_dir.glob('*.mov'))
    avi_files = list(video_dir.glob('*.avi'))
    
    total = len(mp4_files) + len(mov_files) + len(avi_files)
    
    print(f"\n{Colors.BOLD}Video Files in videos/ directory:{Colors.ENDC}")
    print(f"  MP4:  {len(mp4_files)}")
    print(f"  MOV:  {len(mov_files)}")
    print(f"  AVI:  {len(avi_files)}")
    print(f"  {Colors.GREEN}Total: {total}{Colors.ENDC}\n")
    
    if total > 0:
        total_size = sum(f.stat().st_size for f in mp4_files + mov_files + avi_files)
        size_mb = total_size / (1024 * 1024)
        print(f"  Total size: {size_mb:.1f} MB\n")

def check_model_status():
    """Check Ollama model status"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            print(f"\n{Colors.BOLD}Ollama Models Installed:{Colors.ENDC}")
            
            if not models:
                print(f"  {Colors.YELLOW}No models installed{Colors.ENDC}")
            else:
                for model in models:
                    name = model['name']
                    size = model.get('size', 0) / (1024**3)  # Convert to GB
                    
                    is_qwen = 'qwen' in name.lower()
                    color = Colors.GREEN if is_qwen else Colors.ENDC
                    marker = '✓' if is_qwen else ' '
                    
                    print(f"  {color}{marker} {name} ({size:.1f} GB){Colors.ENDC}")
            
            print()
        else:
            print(f"{Colors.RED}Cannot connect to Ollama{Colors.ENDC}")
            
    except ImportError:
        print(f"{Colors.RED}requests module not installed{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Ollama not running - start with: ollama serve{Colors.ENDC}")

def view_csv():
    """Display CSV contents"""
    csv_path = Path('/home/claude/instagram-caption-generator/output/instagram_captions.csv')
    
    if not csv_path.exists():
        print(f"{Colors.YELLOW}No output CSV found yet. Generate captions first.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.BOLD}Recent Entries from CSV:{Colors.ENDC}\n")
    
    try:
        # Read and display last 5 lines
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            
        if len(lines) <= 1:
            print(f"{Colors.YELLOW}CSV is empty{Colors.ENDC}")
        else:
            # Show header
            print(f"{Colors.CYAN}{lines[0].strip()}{Colors.ENDC}")
            
            # Show last 5 data rows
            for line in lines[-5:]:
                print(line.strip())
        
        print(f"\n{Colors.GREEN}Total entries: {len(lines) - 1}{Colors.ENDC}")
        print(f"Full path: {csv_path}\n")
        
    except Exception as e:
        print(f"{Colors.RED}Error reading CSV: {e}{Colors.ENDC}")

def view_logs():
    """Display recent log entries"""
    log_dir = Path('/home/claude/instagram-caption-generator/logs')
    
    if not log_dir.exists():
        print(f"{Colors.YELLOW}No logs directory found{Colors.ENDC}")
        return
    
    log_files = sorted(log_dir.glob('*.log'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not log_files:
        print(f"{Colors.YELLOW}No log files found{Colors.ENDC}")
        return
    
    latest_log = log_files[0]
    print(f"\n{Colors.BOLD}Latest Log: {latest_log.name}{Colors.ENDC}\n")
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            
        # Show last 20 lines
        for line in lines[-20:]:
            # Color code log levels
            if 'ERROR' in line:
                print(f"{Colors.RED}{line.strip()}{Colors.ENDC}")
            elif 'WARNING' in line:
                print(f"{Colors.YELLOW}{line.strip()}{Colors.ENDC}")
            elif 'INFO' in line:
                print(line.strip())
            else:
                print(line.strip())
        
        print(f"\n{Colors.GREEN}Showing last 20 lines{Colors.ENDC}")
        print(f"Full log: {latest_log}\n")
        
    except Exception as e:
        print(f"{Colors.RED}Error reading log: {e}{Colors.ENDC}")

def clean_output():
    """Remove output CSV"""
    csv_path = Path('/home/claude/instagram-caption-generator/output/instagram_captions.csv')
    
    if not csv_path.exists():
        print(f"{Colors.YELLOW}No output CSV to clean{Colors.ENDC}")
        return
    
    print(f"\n{Colors.YELLOW}⚠ WARNING: This will delete the output CSV file!{Colors.ENDC}")
    confirm = input("Are you sure? (yes/no): ").lower()
    
    if confirm == 'yes':
        try:
            csv_path.unlink()
            print(f"{Colors.GREEN}✓ Output CSV deleted{Colors.ENDC}\n")
        except Exception as e:
            print(f"{Colors.RED}Error deleting CSV: {e}{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}Cancelled{Colors.ENDC}\n")

def main():
    """Main menu loop"""
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input(f"{Colors.BOLD}Select option (0-9): {Colors.ENDC}").strip()
            
            if choice == '1':
                run_command('python test_system.py', 'Running system tests')
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '2':
                run_command('python caption_generator.py', 'Generating captions')
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '3':
                run_command('python caption_generator.py --force-reprocess', 
                          'Generating captions (force reprocess)')
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '4':
                view_csv()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '5':
                get_video_count()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '6':
                check_model_status()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '7':
                run_command('ollama pull qwen2.5-vl:7b', 'Pulling/updating model')
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '8':
                view_logs()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '9':
                clean_output()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == '0':
                print(f"\n{Colors.GREEN}Thanks for using Caption Generator! 🎬✨{Colors.ENDC}\n")
                break
            
            else:
                print(f"{Colors.RED}Invalid option. Please select 0-9.{Colors.ENDC}")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Goodbye! 👋{Colors.ENDC}\n")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

if __name__ == '__main__':
    main()
