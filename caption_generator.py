#!/usr/bin/env python3
"""
Instagram Caption Generator for Short-Form Videos
Uses Qwen2.5-VL via Ollama for video analysis and caption generation
Optimized for maximum Instagram reach and virality (2026 algorithm)
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import csv
import base64
import time

# Check for required packages
try:
    import requests
except ImportError:
    print("Installing required package: requests")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "--break-system-packages"])
    import requests


@dataclass
class VideoMetadata:
    """Metadata extracted from video file"""
    filename: str
    filepath: str
    duration: float
    width: int
    height: int
    file_size: int
    fps: float


@dataclass
class CaptionData:
    """Generated caption data for Instagram"""
    hook: str
    description: str
    hashtags: List[str]
    category: str
    virality_score: int
    full_caption: str


@dataclass
class ProcessedVideo:
    """Complete processed video entry"""
    video_filename: str
    title: str
    hook: str
    description: str
    hashtags: str
    full_caption: str
    category: str
    virality_score: int
    duration: float
    resolution: str
    processed_timestamp: str


class OllamaVideoAnalyzer:
    """Interface to Ollama for video analysis using Qwen2.5-VL"""
    
    def __init__(self, model: str = "qwen2.5-vl:7b", ollama_host: str = "http://localhost:11434"):
        self.model = model
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
        
    def check_model_availability(self) -> bool:
        """Check if the required model is available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]
                if self.model in available_models:
                    self.logger.info(f"✓ Model {self.model} is available")
                    return True
                else:
                    self.logger.warning(f"Model {self.model} not found. Available models: {available_models}")
                    return False
            return False
        except Exception as e:
            self.logger.error(f"Failed to check Ollama models: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the required model if not available"""
        self.logger.info(f"Pulling model {self.model}...")
        try:
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=600
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        self.logger.info(f"  {data['status']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pull model: {e}")
            return False
    
    def analyze_video(self, video_path: str, title: str, max_retries: int = 3) -> Optional[CaptionData]:
        """
        Analyze video and generate Instagram-optimized caption
        
        Args:
            video_path: Path to video file
            title: User-provided title for the video
            max_retries: Number of retry attempts for API calls
            
        Returns:
            CaptionData object or None if analysis fails
        """
        prompt = self._create_instagram_prompt(title)
        
        # Prepare the video for Ollama (base64 encode)
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to read video {video_path}: {e}")
            return None
        
        # Call Ollama API
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Analyzing video (attempt {attempt + 1}/{max_retries})...")
                
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [video_base64],
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                        }
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    
                    # Parse the JSON response
                    caption_data = self._parse_llm_response(response_text)
                    if caption_data:
                        self.logger.info("✓ Successfully generated caption")
                        return caption_data
                    else:
                        self.logger.warning(f"Failed to parse LLM response on attempt {attempt + 1}")
                else:
                    self.logger.error(f"Ollama API error: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
        
        return None
    
    def _create_instagram_prompt(self, title: str) -> str:
        """Create optimized prompt for Instagram caption generation"""
        return f"""You are an expert Instagram content strategist specializing in ASMR and short-form viral content. You understand the Instagram algorithm's 2026 preferences: high watch time, immediate hooks, emotional resonance, and niche-specific targeting.

Analyze this video titled "{title}".

Generate Instagram-optimized content with these STRICT requirements:

1. HOOK (8-12 words): Create an immediate attention-grabbing opening line that creates curiosity or emotional resonance. Must make viewers want to watch.

2. DESCRIPTION (2-3 sentences, 100-150 chars): 
   - Integrate keywords naturally
   - Focus on viewer benefit or emotional outcome
   - Include subtle call-to-action
   - Use conversational tone

3. HASHTAGS (exactly 3-5):
   - 1-2 niche-specific (high relevance, low competition)
   - 1-2 discovery (medium competition)
   - 1 trending/broad (if relevant)
   - For ASMR content prioritize: #asmr #asmrsounds #relaxing
   
4. CONTENT CATEGORY: Classify as one of [ASMR, Satisfying, Tutorial, Story, Transition, Dance, Comedy, Educational]

5. VIRALITY PREDICTION (0-100): Estimate viral potential based on:
   - Hook strength
   - Visual appeal
   - Trend alignment
   - Emotional resonance

Output ONLY valid JSON in this exact format with NO additional text before or after:
{{
  "hook": "...",
  "description": "...",
  "hashtags": ["tag1", "tag2", "tag3"],
  "category": "...",
  "virality_score": 75
}}"""
    
    def _parse_llm_response(self, response_text: str) -> Optional[CaptionData]:
        """Parse LLM response and extract caption data"""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.error("No JSON found in response")
                return None
            
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['hook', 'description', 'hashtags', 'category', 'virality_score']
            if not all(field in data for field in required_fields):
                self.logger.error(f"Missing required fields in response")
                return None
            
            # Ensure hashtags are properly formatted
            hashtags = data['hashtags']
            if isinstance(hashtags, str):
                hashtags = [tag.strip() for tag in hashtags.split(',')]
            
            # Add # prefix if not present
            hashtags = [tag if tag.startswith('#') else f"#{tag}" for tag in hashtags]
            
            # Limit to 5 hashtags
            hashtags = hashtags[:5]
            
            # Create full caption
            full_caption = self._format_full_caption(
                data['hook'],
                data['description'],
                hashtags
            )
            
            return CaptionData(
                hook=data['hook'],
                description=data['description'],
                hashtags=hashtags,
                category=data['category'],
                virality_score=int(data['virality_score']),
                full_caption=full_caption
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.debug(f"Response text: {response_text}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return None
    
    def _format_full_caption(self, hook: str, description: str, hashtags: List[str]) -> str:
        """Format complete Instagram caption"""
        caption_parts = [
            hook,
            "",  # Empty line for spacing
            description,
            "",  # Empty line before hashtags
            " ".join(hashtags)
        ]
        return "\n".join(caption_parts)


class VideoProcessor:
    """Process video files and extract metadata"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_metadata(self, video_path: str) -> Optional[VideoMetadata]:
        """Extract metadata from video using ffprobe"""
        try:
            # Use ffprobe to get video metadata
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.logger.error(f"ffprobe failed for {video_path}")
                return None
            
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
            
            if not video_stream:
                self.logger.error(f"No video stream found in {video_path}")
                return None
            
            format_info = data.get('format', {})
            
            return VideoMetadata(
                filename=Path(video_path).name,
                filepath=video_path,
                duration=float(format_info.get('duration', 0)),
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                file_size=int(format_info.get('size', 0)),
                fps=eval(video_stream.get('r_frame_rate', '0/1'))  # Convert "30/1" to 30.0
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {video_path}: {e}")
            return None


class CSVManager:
    """Manage CSV file for video staging tool"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.logger = logging.getLogger(__name__)
    
    def load_processed_videos(self) -> set:
        """Load set of already processed video filenames"""
        processed = set()
        
        if not os.path.exists(self.csv_path):
            return processed
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'video_filename' in row:
                        processed.add(row['video_filename'])
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
        
        return processed
    
    def append_video(self, video_data: ProcessedVideo) -> bool:
        """Append new video entry to CSV"""
        try:
            file_exists = os.path.exists(self.csv_path)
            
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'video_filename', 'title', 'hook', 'description', 
                    'hashtags', 'full_caption', 'category', 'virality_score',
                    'duration', 'resolution', 'processed_timestamp'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(asdict(video_data))
            
            self.logger.info(f"✓ Added {video_data.video_filename} to CSV")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to CSV: {e}")
            return False


class CaptionGeneratorPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, videos_dir: str, output_csv: str, ollama_model: str = "qwen2.5-vl:7b"):
        self.videos_dir = Path(videos_dir)
        self.output_csv = output_csv
        
        self.video_processor = VideoProcessor()
        self.analyzer = OllamaVideoAnalyzer(model=ollama_model)
        self.csv_manager = CSVManager(output_csv)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Configure logging"""
        log_dir = Path('/home/claude/instagram-caption-generator/logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"caption_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def get_video_title(self, video_path: Path) -> str:
        """Extract title from video filename or use filename as fallback"""
        # Remove extension and replace underscores/hyphens with spaces
        title = video_path.stem.replace('_', ' ').replace('-', ' ')
        return title.title()
    
    def run(self, force_reprocess: bool = False):
        """
        Run the complete pipeline
        
        Args:
            force_reprocess: If True, reprocess all videos regardless of CSV state
        """
        self.logger.info("="*60)
        self.logger.info("Instagram Caption Generator Pipeline Starting")
        self.logger.info("="*60)
        
        # Check Ollama setup
        if not self.analyzer.check_model_availability():
            self.logger.info(f"Model {self.analyzer.model} not available, attempting to pull...")
            if not self.analyzer.pull_model():
                self.logger.error("Failed to set up model. Exiting.")
                return
        
        # Get list of video files
        video_files = list(self.videos_dir.glob('*.mp4')) + \
                     list(self.videos_dir.glob('*.mov')) + \
                     list(self.videos_dir.glob('*.avi'))
        
        if not video_files:
            self.logger.warning(f"No video files found in {self.videos_dir}")
            return
        
        self.logger.info(f"Found {len(video_files)} video files")
        
        # Load already processed videos
        processed_videos = set() if force_reprocess else self.csv_manager.load_processed_videos()
        self.logger.info(f"Already processed: {len(processed_videos)} videos")
        
        # Process each video
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for video_path in video_files:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing: {video_path.name}")
            self.logger.info(f"{'='*60}")
            
            # Skip if already processed
            if video_path.name in processed_videos and not force_reprocess:
                self.logger.info(f"⊙ Skipping (already processed)")
                skip_count += 1
                continue
            
            # Extract metadata
            metadata = self.video_processor.extract_metadata(str(video_path))
            if not metadata:
                self.logger.error(f"✗ Failed to extract metadata")
                fail_count += 1
                continue
            
            self.logger.info(f"  Duration: {metadata.duration:.2f}s")
            self.logger.info(f"  Resolution: {metadata.width}x{metadata.height}")
            
            # Get title
            title = self.get_video_title(video_path)
            self.logger.info(f"  Title: {title}")
            
            # Analyze video and generate caption
            caption_data = self.analyzer.analyze_video(str(video_path), title)
            
            if not caption_data:
                self.logger.error(f"✗ Failed to generate caption")
                fail_count += 1
                continue
            
            self.logger.info(f"  ✓ Generated caption:")
            self.logger.info(f"    Hook: {caption_data.hook}")
            self.logger.info(f"    Category: {caption_data.category}")
            self.logger.info(f"    Virality Score: {caption_data.virality_score}/100")
            self.logger.info(f"    Hashtags: {', '.join(caption_data.hashtags)}")
            
            # Create ProcessedVideo entry
            processed_video = ProcessedVideo(
                video_filename=video_path.name,
                title=title,
                hook=caption_data.hook,
                description=caption_data.description,
                hashtags=' '.join(caption_data.hashtags),
                full_caption=caption_data.full_caption,
                category=caption_data.category,
                virality_score=caption_data.virality_score,
                duration=metadata.duration,
                resolution=f"{metadata.width}x{metadata.height}",
                processed_timestamp=datetime.now().isoformat()
            )
            
            # Append to CSV
            if self.csv_manager.append_video(processed_video):
                success_count += 1
            else:
                fail_count += 1
        
        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Pipeline Complete")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"✓ Successfully processed: {success_count}")
        self.logger.info(f"⊙ Skipped (already done): {skip_count}")
        self.logger.info(f"✗ Failed: {fail_count}")
        self.logger.info(f"Output CSV: {self.output_csv}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Instagram captions for short-form videos using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in default directory
  python caption_generator.py
  
  # Process videos from specific directory
  python caption_generator.py --videos-dir /path/to/videos
  
  # Force reprocess all videos
  python caption_generator.py --force-reprocess
  
  # Use different model
  python caption_generator.py --model qwen3-vl
        """
    )
    
    parser.add_argument(
        '--videos-dir',
        default='/home/claude/instagram-caption-generator/videos',
        help='Directory containing video files (default: ./videos)'
    )
    
    parser.add_argument(
        '--output-csv',
        default='/home/claude/instagram-caption-generator/output/instagram_captions.csv',
        help='Output CSV file path (default: ./output/instagram_captions.csv)'
    )
    
    parser.add_argument(
        '--model',
        default='qwen2.5-vl:7b',
        help='Ollama model to use (default: qwen2.5-vl:7b)'
    )
    
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Reprocess all videos, even if already in CSV'
    )
    
    args = parser.parse_args()
    
    # Create pipeline and run
    pipeline = CaptionGeneratorPipeline(
        videos_dir=args.videos_dir,
        output_csv=args.output_csv,
        ollama_model=args.model
    )
    
    pipeline.run(force_reprocess=args.force_reprocess)


if __name__ == '__main__':
    main()
