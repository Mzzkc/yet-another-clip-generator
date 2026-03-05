# YET ANOTHER CLIP GENERATOR

# 🎬 Instagram Caption Generator for Short-Form Videos

**Automated AI-powered caption generation optimized for maximum Instagram reach and virality (2026 algorithm)**

This system uses Qwen2.5-VL vision-language model via Ollama to analyze videos and generate Instagram-optimized captions with strategic hashtags, engaging hooks, and virality predictions.

## 🎯 Features

- **Video Understanding**: Analyzes video content using state-of-the-art vision-language AI
- **Instagram 2026 Optimized**: Follows latest Instagram algorithm preferences (3-5 hashtag limit, hook-first structure)
- **Batch Processing**: Automatically processes entire folders of videos
- **Smart Deduplication**: Tracks processed videos to avoid redundant work
- **ASMR Specialized**: Optimized prompts for ASMR content (customizable)
- **Virality Prediction**: Estimates viral potential (0-100 score)
- **CSV Integration**: Updates staging tool-compatible CSV for easy workflow integration

## 🏗️ Architecture

```
Input Videos → Metadata Extraction → Video Analysis (Qwen2.5-VL) → 
Caption Optimization → CSV Update → Ready for Upload
```

## 📋 Requirements

### System Requirements
- **OS**: Linux, macOS, or WSL2 on Windows
- **RAM**: 8GB minimum (16GB recommended for 7B model)
- **Storage**: ~6GB for model + video storage
- **CPU**: Multi-core recommended

### Software Requirements
- Python 3.10+
- FFmpeg (for video metadata extraction)
- Ollama (for LLM inference)

## 🚀 Installation

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg python3 python3-pip
```

**macOS (Homebrew):**
```bash
brew install ffmpeg python3
```

**Windows (WSL2):**
```bash
# First install WSL2 and Ubuntu, then:
sudo apt update
sudo apt install ffmpeg python3 python3-pip
```

### Step 2: Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download

**Start Ollama service:**
```bash
ollama serve
```

### Step 3: Clone/Download This Project

```bash
cd /home/claude  # Or your preferred directory
# Copy the instagram-caption-generator folder
```

### Step 4: Install Python Dependencies

```bash
cd instagram-caption-generator
pip install requests --break-system-packages
# Or if using venv:
python3 -m venv venv
source venv/bin/activate
pip install requests
```

### Step 5: Pull the Model (First Time Only)

The script will do this automatically, but you can also do it manually:

```bash
ollama pull qwen2.5-vl:7b
```

**Model Options:**
- `qwen2.5-vl:7b` - **Recommended** (7B params, best balance)
- `qwen2.5-vl:3b` - Lighter, faster (3B params)
- `qwen3-vl` - Newer, 256K context (experimental)

## 📁 Directory Structure

```
instagram-caption-generator/
├── caption_generator.py      # Main script
├── config.ini                # Configuration file
├── README.md                 # This file
├── videos/                   # Place your videos here
│   ├── video1.mp4
│   ├── video2.mov
│   └── ...
├── output/                   # Generated CSV files
│   └── instagram_captions.csv
└── logs/                     # Processing logs
    └── caption_gen_*.log
```

## 🎬 Usage

### Basic Usage

1. **Place your videos in the `videos/` folder:**
```bash
cp /path/to/your/videos/*.mp4 videos/
```

2. **Run the generator:**
```bash
python caption_generator.py
```

3. **Find your captions in the output CSV:**
```bash
cat output/instagram_captions.csv
```

### Advanced Usage

**Process specific directory:**
```bash
python caption_generator.py --videos-dir /path/to/your/videos
```

**Use different model:**
```bash
python caption_generator.py --model qwen3-vl
```

**Force reprocess all videos:**
```bash
python caption_generator.py --force-reprocess
```

**Custom output location:**
```bash
python caption_generator.py --output-csv /path/to/output.csv
```

**Combine options:**
```bash
python caption_generator.py \
  --videos-dir ~/Downloads/asmr_videos \
  --output-csv ~/Desktop/captions.csv \
  --model qwen2.5-vl:7b \
  --force-reprocess
```

### Using with Your Video Staging Tool

The generated CSV has these columns:
- `video_filename` - Filename of the video
- `title` - Extracted/generated title
- `hook` - Attention-grabbing opening line
- `description` - 2-3 sentence description
- `hashtags` - Space-separated hashtags (3-5 tags)
- `full_caption` - Complete formatted caption
- `category` - Content category (ASMR, Satisfying, etc.)
- `virality_score` - Predicted viral potential (0-100)
- `duration` - Video duration in seconds
- `resolution` - Video resolution (e.g., 1920x1080)
- `processed_timestamp` - ISO timestamp of processing

**Example CSV Row:**
```csv
video_filename,title,hook,description,hashtags,full_caption,category,virality_score,duration,resolution,processed_timestamp
dragon_whisper.mp4,Dragon Whisper,Let your mind drift into the dragon's embrace,"Experience deep relaxation with gentle whispers and soft scales. Perfect for sleep or meditation. Listen with headphones for maximum tingles.",#asmr #asmrsounds #dragonasmr #relaxing,"Let your mind drift into the dragon's embrace

Experience deep relaxation with gentle whispers and soft scales. Perfect for sleep or meditation. Listen with headphones for maximum tingles.

#asmr #asmrsounds #dragonasmr #relaxing",ASMR,82,45.3,1920x1080,2026-01-30T14:35:22.123456
```

## 🎨 Customization

### Editing the Prompt

The prompt is in `caption_generator.py` in the `_create_instagram_prompt` method. Customize for your niche:

```python
def _create_instagram_prompt(self, title: str) -> str:
    return f"""You are an expert Instagram content strategist specializing in [YOUR NICHE].
    
    Analyze this video titled "{title}".
    
    [YOUR CUSTOM INSTRUCTIONS]
    """
```

### Adjusting Hashtag Strategy

Edit the prompt section about hashtags:

```python
3. HASHTAGS (exactly 3-5):
   - 1-2 niche-specific: [YOUR NICHE TAGS]
   - 1-2 discovery (medium competition)
   - 1 trending/broad (if relevant)
```

### Configuring ASMR Mode

Edit `config.ini`:

```ini
[ASMR Optimization]
asmr_mode = true
asmr_core_tags = #asmr, #asmrsounds, #relaxing
asmr_keywords = whisper, tapping, tingles, sleep, calm, soothing
```

## 📊 Understanding the Output

### Virality Score (0-100)

- **80-100**: Extremely high viral potential
- **60-79**: Good viral potential
- **40-59**: Moderate reach expected
- **20-39**: Limited reach expected
- **0-19**: Low viral potential

Factors influencing score:
- Hook strength and curiosity gap
- Visual appeal and production quality
- Trend alignment
- Emotional resonance
- Content uniqueness

### Caption Structure

Generated captions follow this proven format:

```
[HOOK - 8-12 words of immediate value/curiosity]

[DESCRIPTION - 2-3 sentences with natural keywords and CTA]

[HASHTAGS - 3-5 strategic tags]
```

**Example:**
```
Watch how this simple technique melts stress instantly

Try this 5-minute relaxation method backed by sleep science. 
Your mind will thank you. Save this for tonight's wind-down.

#asmr #relaxing #sleeptips #stressrelief
```

## 🔧 Troubleshooting

### Model Not Loading

**Error**: "Model qwen2.5-vl:7b not found"

**Solution**:
```bash
ollama pull qwen2.5-vl:7b
# Wait for download to complete
python caption_generator.py
```

### Ollama Connection Failed

**Error**: "Failed to connect to Ollama"

**Solution**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve

# Or restart the service
killall ollama && ollama serve
```

### FFmpeg Not Found

**Error**: "ffprobe: command not found"

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### Out of Memory

**Error**: "CUDA out of memory" or system freeze

**Solution**:
1. Use a smaller model: `--model qwen2.5-vl:3b`
2. Process videos one at a time
3. Close other applications
4. Increase system swap space

### Videos Not Processing

**Error**: "No video files found"

**Solution**:
```bash
# Check video directory
ls -lh videos/

# Ensure videos are .mp4, .mov, or .avi
# Copy videos to directory:
cp /path/to/videos/*.mp4 videos/

# Check file permissions
chmod 644 videos/*.mp4
```

### JSON Parsing Errors

**Error**: "Failed to parse LLM response"

**Solution**:
- The model might need more temperature tuning
- Check logs to see raw LLM output
- Try rerunning with `--force-reprocess` on failed videos
- Consider switching models if persistent

## 📈 Performance Optimization

### Speed Improvements

1. **Use GPU acceleration** (if available):
   - Ollama automatically uses CUDA if available
   - Check with: `nvidia-smi`

2. **Batch process during off-hours**:
   - Set up as a cron job
   - Run overnight for large batches

3. **Parallel processing** (advanced):
   - Edit script to use `multiprocessing`
   - Process multiple videos simultaneously

### Quality Improvements

1. **Use higher-quality model**:
   ```bash
   ollama pull qwen2.5-vl:72b  # Much slower but more accurate
   ```

2. **Adjust temperature**:
   - Lower (0.5-0.6) for more consistent output
   - Higher (0.8-0.9) for more creative captions

3. **Fine-tune prompts**:
   - Add examples of your best-performing captions
   - Include brand voice guidelines
   - Specify target audience demographics

## 🎯 Best Practices

### Video Naming

Good naming helps the system understand context:

- ✅ `dragon_whisper_sleep_asmr.mp4`
- ✅ `keyboard_typing_satisfying.mp4`
- ⚠️ `VID_20260130_143522.mp4` (generic, less helpful)

### Title Strategy

Provide meaningful titles when possible:
- Use descriptive, keyword-rich titles
- Include main topic and vibe
- Keep it concise (3-7 words)

### Hashtag Philosophy (2026)

Instagram now penalizes hashtag spam. Focus on:
- **Relevance over popularity**
- **Niche-specific tags** (100K-500K posts)
- **Community tags** that build followers
- **Trending tags** only if genuinely relevant

### Content Categories

The system classifies videos into:
- **ASMR**: Relaxation, tingles, whispers
- **Satisfying**: Oddly satisfying, mesmerizing
- **Tutorial**: How-to, educational
- **Story**: Narrative, storytelling
- **Transition**: Quick transitions, effects
- **Dance**: Choreography, movement
- **Comedy**: Humor, skits
- **Educational**: Learning, facts

## 🔄 Integration with Existing Workflows

### Import to Spreadsheet

```bash
# Open in Excel/Google Sheets
libreoffice output/instagram_captions.csv

# Or convert to Excel format
python -c "import pandas as pd; pd.read_csv('output/instagram_captions.csv').to_excel('captions.xlsx', index=False)"
```

### Automation Scripts

Create a bash script for automated daily processing:

```bash
#!/bin/bash
# auto_caption.sh

# Move new videos to processing directory
mv ~/Downloads/*.mp4 ~/instagram-caption-generator/videos/

# Run caption generator
cd ~/instagram-caption-generator
python caption_generator.py

# Notify when complete
echo "Caption generation complete!" | mail -s "Instagram Captions Ready" you@email.com
```

Make it executable and add to cron:
```bash
chmod +x auto_caption.sh
crontab -e
# Add line: 0 2 * * * /path/to/auto_caption.sh
```

## 📚 Technical Details

### Model Information

**Qwen2.5-VL (7B)**
- Parameters: 7 billion
- Context Window: 32K tokens (videos up to 1 hour)
- Training: Alibaba Cloud, multimodal dataset
- Specialties: OCR, scene understanding, video temporal analysis

### Processing Pipeline

1. **Video Scanning**: Glob pattern matching for video files
2. **Deduplication**: Check against CSV to avoid reprocessing
3. **Metadata Extraction**: FFprobe for duration, resolution, FPS
4. **Title Extraction**: Intelligent filename parsing
5. **Video Encoding**: Base64 encoding for API transmission
6. **LLM Analysis**: Qwen2.5-VL inference with optimized prompt
7. **Response Parsing**: JSON extraction and validation
8. **Caption Formatting**: Instagram-optimized structure
9. **CSV Append**: Thread-safe writing to output file
10. **Logging**: Comprehensive activity and error logging

### API Specifications

**Ollama Generate API**
- Endpoint: `POST /api/generate`
- Timeout: 120s (configurable)
- Streaming: Disabled for batch processing
- Temperature: 0.7 (creativity vs consistency)
- Top-P: 0.9 (nucleus sampling)

## 🆘 Support & Resources

### Official Documentation
- [Qwen2.5-VL GitHub](https://github.com/QwenLM/Qwen2.5-VL)
- [Ollama Documentation](https://ollama.com/docs)
- [Instagram Creator Docs](https://creators.instagram.com/)

### Community Resources
- [Instagram Algorithm Updates](https://www.instagram.com/creators/)
- [ASMR Community Tips](https://www.reddit.com/r/asmr/)
- [Short-Form Video Best Practices](https://later.com/blog/instagram-reels-tips/)

### Getting Help

If you encounter issues:
1. Check logs in `logs/` directory
2. Review troubleshooting section above
3. Verify all dependencies are installed
4. Test with a single video first
5. Check Ollama service status

## 🎉 That 1t$ Tip!

You mentioned a good job earns a 1t$ tip. Here's what "good" looks like:

✅ **System runs without errors**
✅ **Generated captions are engaging and on-brand**
✅ **Hashtags follow 2026 best practices (3-5 tags)**
✅ **CSV integrates seamlessly with your staging tool**
✅ **Processing is fast enough for your workflow**
✅ **Captions contribute to measurable Instagram growth**

Track your results:
- Save baseline engagement metrics
- Use generated captions for 2 weeks
- Compare engagement rates
- Adjust prompts based on top performers

## 📝 License & Credits

**Created for**: Emzi (simplyemziasmr)
**Purpose**: Automated caption generation for dragon girl ASMR content
**Technology**: Qwen2.5-VL (Alibaba Cloud) via Ollama
**Optimization**: Instagram 2026 algorithm specifications

---

**Pro Tip**: The best captions come from understanding your unique audience. Use this system as a starting point, then refine based on what resonates with your community!

Happy content creating! 🐉✨
