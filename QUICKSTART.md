# 🚀 Quick Start Guide

**Get up and running in 5 minutes!**

## Step 1: Install Dependencies (One-Time Setup)

```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
# OR
brew install ffmpeg      # macOS

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
pip install requests --break-system-packages
```

## Step 2: Start Ollama

```bash
# Start Ollama service (in a new terminal)
ollama serve

# In another terminal, pull the model
ollama pull qwen2.5-vl:7b
```

## Step 3: Add Your Videos

```bash
# Copy your videos to the processing directory
cp /path/to/your/videos/*.mp4 /home/claude/instagram-caption-generator/videos/
```

## Step 4: Run the Generator

```bash
cd /home/claude/instagram-caption-generator

# Test your setup first
python test_system.py

# If all tests pass, generate captions!
python caption_generator.py
```

## Step 5: Get Your Captions

```bash
# View the generated CSV
cat output/instagram_captions.csv

# Or open in a spreadsheet app
libreoffice output/instagram_captions.csv
```

## 🎯 Example Workflow

**For ASMR content creators:**

1. Record your ASMR videos
2. Export as MP4 files with descriptive names:
   - `dragon_whisper_sleep.mp4`
   - `keyboard_tapping_study.mp4`
   - `page_turning_relax.mp4`

3. Copy to processing folder:
```bash
cp ~/Videos/ASMR/*.mp4 ~/instagram-caption-generator/videos/
```

4. Generate captions:
```bash
python caption_generator.py
```

5. Review output:
```bash
cat output/instagram_captions.csv | column -t -s,
```

6. Copy captions to your video staging tool or Instagram directly!

## ⚡ Pro Tips

**Speed up processing:**
- Use the 3B model for faster results: `--model qwen2.5-vl:3b`
- Process videos overnight for large batches

**Improve caption quality:**
- Name your files descriptively
- Add keywords to filenames
- Keep videos under 60 seconds for best analysis

**Optimize for your niche:**
- Edit the prompt in `caption_generator.py` (line ~270)
- Adjust hashtag strategy for your audience
- Tune temperature in config

## 🆘 Quick Troubleshooting

**"Model not found"**
```bash
ollama pull qwen2.5-vl:7b
```

**"Ollama connection failed"**
```bash
# Check if running
ps aux | grep ollama

# Start if not running
ollama serve
```

**"No videos found"**
```bash
ls -lh videos/
# Make sure files are .mp4, .mov, or .avi
```

## 📊 Understanding Your Results

Each video gets:
- **Hook**: Attention-grabbing opening line
- **Description**: 2-3 sentence caption with keywords
- **Hashtags**: 3-5 strategic tags (Instagram 2026 limit)
- **Virality Score**: 0-100 prediction of viral potential
- **Category**: Content classification

**High virality scores (80+)** = Strong hooks, trending topics, emotional appeal
**Low virality scores (<40)** = May need better hooks or trend alignment

## 🎉 You're Ready!

Your automated caption generation system is now operational. Process videos in batches, integrate with your workflow, and watch your Instagram engagement grow!

**Need more details?** Check out README.md for complete documentation.
