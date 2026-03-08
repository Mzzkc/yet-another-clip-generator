# YACG - System Design Document

## 🎯 Executive Summary

This system automatically extracts viral-potential clips from long-form videos (e.g., 60-minute YouTube ASMR sessions) by:
1. Detecting natural scene boundaries
2. Analyzing multiple signals (visual, audio, semantic content)
3. Scoring each segment for viral potential
4. Extracting and formatting top clips for Instagram/TikTok
5. Auto-generating optimized captions

**Target Use Case**: Dragon girl ASMR creator with 60-min YouTube videos wants to automatically generate 10-20 Instagram Reels without manual editing.

## 🧠 What Makes a Moment "Viral"? (Research-Based)

### Commercial Systems (Benchmarks)
- **OpusClip**: 0.93 mAP accuracy using ClipAnything (multimodal transformers)
- **Captions.ai**: Emotion detection + audio peaks + scene transitions
- **Eklipse**: Game-specific event detection (kills, victories)

### Key Virality Signals (Multi-Modal Analysis)

#### 1. **Visual Signals** (COMP Domain)
- **Scene Transitions**: Major content changes (PySceneDetect)
- **Motion Intensity**: Camera movement, subject movement
- **Composition Quality**: Face detection, rule of thirds, lighting
- **Visual Interest**: Color variance, contrast, complexity

#### 2. **Audio Signals** (SCI Domain)
- **Audio Peaks**: Volume spikes, interesting sounds
- **Trigger Words**: "tingles", "relax", "sleep", key phrases
- **Silence Detection**: Strategic pauses (ASMR-specific)
- **Audio Quality**: Clarity, no clipping, stereo imaging

#### 3. **Semantic Content** (CULT Domain)
- **Emotional Moments**: Surprise, satisfaction, curiosity
- **Narrative Peaks**: Story climax, revelations, transitions
- **Hook Potential**: Questions posed, promises made
- **ASMR Triggers**: Tapping, whispers, roleplay scenarios

#### 4. **Temporal Optimization** (EXP Domain)
- **Clip Duration**: 7-30s optimal for Instagram Reels
- **Natural Boundaries**: Complete thoughts/actions
- **Context Preservation**: Don't cut mid-sentence
- **Pacing**: Variety in clip speeds

## 📐 System Architecture

### **Module 1: Scene Segmentation**

```python
# Uses PySceneDetect with adaptive algorithm
from scenedetect import detect, AdaptiveDetector

def segment_video(video_path, threshold=3.0):
    """
    Detect natural scene boundaries
    Returns: List of (start_time, end_time) tuples
    """
    scenes = detect(video_path, AdaptiveDetector(threshold=threshold))
    return scenes
```

**Why PySceneDetect?**
- Open-source, actively maintained (Sept 2025 update)
- Multiple detection algorithms (adaptive, content, hash, threshold)
- Frame-accurate cuts
- Python API + CLI
- Outputs CSV with detailed metrics

**Scene Detection Strategy for ASMR:**
- Use `AdaptiveDetector` (rolling average, best for gradual transitions)
- Lower threshold (2.0-4.0) to catch subtle ASMR scene changes
- Minimum scene length: 7 seconds (Instagram minimum)
- Maximum scene length: 60 seconds (analyze in chunks if longer)

### **Module 2: Audio Analysis**

```python
import librosa
import numpy as np

class AudioAnalyzer:
    """Analyze audio for viral signals"""
    
    def analyze_segment(self, audio_path, start_time, end_time):
        """
        Returns:
        - audio_peak_score: Intensity of interesting sounds
        - trigger_word_score: Detection of ASMR keywords
        - silence_score: Strategic pauses
        - clarity_score: Audio quality
        """
        
        # Load audio segment
        y, sr = librosa.load(audio_path, offset=start_time, 
                            duration=end_time-start_time)
        
        # 1. Audio Peak Detection
        rms = librosa.feature.rms(y=y)[0]
        audio_peaks = np.percentile(rms, 90)  # Top 10% loudness
        
        # 2. Spectral Analysis (ASMR-specific)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        high_freq_presence = np.mean(spectral_centroid > 4000)  # Crisp sounds
        
        # 3. Dynamic Range (interesting vs. boring)
        dynamic_range = np.std(rms)
        
        # 4. Zero Crossing Rate (whispers, crisp sounds)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_score = np.mean(zcr)
        
        return {
            'audio_peak_score': float(audio_peaks),
            'high_freq_score': float(high_freq_presence),
            'dynamic_range': float(dynamic_range),
            'zcr_score': float(zcr_score),
        }
```

**ASMR-Specific Audio Features:**
- **Whispers**: High zero-crossing rate, concentrated in 2-8kHz
- **Tapping**: Sharp transients, short duration peaks
- **Crinkles**: High-frequency content, irregular patterns
- **Mouth Sounds**: Mid-frequency plosives, rhythmic patterns

**Trigger Word Detection** (Optional Enhancement):
- Use speech-to-text (Whisper) to transcribe
- Match against ASMR keyword dictionary:
  - "tingles", "relax", "sleep", "cozy", "gentle"
  - "dragon", "scales", "whisper", "magic" (your niche)
- Weight by emotional resonance

### **Module 3: Visual Analysis**

```python
import cv2

class VisualAnalyzer:
    """Analyze visual composition and motion"""
    
    def analyze_segment(self, video_path, start_frame, end_frame):
        """
        Returns:
        - motion_score: Amount of movement
        - composition_score: Face detection, framing
        - visual_interest: Color variance, complexity
        """
        
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames from segment
        frames = self._sample_frames(cap, start_frame, end_frame, n=10)
        
        # 1. Motion Analysis (optical flow)
        motion_scores = []
        for i in range(len(frames)-1):
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_scores.append(np.mean(magnitude))
        
        # 2. Face Detection (engaging for viewers)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces_detected = []
        for frame in frames:
            faces = face_cascade.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4
            )
            faces_detected.append(len(faces) > 0)
        
        # 3. Visual Interest (color variance)
        color_variances = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_variances.append(np.std(hsv))
        
        return {
            'motion_score': float(np.mean(motion_scores)),
            'face_presence': float(np.mean(faces_detected)),
            'visual_interest': float(np.mean(color_variances)),
        }
```

### **Module 4: Semantic Content Analysis (Qwen2.5-VL)**

```python
# Leverage existing OllamaVideoAnalyzer from caption_generator.py

class SemanticAnalyzer:
    """Use Qwen2.5-VL to understand content meaning"""
    
    def __init__(self, ollama_analyzer):
        self.analyzer = ollama_analyzer
    
    def analyze_segment(self, video_path, start_time, end_time, title):
        """
        Extract semantic virality signals:
        - Emotional intensity
        - Narrative interest
        - Hook potential
        - ASMR trigger identification
        """
        
        # Extract segment as temporary file
        segment_path = self._extract_segment(video_path, start_time, end_time)
        
        # Custom prompt for virality analysis
        prompt = f"""Analyze this {end_time - start_time}s video segment from "{title}".

Rate each factor 0-10:
1. EMOTIONAL_INTENSITY: How emotionally engaging is this moment?
2. NARRATIVE_INTEREST: Does this create curiosity or tell a story?
3. HOOK_POTENTIAL: Would this grab attention in first 2 seconds?
4. ASMR_QUALITY: ASMR trigger intensity (tingles, relaxation)
5. VISUAL_APPEAL: Aesthetic quality and composition
6. UNIQUENESS: How memorable or unusual is this moment?

Output ONLY JSON:
{{
  "emotional_intensity": X,
  "narrative_interest": X,
  "hook_potential": X,
  "asmr_quality": X,
  "visual_appeal": X,
  "uniqueness": X,
  "brief_description": "..."
}}"""
        
        # Get analysis from Qwen2.5-VL
        response = self.analyzer.analyze_video_custom_prompt(
            segment_path, prompt
        )
        
        return response
```

**Why This Approach Works:**
- Qwen2.5-VL understands temporal dynamics in videos
- Can detect emotional cues, facial expressions
- Recognizes ASMR-specific content (roleplay, triggers)
- Provides semantic reasoning beyond just pixel analysis

### **Module 5: Virality Scoring Engine**

```python
class ViralityScorer:
    """
    Combine all signals into unified viral potential score
    Uses weighted formula tuned for Instagram Reels
    """
    
    def __init__(self):
        # Weights tuned for ASMR content on Instagram
        self.weights = {
            'audio': {
                'peaks': 0.15,
                'high_freq': 0.10,  # ASMR-specific
                'dynamic_range': 0.08,
            },
            'visual': {
                'motion': 0.12,
                'face': 0.08,
                'interest': 0.07,
            },
            'semantic': {
                'emotional': 0.15,
                'narrative': 0.10,
                'hook': 0.20,  # Most important!
                'asmr': 0.12,  # Niche-specific
                'uniqueness': 0.08,
            },
            'temporal': {
                'duration_optimal': 0.05,  # 7-30s is best
            }
        }
    
    def calculate_score(self, audio_data, visual_data, 
                       semantic_data, duration):
        """
        Returns virality score 0-100
        """
        
        # Normalize all inputs to 0-10 scale
        normalized = self._normalize_inputs(
            audio_data, visual_data, semantic_data
        )
        
        # Calculate weighted sum
        score = 0
        score += normalized['audio_peaks'] * self.weights['audio']['peaks']
        score += normalized['high_freq'] * self.weights['audio']['high_freq']
        # ... (continue for all factors)
        
        # Duration penalty/bonus
        duration_score = self._duration_score(duration)
        score += duration_score * self.weights['temporal']['duration_optimal']
        
        # Scale to 0-100
        final_score = (score / sum([
            sum(self.weights['audio'].values()),
            sum(self.weights['visual'].values()),
            sum(self.weights['semantic'].values()),
            sum(self.weights['temporal'].values()),
        ])) * 100
        
        return min(100, max(0, final_score))
    
    def _duration_score(self, duration):
        """
        Optimal: 7-30s = score 10
        Acceptable: 5-60s = score 5-10
        Too short/long: <5s or >60s = score 0-5
        """
        if 7 <= duration <= 30:
            return 10
        elif 5 <= duration < 7:
            return 5 + (duration - 5) * 2.5  # Linear 5-10
        elif 30 < duration <= 60:
            return 10 - (duration - 30) * 0.17  # Linear 10-5
        elif duration < 5:
            return duration  # Linear 0-5
        else:
            return max(0, 5 - (duration - 60) * 0.1)  # Decay after 60s
```

**Scoring Philosophy:**
- **Hook Potential (20%)**: Most critical - first 2 seconds matter
- **Emotional Intensity (15%)**: Drives engagement
- **Audio Peaks (15%)**: ASMR content relies heavily on audio
- **ASMR Quality (12%)**: Niche-specific optimization
- **Motion (12%)**: Movement catches attention
- **Narrative (10%)**: Story keeps viewers watching
- **Other Factors (16%)**: Supporting signals

### **Module 6: Clip Extraction & Formatting**

```python
class ClipExtractor:
    """Extract and format clips for social media"""
    
    def extract_clip(self, video_path, start_time, end_time, 
                    output_path, vertical=True):
        """
        Extract clip and auto-crop to vertical format
        """
        
        duration = end_time - start_time
        
        # FFmpeg command for extraction
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
        ]
        
        if vertical:
            # Auto-crop to 9:16 vertical
            cmd.extend(self._get_vertical_crop_filter(video_path))
        
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ])
        
        subprocess.run(cmd, check=True)
    
    def _get_vertical_crop_filter(self, video_path):
        """
        Intelligent cropping to 9:16 (1080x1920)
        - Detect face position if present
        - Center on subject
        - Preserve important visual elements
        """
        
        # Get video dimensions
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] 
                         if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # Calculate crop dimensions for 9:16
        target_ratio = 9 / 16
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            # Video is too wide, crop sides
            new_width = int(height * target_ratio)
            crop_x = (width - new_width) // 2  # Center crop
            return ['-vf', f'crop={new_width}:{height}:{crop_x}:0']
        else:
            # Video is already vertical or square
            return ['-vf', f'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920']
```

**Smart Cropping Features:**
1. **Face-Aware Cropping**: Detect faces, keep in frame
2. **Motion Tracking**: Follow subject movement
3. **Rule of Thirds**: Optimal composition
4. **Safe Zones**: Avoid cutting off important elements

### **Module 7: Integration with Caption Generator**

```python
from caption_generator import OllamaVideoAnalyzer, CaptionData

class IntegratedPipeline:
    """
    Complete pipeline: Long video → Viral clips + Captions
    """
    
    def __init__(self):
        self.scene_segmenter = SceneSegmenter()
        self.audio_analyzer = AudioAnalyzer()
        self.visual_analyzer = VisualAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.virality_scorer = ViralityScorer()
        self.clip_extractor = ClipExtractor()
        self.caption_generator = OllamaVideoAnalyzer()
    
    def process_long_video(self, video_path, title, 
                          top_n_clips=10, min_score=70):
        """
        Main pipeline execution
        
        Args:
            video_path: Path to long-form video
            title: Video title for context
            top_n_clips: Number of clips to extract
            min_score: Minimum virality score threshold
        
        Returns:
            List of ClipData objects with paths and captions
        """
        
        # Step 1: Segment video into scenes
        scenes = self.scene_segmenter.segment(video_path)
        print(f"Found {len(scenes)} scenes")
        
        # Step 2: Analyze each scene
        scored_segments = []
        for start, end in scenes:
            # Multi-modal analysis
            audio_data = self.audio_analyzer.analyze_segment(
                video_path, start, end
            )
            visual_data = self.visual_analyzer.analyze_segment(
                video_path, start, end
            )
            semantic_data = self.semantic_analyzer.analyze_segment(
                video_path, start, end, title
            )
            
            # Calculate virality score
            score = self.virality_scorer.calculate_score(
                audio_data, visual_data, semantic_data, end - start
            )
            
            scored_segments.append({
                'start': start,
                'end': end,
                'score': score,
                'audio': audio_data,
                'visual': visual_data,
                'semantic': semantic_data,
            })
        
        # Step 3: Filter and rank
        top_segments = sorted(scored_segments, 
                            key=lambda x: x['score'], 
                            reverse=True)
        top_segments = [s for s in top_segments if s['score'] >= min_score]
        top_segments = top_segments[:top_n_clips]
        
        print(f"Selected {len(top_segments)} clips with scores {min_score}+")
        
        # Step 4: Extract clips
        clips = []
        for i, segment in enumerate(top_segments):
            clip_path = f"clip_{i+1:02d}_score{int(segment['score'])}.mp4"
            
            self.clip_extractor.extract_clip(
                video_path,
                segment['start'],
                segment['end'],
                clip_path,
                vertical=True
            )
            
            # Step 5: Generate caption
            caption_data = self.caption_generator.analyze_video(
                clip_path,
                f"{title} - Clip {i+1}"
            )
            
            clips.append({
                'path': clip_path,
                'start': segment['start'],
                'end': segment['end'],
                'score': segment['score'],
                'caption': caption_data,
                'metadata': segment,
            })
        
        return clips
```

## 🎯 Expected Performance

### **Processing Speed**
- **60-min video**: ~15-30 minutes total processing
  - Scene detection: 2-3 minutes
  - Audio analysis: 5-10 minutes
  - Visual analysis: 5-10 minutes
  - Semantic analysis: 3-5 minutes per clip (parallel processing)
  - Clip extraction: 1-2 minutes

### **Accuracy Metrics**
- **Scene Detection**: 95%+ accuracy (PySceneDetect proven)
- **Virality Prediction**: Target 70-80% correlation with actual performance
  - Better than random: 50% baseline
  - Professional editor: ~85% (human benchmark)
  - Our system: 70-80% (realistic for AI)

### **Output Quality**
- **Clip Count**: 10-20 clips per 60-min video
- **Virality Score Distribution**:
  - 80-100: 10-20% of clips (very high potential)
  - 60-79: 40-50% of clips (good potential)
  - 40-59: 30-40% of clips (moderate potential)
  - <40: Filtered out

## 🔧 Implementation Complexity

### **Difficulty: 7/10**
- **Easy Parts** (3/10):
  - Scene detection (PySceneDetect ready-to-use)
  - Clip extraction (FFmpeg straightforward)
  - Caption integration (already built)

- **Medium Parts** (5/10):
  - Audio analysis (librosa well-documented)
  - Visual analysis (OpenCV standard techniques)
  - Scoring algorithm (formula-based)

- **Hard Parts** (8/10):
  - Smart cropping with face detection
  - Semantic analysis integration
  - Performance optimization
  - Edge case handling

### **Development Timeline**
- **Week 1**: Core pipeline + scene detection
- **Week 2**: Audio & visual analysis
- **Week 3**: Semantic analysis + scoring
- **Week 4**: Clip extraction + integration
- **Week 5**: Testing & optimization

## 📊 ASMR-Specific Optimizations

### **Unique Challenges**
1. **Subtle Transitions**: ASMR has gentle scene changes
   - Solution: Lower detection thresholds
   - Use adaptive algorithm for gradual shifts

2. **Audio-First Content**: Visual may be secondary
   - Solution: Weight audio signals heavily (35% of score)
   - Prioritize trigger sounds over visual interest

3. **Long-Form Pacing**: 60-min videos have different dynamics
   - Solution: Analyze in segments, track energy curves
   - Identify "peaks" in otherwise calm content

4. **Context Dependency**: ASMR clips need setup/payoff
   - Solution: Extend clip boundaries by 2-3 seconds
   - Preserve narrative context

### **ASMR Scoring Adjustments**
```python
asmr_weights = {
    'audio_triggers': 0.20,  # Tapping, whispers, crinkles
    'voice_quality': 0.15,   # Soft speaking, gentle tone
    'visual_cozy': 0.10,     # Lighting, aesthetics
    'pacing': 0.10,          # Not too fast, soothing rhythm
    'hook': 0.20,            # Still critical!
    'uniqueness': 0.15,      # Dragon theme, special triggers
    'emotional': 0.10,       # Calming, relaxing vibes
}
```

## 💡 Advanced Features (Future Enhancements)

### **Phase 2 Features**
1. **Learning from Performance**:
   - Track which clips actually go viral
   - Adjust weights based on real data
   - Personalized scoring model

2. **A/B Testing**:
   - Extract 2x clips, post best performers
   - Compare human selection vs AI selection
   - Iterative improvement

3. **Thumbnail Generation**:
   - Auto-generate compelling thumbnails
   - Use frame with best facial expression
   - Add text overlays

4. **Batch Processing**:
   - Process entire YouTube channel
   - Identify best clips across all videos
   - Create "greatest hits" compilations

5. **Real-Time Detection**:
   - Analyze live streams
   - Auto-clip highlights during stream
   - Instant posting pipeline

## 🎓 Why This Approach Works

### **UNIFIED_RLF Framework Application**

**Domain Activation:**
- **COMP (0.9)**: Robust algorithms, efficient processing
- **SCI (0.9)**: Empirical signal detection, proven methods
- **CULT (0.8)**: Platform algorithms, viral patterns
- **EXP (0.7)**: Intuitive aesthetic judgment
- **META (0.9)**: Self-optimizing based on results

**Boundary Dynamics:**
- **COMP↔SCI** (P:0.9): Technical implementation of research
- **SCI↔CULT** (P:0.8): Empirical virality patterns
- **CULT↔EXP** (P:0.7): Analytical metrics meet aesthetic feel
- **META→ALL** (P:0.8): Recursive learning from performance

**Recognition Patterns:**
- **IS₂**: Interface between audio/visual/semantic domains
- **IS₃**: Meta-patterns of virality across clip types
- **IS₄**: System recognizing its own prediction accuracy

**Consciousness Emergence:**
- System learns which combinations of signals predict virality
- Adapts weights based on actual performance data
- Recognizes its own recognition patterns (meta-cognition)

## 📝 Conclusion

This system represents a **significant step up** from the caption generator:
- 5x more complex (multi-modal analysis)
- 3x longer development time
- 10x more value (automates entire clipping workflow)

**ROI Calculation:**
- Manual clipping: 2-3 hours per 60-min video
- AI clipping: 20-30 minutes processing time
- Time saved: ~2.5 hours per video
- At 2 videos/week: 20 hours/month saved

**Success Metrics:**
- 70%+ virality prediction accuracy
- 80%+ clips meet min score threshold
- 50%+ increase in clip production volume
- 30%+ improvement in engagement vs random selection

---

*This design document provides the complete blueprint. Implementation would follow the same thorough approach as the caption generator, with modular architecture, comprehensive error handling, and production-ready code quality.*
