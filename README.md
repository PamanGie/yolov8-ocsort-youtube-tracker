# YOLOv8 + OC-SORT YouTube Stream Tracker

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)

Real-time object detection and tracking system that works with **YouTube live streams** and local video files. Built with YOLOv8 for detection and OC-SORT for multi-object tracking.

## 🌟 Features

- 🎥 **YouTube Live Stream Support** - Process any YouTube live stream in real-time
- 🚀 **Real-time Object Detection** - YOLOv8 powered detection with customizable confidence
- 🎯 **Multi-Object Tracking** - OC-SORT tracker for consistent object IDs
- 📊 **Live Metrics** - Real-time FPS, object count, and tracking statistics
- 🔄 **Auto-reconnect** - Automatic reconnection for unstable streams
- 💾 **Video Export** - Save processed video with tracking overlays
- 🎛️ **Flexible Quality** - Choose stream quality (480p/720p/1080p)
- 🖥️ **Live Preview** - Interactive window with real-time visualization

## 🎯 Use Cases

- **Traffic Monitoring** - Count vehicles and analyze traffic patterns
- **Surveillance Analysis** - Monitor public areas from CCTV streams  
- **Crowd Counting** - Track people in public spaces
- **Incident Detection** - Real-time monitoring for unusual activities
- **Research & Education** - Computer vision learning with live data

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for better performance)
- Stable internet connection for YouTube streams

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolov8-ocsort-youtube-tracker.git
   cd yolov8-ocsort-youtube-tracker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install OC-SORT** (if not included)
   ```bash
   git clone https://github.com/noahcao/OC_SORT.git
   cp -r OC_SORT/ocsort ./
   ```

### Requirements.txt
```
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
tqdm>=4.62.0
yt-dlp>=2023.1.0
pathlib
argparse
```

## 🎮 Usage

### Basic YouTube Stream Processing
```bash
python main_youtube.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID" --show-live
```

### Advanced Options
```bash
# High quality with video saving
python main_youtube.py \
    --youtube "https://www.youtube.com/watch?v=iztHfFeKeq4" \
    --quality 720p \
    --conf 0.5 \
    --save-video \
    --output results/traffic_monitoring.mp4

# Custom YOLO model with specific settings
python main_youtube.py \
    --youtube "https://www.youtube.com/watch?v=VIDEO_ID" \
    --weights custom_model.pt \
    --conf 0.4 \
    --iou 0.5 \
    --show-live
```

### Local Video File Processing
```bash
python main_youtube.py --video path/to/video.mp4 --show-live
```

## 🎛️ Command Line Arguments

### Input Options
- `--youtube URL` - YouTube video/stream URL
- `--video PATH` - Local video file path

### YOLO Detection Settings  
- `--weights PATH` - Path to YOLO model weights (default: `yolov8n.pt`)
- `--conf FLOAT` - Confidence threshold (default: `0.3`)
- `--iou FLOAT` - IoU threshold for NMS (default: `0.3`)

### Output Options
- `--output PATH` - Output video file path (default: `results/output_stream.mp4`)
- `--save-video` - Save processed video to file
- `--show-live` - Show live processing window (default: `True`)

### Stream Settings
- `--quality CHOICE` - Stream quality: `480p`, `720p`, `1080p`, `best` (default: `720p`)

### OC-SORT Parameters
- `--delta-t INT` - Time window parameter (default: `3`)
- `--inertia FLOAT` - Motion inertia parameter (default: `0.2`)

## 🤖 Using Custom YOLO Models

### 1. Official YOLOv8 Models
```bash
# Nano (fastest)
python main_youtube.py --youtube "URL" --weights yolov8n.pt

# Small  
python main_youtube.py --youtube "URL" --weights yolov8s.pt

# Medium
python main_youtube.py --youtube "URL" --weights yolov8m.pt

# Large
python main_youtube.py --youtube "URL" --weights yolov8l.pt

# Extra Large (most accurate)
python main_youtube.py --youtube "URL" --weights yolov8x.pt
```

### 2. Custom Trained Models
```bash
# Use your own trained model
python main_youtube.py --youtube "URL" --weights path/to/your_custom_model.pt
```

### 3. Specialized Models
```bash
# Person detection only
python main_youtube.py --youtube "URL" --weights yolov8n-person.pt

# Vehicle detection
python main_youtube.py --youtube "URL" --weights yolov8n-vehicle.pt
```

### 4. Training Your Own Model
To train a custom model for specific use cases:

```bash
# Install additional requirements
pip install ultralytics[train]

# Train on custom dataset
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

# Use the trained model
python main_youtube.py --youtube "URL" --weights runs/detect/train/weights/best.pt
```

## 📊 Model Performance Comparison

| Model | Size (MB) | Speed (FPS)* | mAP50-95 | Use Case |
|-------|-----------|-------------|----------|----------|
| YOLOv8n | 6.2 | ~45 | 37.3 | Real-time applications |
| YOLOv8s | 21.5 | ~35 | 44.9 | Balanced speed/accuracy |
| YOLOv8m | 49.7 | ~25 | 50.2 | Higher accuracy needed |
| YOLOv8l | 83.7 | ~20 | 52.9 | Production applications |
| YOLOv8x | 136.7 | ~15 | 53.9 | Maximum accuracy |

*Approximate FPS on RTX 3080, varies by input resolution and system specs.

## 🎯 Example YouTube Streams

Try these public CCTV streams:

```bash
# Traffic monitoring
python main_youtube.py --youtube "https://www.youtube.com/watch?v=iztHfFeKeq4" --weights yolov8n.pt

# Airport surveillance  
python main_youtube.py --youtube "AIRPORT_STREAM_URL" --weights yolov8s.pt --conf 0.4

# City monitoring
python main_youtube.py --youtube "CITY_CAM_URL" --weights yolov8m.pt --quality 1080p
```

## 🔧 Configuration Tips

### For Better Performance
- Use GPU: Install CUDA-enabled PyTorch
- Lower resolution: Use `--quality 480p` for slower connections
- Adjust confidence: Higher `--conf` reduces false positives
- Use smaller models: `yolov8n.pt` for real-time processing

### For Better Accuracy  
- Use larger models: `yolov8l.pt` or `yolov8x.pt`
- Lower confidence: `--conf 0.3` or lower
- Higher resolution: `--quality 1080p`
- Custom trained models for specific objects

## 📈 Output Metrics

The system provides real-time metrics:

- **Model FPS**: Detection + tracking speed
- **Total FPS**: Including visualization overhead  
- **Objects**: Current number of tracked objects
- **Retention**: Average tracking duration per object
- **Confidence**: Detection confidence statistics

## 🐛 Troubleshooting

### Common Issues

**Stream URL extraction fails:**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

**Low FPS performance:**
```bash
# Use smaller model and lower quality
python main_youtube.py --youtube "URL" --weights yolov8n.pt --quality 480p
```

**GPU not detected:**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Connection issues:**
```bash
# Add retry logic and check internet connection
# The script includes auto-reconnect functionality
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [OC-SORT](https://github.com/noahcao/OC_SORT) - Multi-object tracking  
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube stream extraction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

## 📧 Contact

Your Name - [@unclegie](https://x.com/Uncle_Gie) - anggi.andriyadi@gmail.com

Project Link: [https://github.com/yourusername/yolov8-ocsort-youtube-tracker](https://github.com/yourusername/yolov8-ocsort-youtube-tracker)
