import cv2
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ocsort.ocsort import OCSort
import yt_dlp
import threading
import queue
import subprocess
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 with OC-SORT YouTube Live Stream Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments - support both video file and YouTube URL
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str,
                      help='Path to video file for tracking')
    group.add_argument('--youtube', type=str,
                      help='YouTube URL for live stream tracking')
    
    # YOLO arguments
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                      help='Path to YOLOv8 weights file')
    
    parser.add_argument('--conf', type=float, default=0.3,
                      help='Confidence threshold for detection')
    
    parser.add_argument('--iou', type=float, default=0.3,
                      help='IoU threshold for NMS')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                      default='results/output_stream.mp4',
                      help='Path to save output video (optional for live view)')
    
    parser.add_argument('--save-video', action='store_true',
                      help='Save processed video to file')
    
    parser.add_argument('--show-live', action='store_true', default=True,
                      help='Show live processing window')
    
    # YouTube specific arguments
    parser.add_argument('--quality', type=str, default='720p',
                      choices=['480p', '720p', '1080p', 'best'],
                      help='YouTube stream quality')
    
    # OC-SORT specific parameters
    parser.add_argument('--delta-t', type=int, default=3,
                      help='Time window parameter')
    parser.add_argument('--inertia', type=float, default=0.2,
                      help='Motion inertia parameter')
    
    return parser.parse_args()

def get_youtube_stream_url(youtube_url, quality='720p'):
    """Extract direct stream URL from YouTube using yt-dlp"""
    print(f"🔍 Extracting stream URL from YouTube...")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': f'best[height<={quality[:-1]}]/best',  # Remove 'p' from quality
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            # Get the direct stream URL
            if 'url' in info:
                stream_url = info['url']
                print(f"✅ Stream URL extracted successfully")
                print(f"   Title: {info.get('title', 'Unknown')}")
                print(f"   Duration: {'Live Stream' if info.get('is_live') else info.get('duration', 'Unknown')}")
                return stream_url
            else:
                print("❌ Could not extract stream URL")
                return None
                
    except Exception as e:
        print(f"❌ Error extracting YouTube URL: {str(e)}")
        return None

def process_stream(source, model, tracker, class_names, args):
    """Process video stream (file or YouTube) with OC-SORT tracking"""
    
    # Determine if it's a YouTube URL or local file
    is_youtube = source.startswith('http')
    
    if is_youtube:
        print(f"\n🔍 Processing YouTube stream: {source}")
        # Get actual stream URL
        stream_url = get_youtube_stream_url(source, args.quality)
        if not stream_url:
            print("❌ Failed to get YouTube stream URL")
            return
        cap_source = stream_url
    else:
        print(f"\n🔍 Processing video file: {source}")
        cap_source = source
    
    print("=====================================")
    
    # Open video source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # For live streams, total frames might be 0 or invalid
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or is_youtube:
        total_frames = float('inf')  # Unknown for live streams
        print(f"Live stream: {width}x{height} at {fps} FPS")
    else:
        print(f"Video: {width}x{height} at {fps} FPS, {total_frames} frames")
    
    # Setup output video writer if saving is enabled
    video_writer = None
    if args.save_video:
        output_video_path = Path(args.output)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path), 
            fourcc, 
            fps if fps > 0 else 25.0,  # Default to 25 FPS if unknown
            (width, height)
        )
        print(f"💾 Output video will be saved to: {output_video_path}")
    
    # Track confidence values for metrics
    all_confidences = []
    
    # Variables for processing and metrics
    frame_count = 0
    object_count = 0
    total_model_time = 0
    total_full_time = 0
    model_fps_history = []
    full_fps_history = []
    processing_start_time = time.time()
    
    # Dictionary to track object retention
    track_history = {}
    completed_tracks = []
    
    print("🎬 Starting processing... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_youtube:
                    print("\n⚠️  Stream ended or connection lost")
                    # Try to reconnect for live streams
                    print("🔄 Attempting to reconnect...")
                    cap.release()
                    time.sleep(2)
                    
                    # Re-extract stream URL (it might have changed)
                    new_stream_url = get_youtube_stream_url(source, args.quality)
                    if new_stream_url:
                        cap = cv2.VideoCapture(new_stream_url)
                        if cap.isOpened():
                            print("✅ Reconnected successfully")
                            continue
                    
                    print("❌ Failed to reconnect")
                break
            
            frame_count += 1
            frame_start_time = time.time()
            
            # YOLO detection - measure model time separately
            model_start_time = time.time()
            
            # YOLO detection
            results = model(frame, conf=args.conf)[0]
            
            # Extract bounding boxes, scores, and classes
            detections = []
            for r in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = r
                detections.append([x1, y1, x2, y2, score, class_id])
                all_confidences.append(float(score))
            
            # Convert to torch tensor
            if len(detections) > 0:
                detections_tensor = torch.tensor(detections)
            else:
                detections_tensor = torch.zeros((0, 6))
            
            # Update tracker with detections
            tracks = tracker.update(detections_tensor, None)
            
            # Calculate model time
            model_end_time = time.time()
            model_time = model_end_time - model_start_time
            total_model_time += model_time
            model_fps = 1.0 / model_time if model_time > 0 else 0
            model_fps_history.append(model_fps)
            
            # Count objects in current frame
            current_objects = len(tracks)
            object_count += current_objects
            
            # Update track history for retention calculation
            active_track_ids = set()
            for track in tracks:
                track_id = int(track[4])
                active_track_ids.add(track_id)
                
                if track_id not in track_history:
                    track_history[track_id] = {
                        'total_frames': 1,
                        'continuous_detection': 1,
                        'max_continuous_detection': 1,
                        'last_detected_frame': frame_count
                    }
                else:
                    track = track_history[track_id]
                    
                    if track['last_detected_frame'] == frame_count - 1:
                        track['continuous_detection'] += 1
                    else:
                        track['continuous_detection'] = 1
                    
                    track['max_continuous_detection'] = max(
                        track['max_continuous_detection'],
                        track['continuous_detection']
                    )
                    
                    track['total_frames'] += 1
                    track['last_detected_frame'] = frame_count
            
            # Check for completed tracks
            all_track_ids = list(track_history.keys())
            for track_id in all_track_ids:
                if track_id not in active_track_ids:
                    track_info = track_history[track_id]
                    if track_info['last_detected_frame'] < frame_count - 5:
                        completed_tracks.append(track_info)
                        del track_history[track_id]
            
            # Calculate retention metrics
            avg_retention_frames = 0
            max_retention_frames = 0
            
            all_retention_data = completed_tracks.copy()
            all_retention_data.extend(track_history.values())
            
            if all_retention_data:
                retention_values = [track['total_frames'] for track in all_retention_data]
                avg_retention_frames = sum(retention_values) / len(retention_values)
                max_retention_frames = max(track['max_continuous_detection'] 
                                          for track in all_retention_data)
            
            # Create visualization
            vis_frame = frame.copy()
            
            # Draw tracks
            for track in tracks:
                x1, y1, x2, y2, track_id, cls, conf = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                class_id = int(cls)
                
                # Select color based on track_id
                color = (
                    int(hash(str(track_id)) % 255),
                    int(hash(str(track_id*2)) % 255),
                    int(hash(str(track_id*3)) % 255)
                )
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Get class name
                if class_id in class_names:
                    class_name = class_names[class_id]
                else:
                    class_name = f"Unknown-{class_id}"
                
                # Draw ID and class name
                confidence = round(float(conf) * 100, 1)
                text = f"ID: {track_id} | {class_name} {confidence}%"
                
                # Text background
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # Draw text
                cv2.putText(vis_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Calculate full frame time
            frame_end_time = time.time()
            full_time = frame_end_time - frame_start_time
            total_full_time += full_time
            full_fps = 1.0 / full_time if full_time > 0 else 0
            full_fps_history.append(full_fps)
            
            # Add information overlay
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
            
            # Calculate recent average FPS
            avg_model_fps = sum(model_fps_history[-30:]) / min(30, len(model_fps_history))
            avg_full_fps = sum(full_fps_history[-30:]) / min(30, len(full_fps_history))
            
            # Display metrics
            y_offset = 30
            metrics = [
                f"{'LIVE STREAM' if is_youtube else 'VIDEO FILE'}",
                f"Model FPS: {avg_model_fps:.1f}",
                f"Total FPS: {avg_full_fps:.1f}",
                f"Objects: {current_objects}",
                f"Frame: {frame_count}",
                f"Model time: {model_time*1000:.1f} ms",
                f"Total time: {full_time*1000:.1f} ms",
                f"Avg Retention: {avg_retention_frames:.1f} frames",
                f"Max Continuous: {max_retention_frames} frames"
            ]
            
            for i, metric in enumerate(metrics):
                color = (0, 255, 255) if i == 0 and is_youtube else (0, 255, 0)  # Yellow for LIVE STREAM
                cv2.putText(vis_frame, metric, (20, y_offset + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save frame if video writer is enabled
            if video_writer:
                video_writer.write(vis_frame)
            
            # Show live preview if enabled
            if args.show_live:
                cv2.imshow('YOLOv8 + OC-SORT Live Tracking', vis_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopping by user request...")
                    break
            
            # Print progress every 50 frames
            if frame_count % 50 == 0:
                elapsed_time = time.time() - processing_start_time
                
                print(f"Frame: {frame_count} | " 
                      f"Model FPS: {avg_model_fps:.1f} | "
                      f"Total FPS: {avg_full_fps:.1f} | "
                      f"Objects: {current_objects} | "
                      f"Avg Retention: {avg_retention_frames:.1f} | "
                      f"Max Continuous: {max_retention_frames} | "
                      f"Runtime: {elapsed_time:.0f}s")
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if args.show_live:
            cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - processing_start_time
        if frame_count > 0:
            print(f"\n✅ Processing completed!")
            print(f"  • Processed {frame_count} frames in {total_time:.2f} seconds")
            print(f"  • Average Model FPS: {frame_count/total_model_time:.1f}")
            print(f"  • Average Total FPS: {frame_count/total_full_time:.1f}")
            print(f"  • Average objects per frame: {object_count/frame_count:.1f}")
            
            if args.save_video and video_writer:
                print(f"  • Video saved to: {args.output}")

def main():
    # Parse arguments
    args = parse_args()
    
    try:
        # Initialize YOLOv8 model
        print(f"\n🚀 Loading YOLOv8 model from {args.weights}...")
        model = YOLO(args.weights)
        class_names = model.names
        
        # Initialize OC-SORT tracker
        print("Initializing OC-SORT tracker...")
        tracker = OCSort(
            det_thresh=args.conf,
            iou_threshold=args.iou,
            delta_t=args.delta_t,
            inertia=args.inertia
        )
        
        # Determine source
        source = args.youtube if args.youtube else args.video
        
        # Process stream
        process_stream(source, model, tracker, class_names, args)
    
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✨ YOLOv8 + OC-SORT finished!")

if __name__ == "__main__":
    main()