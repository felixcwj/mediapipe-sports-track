import cv2
import mediapipe as mp
import numpy as np
import os
import requests
from collections import defaultdict

# Constants
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'
MODEL_PATH = 'efficientdet_lite0.tflite'
INPUT_VIDEO = 'input2.mp4'
OUTPUT_VIDEO = 'output_step1.mp4'

def download_model(url, path):
    if not os.path.exists(path):
        print(f"Downloading model from {url}...")
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded.")
    else:
        print("Model already exists.")

def analyze_background(frame):
    """
    Analyzes the background to detect if it's a player close-up shot.
    Player close-ups typically have:
    - Blurred background (low edge density in background)
    - Uniform/simple background colors (low color variance)
    
    Returns: (is_closeup, background_score)
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate edge density using Laplacian variance (measure of blur)
    laplacian_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
    
    # Calculate color variance in HSV space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_var = np.var(hsv[:, :, 0])
    s_var = np.var(hsv[:, :, 1])
    
    # Close-up shots tend to have:
    # - Lower edge variance (blurred background)
    # - Moderate color variance (not too chaotic like crowd)
    
    # Thresholds (tuned empirically based on test data)
    # Frame 810 (27s close-up): Lap=20.53, H=3597.96
    is_blurred = laplacian_var < 100  # Lower = more blurred (very strict for close-ups)
    is_simple_background = h_var < 5000  # Adjusted for player close-ups with varied colors
    
    # Score for debugging
    background_score = {
        'laplacian_var': laplacian_var,
        'h_var': h_var,
        's_var': s_var
    }
    
    # If background is blurred and relatively simple, likely a close-up
    is_closeup = is_blurred and is_simple_background
    
    return is_closeup, background_score

def is_game_scene(frame, green_threshold=0.3):
    """
    Determines if the frame is a game scene based on the ratio of green pixels.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define green range for soccer field (adjust as needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    
    return green_ratio > green_threshold, mask

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def smooth_box(prev_box, curr_box, alpha=0.7):
    """Smooth box coordinates using exponential moving average"""
    if prev_box is None:
        return curr_box
    
    x = int(alpha * curr_box[0] + (1 - alpha) * prev_box[0])
    y = int(alpha * curr_box[1] + (1 - alpha) * prev_box[1])
    w = int(alpha * curr_box[2] + (1 - alpha) * prev_box[2])
    h = int(alpha * curr_box[3] + (1 - alpha) * prev_box[3])
    
    return (x, y, w, h)

def main():
    download_model(MODEL_URL, MODEL_PATH)
    
    # Initialize MediaPipe Object Detector
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        max_results=30,  # Increased to handle more detections
        score_threshold=0.20,  # Increased to reduce false positives (flags, logos)
        running_mode=VisionRunningMode.VIDEO,
        category_allowlist=['person']
    )

    detector = ObjectDetector.create_from_options(options)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_index = 0
    
    # Tracking state
    prev_boxes = {}  # {track_id: (x, y, w, h)}
    next_track_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_index += 1
        
        # Scene Classification
        is_game, green_mask = is_game_scene(frame)
        
        # Convert to RGB for MediaPipe (needed for both cases)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect Objects
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        # If no green field detected, check if it's a close-up shot
        if not is_game:
            is_closeup, bg_score = analyze_background(frame)
            has_person = len(detection_result.detections) > 0
            person_count = len(detection_result.detections)
            
            # Player close-ups: 1-3 people, blurred background, moderate color variance
            # Crowd scenes: many people OR very uniform background (low H variance)
            # Frame 360 (12s crowd): H=1308, Frame 390 (13s player): H=3425
            is_player_closeup = (
                is_closeup and 
                has_person and 
                person_count <= 3 and  # Close-ups usually show 1-3 players
                bg_score['h_var'] > 1500  # Exclude very uniform scenes (crowd with similar clothing)
            )
            
            # If it's a player close-up, draw detected people
            if is_player_closeup:
                # Draw all detected people as players (assuming close-up is of players)
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    w = int(bbox.width)
                    h = int(bbox.height)
                    
                    # Clamp coordinates
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    bottom_center_x = x + w // 2
                    bottom_center_y = y + h
                    
                    # Draw as player (yellow box for close-up)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 255, 255), -1)
                
                cv2.putText(frame, "Player Close-up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                out.write(frame)
                if frame_index % 30 == 0:
                    print(f"Frame {frame_index}: Close-up (Lap: {bg_score['laplacian_var']:.1f}, H: {bg_score['h_var']:.1f})")
                continue
            else:
                # Label as Non-Play and skip detailed detection
                cv2.putText(frame, "Non-Play / Crowd", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame)
                if frame_index % 30 == 0:
                    print(f"Frame {frame_index}: Non-Play")
                continue

        # Game scene with green field - apply filtering and tracking
        ad_zone_threshold = int(height * 0.89)  # ~960 for 1080p
        
        current_detections = []
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            
            # ROI Check: Check if bottom center of bbox is in green area
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)
            
            # Clamp coordinates
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Filter out very small detections (likely partial detections of legs, etc.)
            box_area = w * h
            if box_area < 3000:  # Minimum area threshold (increased from 2000)
                continue
            
            # Filter by aspect ratio - people are taller than wide
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 1.2:  # People should be at least 1.2x taller than wide
                continue
            
            bottom_center_x = x + w // 2
            bottom_center_y = y + h
            
            # Filter out ad zone - SKIP COMPLETELY, don't even draw red box
            if bottom_center_y >= ad_zone_threshold:
                continue
            
            # Check a small window around the feet for green
            check_radius = 5
            y_check = min(bottom_center_y, height - 1)
            x_check = min(max(bottom_center_x, 0), width - 1)
            
            roi_y1 = max(0, y_check - check_radius)
            roi_y2 = min(height, y_check + check_radius)
            roi_x1 = max(0, x_check - check_radius)
            roi_x2 = min(width, x_check + check_radius)
            
            if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                feet_region = green_mask[roi_y1:roi_y2, roi_x1:roi_x2]
                green_pixel_count = np.sum(feet_region > 0)
                total_pixels = feet_region.size
                
                # Calculate green ratio from is_game_scene result
                green_ratio = np.sum(green_mask > 0) / (height * width)
                
                # Adaptive threshold: if scene has lots of green (>0.5), be more lenient
                # This helps catch running players whose feet might be off ground
                green_threshold = 0.05 if green_ratio > 0.5 else 0.1
                
                # If > threshold of pixels around feet are green, consider it a player on field
                if total_pixels > 0 and (green_pixel_count / total_pixels) > green_threshold:
                    current_detections.append({
                        'box': (x, y, w, h),
                        'center': (bottom_center_x, bottom_center_y),
                        'is_valid': True
                    })
        
        # Simple tracking: match current detections with previous boxes
        new_prev_boxes = {}
        used_track_ids = set()
        
        for det in current_detections:
            curr_box = det['box']
            best_iou = 0
            best_track_id = None
            
            # Find best matching previous box
            for track_id, prev_box in prev_boxes.items():
                if track_id in used_track_ids:
                    continue
                iou = calculate_iou(prev_box, curr_box)
                if iou > best_iou and iou > 0.3:  # Threshold for matching
                    best_iou = iou
                    best_track_id = track_id
            
            # Assign track ID
            if best_track_id is not None:
                track_id = best_track_id
                used_track_ids.add(track_id)
                # Smooth the box
                smoothed_box = smooth_box(prev_boxes[track_id], curr_box, alpha=0.7)
            else:
                track_id = next_track_id
                next_track_id += 1
                smoothed_box = curr_box
            
            new_prev_boxes[track_id] = smoothed_box
            det['track_id'] = track_id
            det['smoothed_box'] = smoothed_box
        
        # Draw tracked detections
        for det in current_detections:
            x, y, w, h = det['smoothed_box']
            
            # Expand box slightly to better cover player (10% expansion)
            expand_ratio = 0.10
            x_expand = int(w * expand_ratio / 2)
            y_expand = int(h * expand_ratio / 2)
            
            x_draw = max(0, x - x_expand)
            y_draw = max(0, y - y_expand)
            w_draw = min(w + 2 * x_expand, width - x_draw)
            h_draw = min(h + 2 * y_expand, height - y_draw)
            
            bottom_center_x, bottom_center_y = det['center']
            
            # Check if person is wearing yellow (referee)
            person_roi = frame[y:y+h, x:x+w]
            is_referee = False
            
            if person_roi.size > 0:
                person_hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                
                # Yellow range for referee uniform
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                yellow_mask = cv2.inRange(person_hsv, lower_yellow, upper_yellow)
                
                yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
                
                if yellow_ratio > 0.15:  # If >15% of person is yellow, likely referee
                    is_referee = True
            
            if is_referee:
                # Draw Orange Box for referee (using expanded box)
                cv2.rectangle(frame, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), (0, 165, 255), 2)
                cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 165, 255), -1)
                cv2.putText(frame, "REF", (x_draw, y_draw-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                # Draw Green Box for player with track ID (using expanded box)
                cv2.rectangle(frame, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), (0, 255, 0), 2)
                cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 255, 0), -1)
                # cv2.putText(frame, f"#{det['track_id']}", (x_draw, y_draw-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        prev_boxes = new_prev_boxes
        
        out.write(frame)
        if frame_index % 30 == 0:
            print(f"Processed Frame {frame_index}")

    cap.release()
    out.release()
    print("Processing complete.")

if __name__ == "__main__":
    main()
