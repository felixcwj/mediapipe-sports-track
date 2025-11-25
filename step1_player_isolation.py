import cv2
import mediapipe as mp
import numpy as np
import os
import requests

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

def main():
    download_model(MODEL_URL, MODEL_PATH)
    
    # Initialize MediaPipe Object Detector
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        max_results=20,
        score_threshold=0.3,
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_index += 1
        
        # Scene Classification
        is_game, green_mask = is_game_scene(frame)
        
        if not is_game:
            # Label as Non-Play and skip detailed detection
            cv2.putText(frame, "Non-Play / Crowd", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
            print(f"Frame {frame_index}: Non-Play")
            continue

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect Objects
        # Note: detect_for_video requires timestamp in ms
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        # Filter and Draw
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
            
            bottom_center_x = x + w // 2
            bottom_center_y = y + h
            
            # Check a small window around the feet for green
            check_radius = 5
            y_check = min(bottom_center_y, height - 1)
            x_check = min(max(bottom_center_x, 0), width - 1)
            
            # Simple check: is the pixel at feet green?
            # Or better: check a small region
            roi_y1 = max(0, y_check - check_radius)
            roi_y2 = min(height, y_check + check_radius)
            roi_x1 = max(0, x_check - check_radius)
            roi_x2 = min(width, x_check + check_radius)
            
            if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                feet_region = green_mask[roi_y1:roi_y2, roi_x1:roi_x2]
                green_pixel_count = np.sum(feet_region > 0)
                total_pixels = feet_region.size
                
                # If > 10% of pixels around feet are green, consider it a player on field
                if total_pixels > 0 and (green_pixel_count / total_pixels) > 0.1:
                    # Draw Bounding Box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 255, 0), -1)
                else:
                    # Draw Red Box for filtered out person
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        out.write(frame)
        if frame_index % 30 == 0:
            print(f"Processed Frame {frame_index}")

    cap.release()
    out.release()
    print("Processing complete.")

if __name__ == "__main__":
    main()
