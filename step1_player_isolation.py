import cv2
import mediapipe as mp
import numpy as np
import os
import requests
from tracker import ByteTracker

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

def analyze_motion(frame, prev_frame):
    """
    Analyzes global motion between frames.
    Returns a motion score (0.0 to 1.0).
    """
    if prev_frame is None:
        return 0.0
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold to remove noise
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of changed pixels
    motion_score = np.sum(thresh > 0) / (frame.shape[0] * frame.shape[1])
    return motion_score

def is_game_scene(frame, green_threshold=0.3):
    """
    Determines if the frame is a game scene based on the ratio of green pixels.
    Returns: (is_game, green_mask, field_hull)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define green range for soccer field
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    
    # Find field hull
    field_hull = None
    if green_ratio > 0.1:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Simplify and get convex hull
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            hull = cv2.convexHull(approx)
            
            # Erode hull slightly to avoid ad boards (Issue #3)
            # We can't erode a polygon directly easily, but we can check distance
            field_hull = hull
            
            # Update mask to be the hull
            hull_mask = np.zeros_like(mask)
            cv2.drawContours(hull_mask, [field_hull], -1, 255, -1)
            mask = hull_mask
    
    return green_ratio > green_threshold, mask, field_hull

def main():
    download_model(MODEL_URL, MODEL_PATH)
    
    # Initialize MediaPipe Object Detector
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        max_results=50,
        score_threshold=0.10,  # Very low threshold for ByteTrack to recover
        running_mode=VisionRunningMode.VIDEO,
        category_allowlist=['person']
    )

    detector = ObjectDetector.create_from_options(options)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Initialize Tracker
    tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    prev_frame = None
    frame_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_index += 1
        
        # 1. Scene Analysis
        is_game, green_mask, field_hull = is_game_scene(frame)
        motion_score = analyze_motion(frame, prev_frame)
        prev_frame = frame.copy()
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_var = np.var(hsv_frame[:, :, 0])
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        
        # Crowd Detection (Issue #5, #8)
        # Crowd = Low H Var AND Low Motion
        # Adjusted threshold: H Var < 1400 is definitely crowd. Motion < 0.1 is static.
        is_crowd = (h_var < 1400) and (motion_score < 0.15)
        
        # Close-up Detection (Issue #6, #7, #9)
        # Close-up = High H Var OR High Motion (camera pan)
        # But we rely mostly on box size for close-ups
        
        # 2. Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        dets_high = []
        dets_low = []
        
        # 3. Filter & Categorize Detections
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            score = detection.categories[0].score
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
            
            # Clamp
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Basic Filters
            if w*h < 2000: continue # Min area
            if h/w < 1.2: continue # Aspect ratio
            
            # Scene Context Filtering
            bottom_center = (x + w//2, y + h)
            height_ratio = h / height
            
            is_valid = False
            is_closeup_box = height_ratio > 0.35
            
            if is_crowd and not is_closeup_box:
                # In crowd scenes, ignore normal sized people (likely crowd)
                # Only accept if very high confidence? No, safer to skip.
                continue
                
            if is_closeup_box:
                # Large box - check if it's likely a player
                # If motion is high or H var is decent, accept
                if motion_score > 0.1 or h_var > 1500:
                    is_valid = True
            elif is_game and field_hull is not None:
                # Check if inside field hull (Issue #3)
                # Strict check: -5 pixels buffer
                dist = cv2.pointPolygonTest(field_hull, bottom_center, True)
                if dist >= -5: # Allow slight margin error, but mostly inside
                    is_valid = True
            elif not is_game:
                # Non-game, non-closeup, non-crowd?
                # Maybe a transition. Be conservative.
                pass
            
            if is_valid:
                # Add to lists for ByteTrack
                # Format: [x, y, w, h]
                det_arr = np.array([x, y, w, h])
                if score > 0.4:
                    dets_high.append(det_arr)
                elif score > 0.1:
                    # In crowd scenes, DO NOT use low confidence detections to recover tracks
                    # This prevents noise in the crowd from being picked up
                    if not is_crowd:
                        dets_low.append(det_arr)
        
        # 4. Update Tracker
        tracks = tracker.update(np.array(dets_high) if dets_high else np.empty((0, 4)),
                                np.array(dets_low) if dets_low else np.empty((0, 4)))
        
        # 5. Draw Tracks
        for track in tracks:
            x, y, w, h, track_id = track
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Expand box (Issue #1 - Visual)
            expand_ratio = 0.15
            x_exp = int(w * expand_ratio / 2)
            y_exp = int(h * expand_ratio / 2)
            x = max(0, x - x_exp)
            y = max(0, y - y_exp)
            w = min(width - x, w + 2*x_exp)
            h = min(height - y, h + 2*y_exp)
            
            # Determine Color/Label
            # User Request: Just feature green boxes to all the players.
            color = (0, 255, 0) # Green
            label = ""
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # Draw feet point
            cv2.circle(frame, (x + w//2, y + h), 5, color, -1)
            
            # Draw feet point
            cv2.circle(frame, (x + w//2, y + h), 5, color, -1)

        # Debug Info
        if frame_index % 30 == 0:
            status = "Game" if is_game else "Non-Game"
            if is_crowd: status += " (Crowd)"
            print(f"Frame {frame_index}: {status}, Motion: {motion_score:.2f}, Tracks: {len(tracks)}")
            
        out.write(frame)

    cap.release()
    out.release()
    print("Processing complete.")

if __name__ == "__main__":
    main()
