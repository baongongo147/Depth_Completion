import cv2
import numpy as np
from rplidar import RPLidar
from picamera2 import Picamera2
import math
import os
import time
import json

# --- 1. DIRECTORY SETUP ---
SAVE_DIR = "lidar_image_pairs"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 2. LOAD CALIBRATION ---
try:
    calib = np.load("picam3_calib.npz")
    K, dist_coeffs = calib['mtx'], calib['dist']
except FileNotFoundError:
    print("Warning: picam3_calib.npz not found. Using identity matrix.")
    K = np.eye(3)
    dist_coeffs = np.zeros(5)

# --- 3. PARAMETERS ---
SCREEN_SIZE = (512, 256)
LIDAR_MIN_DIST = 150.0   # 15cm in mm
LIDAR_MAX_DIST = 12000.0 # 12m in mm
fx = K[0, 0]
cam_fov_h = 2 * math.atan(SCREEN_SIZE[0] / (2 * fx)) * (180 / math.pi)

def nothing(x):
    pass

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": SCREEN_SIZE})
    picam2.configure(config)
    picam2.start()

    lidar = RPLidar('/dev/ttyUSB0')
    capture_count = 0
    
    try:
        print(f"Detected Camera FOV: {cam_fov_h:.2f} degrees")
        print("Controls: [SPACE] to Capture, [Q] to Quit")

        for scan in lidar.iter_scans():

            
            raw_rgb = picam2.capture_array()
            # Convert RGB to BGR for OpenCV display/saving
            frame_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            
            projected_view = frame_bgr.copy()
            radar_view = np.zeros((SCREEN_SIZE[1], SCREEN_SIZE[0], 3), dtype=np.uint8)
            cx, cy = SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2
            cv2.line(radar_view, (cx-5, cy), (cx+5, cy), (50, 50, 50), 1)
            cv2.line(radar_view, (cx, cy-5), (cx, cy+5), (50, 50, 50), 1)
            
            # Prepare current scan data for potential saving
            current_scan_data = []

            for (quality, angle, distance) in scan:
                # Filter based on your 15cm - 12m requirement
                if LIDAR_MIN_DIST <= distance <= LIDAR_MAX_DIST:
                    current_scan_data.append([angle, distance])
                    
                    # --- VIZ LOGIC (Project to Camera) ---
                    angle_rad = math.radians(angle)
                    lx = (distance / 1000.0) * math.cos(angle_rad)
                    ly = (distance / 1000.0) * math.sin(angle_rad)

                    # --- VIZ LOGIC (Radar View) ---
                    rx = int(cx + (ly * 80)) 
                    ry = int(cy - (lx * 80))
                    if 0 <= rx < SCREEN_SIZE[0] and 0 <= ry < SCREEN_SIZE[1]:
                        cv2.circle(radar_view, (rx, ry), 1, (255, 255, 255), -1)

            # Display and Interaction
            combined = np.hstack((projected_view, radar_view))
            cv2.imshow('FOV & Extrinsic Alignment', combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            # --- CAPTURE LOGIC ---
            if key == ord(' '):
                timestamp = int(time.time())
                fn = f"cap_{timestamp}_{capture_count}"
                img_path = os.path.join(SAVE_DIR, f"cap_{timestamp}_{capture_count}.jpg")
                
                # Save the image (original BGR frame)
                cv2.imwrite(img_path, frame_bgr)

                json_metadata = {
                    "timestamp": timestamp,
                    "capture_id": capture_count,
                    "lidar_points": current_scan_data
                }
                
                with open(os.path.join(SAVE_DIR, f"{fn}.json"), 'w') as f:
                    json.dump(json_metadata, f, indent=4) # indent=4 makes it readable
                
                print(f"Saved pair {capture_count}: {img_path}")
                capture_count += 1
                
            elif key == ord('q'):
                break

    finally:
        picam2.stop()
        lidar.stop()
        lidar.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()