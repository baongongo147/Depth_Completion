import cv2
import numpy as np
from rplidar import RPLidar
from picamera2 import Picamera2
import math
import os
import time
import json

import common

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
fx = K[0, 0]
cam_fov_h = 2 * math.atan(common.CAMERA_BUFFER_SIZE[0] / (2 * fx)) * (180 / math.pi)

def nothing(x):
    pass


def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": common.CAMERA_BUFFER_SIZE})
    picam2.configure(config)
    picam2.start()

    lidar = RPLidar('/dev/ttyUSB0')
    capture_count = 0
    
    try:
        print(f"Detected Camera FOV: {cam_fov_h:.2f} degrees")
        print("Controls: [SPACE] to Capture, [Q] to Quit")

        for scan in lidar.iter_scans():
            raw_rgb = picam2.capture_array()

            radar_view = np.zeros((common.LIDAR_VIEW_SIZE[0], common.LIDAR_VIEW_SIZE[1], 3), dtype=np.uint8)
            radar_ruler_color = (50, 50, 50)
            cx, cy = common.LIDAR_VIEW_SIZE[0] // 2, common.LIDAR_VIEW_SIZE[1] // 2
            cv2.line(radar_view, (0, cy), (common.LIDAR_VIEW_SIZE[0], cy), radar_ruler_color, 1)
            cv2.line(radar_view, (cx, 0), (cx, common.LIDAR_VIEW_SIZE[1]), radar_ruler_color, 1)
            cv2.putText(radar_view, "-z", (0, cy), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)
            cv2.putText(radar_view, "+z", (common.LIDAR_VIEW_SIZE[0] - 40, cy), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)

            cv2.putText(radar_view, "+x", (cx, 40), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)
            cv2.putText(radar_view, "-x", (cx, common.LIDAR_VIEW_SIZE[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)
            
            # Prepare current scan data for potential saving
            current_scan_data = []

            for (quality, angle, distance) in scan:
                    lx, lz = common.angleDistanceToLidarXZ(angle, distance)
                    current_scan_data.append([lx, 0.0, lz])

                    #Convert to meters for easier to look at
                    lx_m = lx / 1000.0
                    lz_m = lz / 1000.0
                    u = int(cx + (lz_m * common.LIDAR_VIEW_SCALE)) 
                    v = int(cy - (lx_m * common.LIDAR_VIEW_SCALE))
                    if 0 <= u < common.LIDAR_VIEW_SIZE[0] and 0 <= v < common.LIDAR_VIEW_SIZE[1]:
                        text = f"{lx_m:.2f} | {lz_m:.2f}"
                        cv2.putText(radar_view, text, (u, v), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255))
                        cv2.circle(radar_view, (u, v), 1, (255, 255, 255), -1)

            # Display and Interaction
            #combined = np.hstack((raw_rgb, radar_view))
            cv2.imshow('Camera', raw_rgb)
            cv2.imshow('Lidar', radar_view)
            
            key = cv2.waitKey(1) & 0xFF
            
            # --- CAPTURE LOGIC ---
            if key == ord(' '):
                timestamp = int(time.time())
                fn = f"cap_{timestamp}_{capture_count}"
                img_path = os.path.join(SAVE_DIR, f"cap_{timestamp}_{capture_count}.jpg")
                
                # Save the image (original BGR frame)
                cv2.imwrite(img_path, raw_rgb)

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