import cv2
import numpy as np
from rplidar import RPLidar
import pyrealsense2 as rs
import math
import os
import time
import json
import sys
import common

# --- 1. DIRECTORY SETUP ---
SAVE_DIR = "lidar_image_pairs"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 2. LOAD CALIBRATION ---
try:
    calib = np.load("realsense_calib.npz")
    K, dist_coeffs = calib['mtx'], calib['dist']
except FileNotFoundError:
    print("Warning: realsense_calib.npz not found. Using identity matrix.")
    K = np.eye(3)
    dist_coeffs = np.zeros(5)

# --- 3. PARAMETERS ---
fx = K[0, 0]
cam_fov_h = 2 * math.atan(common.CAMERA_BUFFER_SIZE[0] / (2 * fx)) * (180 / math.pi)

def nothing(x):
    pass

def connect_lidar():
    print("\n(Re)connecting to Lidar...")
    lidar = RPLidar('COM3', baudrate=115200, timeout=3)
    time.sleep(2)

    lidar._serial_port.reset_input_buffer()
    lidar._serial_port.reset_output_buffer()

    try: lidar.stop()
    except: pass
    time.sleep(1)

    try: lidar.reset()
    except: pass
    time.sleep(2)

    lidar._serial_port.reset_input_buffer()
    lidar._serial_port.reset_output_buffer()

    print("Lidar health:", lidar.get_health())
    lidar.start_motor()
    time.sleep(1)

    print("Lidar ready!")
    return lidar

def main():
    # REALSENSE INIT (thay Picamera2)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        common.CAMERA_BUFFER_SIZE[0],
        common.CAMERA_BUFFER_SIZE[1],
        rs.format.bgr8,
        30
    )
    pipeline.start(config)
    
    # --- THÊM VÀO ĐÂY ---
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Lidar', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Camera', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    # Duy trì tỉ lệ 16:9 cho cửa sổ camera (ví dụ 1280x720)
    # cv2.resizeWindow('Camera', 1280, 720) 
    # --------------------

    # RPLIDAR (Windows)    
    print(f"Detected Camera FOV: {cam_fov_h:.2f} degrees")
    print("Controls: [SPACE] to Capture, [Q] to Quit")
    run_flag = True
    lidar = connect_lidar()   
    capture_count = 0

    try:
        while True:
            try:
                scan_generator = lidar.iter_scans(max_buf_meas=5000)
                for scan in scan_generator:                    
                    # ===== LẤY FRAME REALSENSE =====
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    raw_rgb = np.asanyarray(color_frame.get_data())
                    
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
                                #cv2.putText(radar_view, text, (u, v), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255))
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
                        print(f"Saved 1920x1080 image: {img_path}")

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
                        run_flag = False
                        break
                if not run_flag: # Kiểm tra để thoát tiếp vòng while
                    break
            except Exception as e:
                if not run_flag: break
                print("\n🔥 LIDAR CRASH:", e)
                print("Reconnecting in 2 seconds...")

                try:
                    lidar.stop()
                    lidar.disconnect()
                except:
                    pass

                time.sleep(2)
                lidar = connect_lidar()
    finally:
        pipeline.stop()
        lidar.stop()
        lidar.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()