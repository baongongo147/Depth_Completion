import cv2
import numpy as np
from rplidar import RPLidar
from picamera2 import Picamera2
import math
import os
import time
import json

import common


# --- 2. LOAD CALIBRATION ---
try:
    calib = np.load("picam3_calib.npz")
    extrin = np.load("extrinsics.npz")
    K, dist_coeffs = calib['mtx'], calib['dist']
    tvec, rvec = extrin['t'], extrin['r']
except FileNotFoundError:
    print("Warning: picam3_calib.npz not found.")

fx = K[0, 0]
cam_fov_h = 2 * math.atan(common.CAMERA_BUFFER_SIZE[0] / (2 * fx)) * (180 / math.pi)

def nothing(x):
    pass


def main():
    global rvec
    global tvec

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": common.CAMERA_BUFFER_SIZE})
    picam2.configure(config)
    picam2.start()

    lidar = RPLidar('/dev/ttyUSB0')
    capture_count = 0
    
    try:
        print(f"Detected Camera FOV: {cam_fov_h:.2f} degrees")
        print("Lidar status" + lidar.get_health()[0])
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
                    current_scan_data.append([lx, 0.0, lz, distance])

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
            cv2.imshow('Lidar', radar_view)

            for p in current_scan_data:
                position = [p[0], p[1], p[2]]
                if position[2] < 0.0:
                    continue
                distance = p[3]
                lidar_pos = np.array(position, dtype=np.float32)
                R_mat, _ = cv2.Rodrigues(rvec)
                lidar_cam_coords = (R_mat @ lidar_pos) - np.array([0, 0, 0], dtype=np.float32)

                


                # rvec = np.array([0, 0, 0], dtype=np.float32)
                # tvec = np.array([0, 0, 0], dtype=np.float32)
                img_pts, jacobian = cv2.projectPoints(lidar_pos, rvec, tvec, K, dist_coeffs)  
                img_pts = img_pts.reshape(-1, 2)

                for pt in img_pts:
                    if not np.all(np.isfinite(pt)):
                        continue
                    # Extract x and y
                    x, y = pt
                    
                    # 3. Cast to int for cv2.circle
                    center = (int(x), int(y))
                    
                    # Optional: Only draw if the point is within the image boundaries
                    height, width = raw_rgb.shape[:2]
                    if 0 <= center[0] < width and 0 <= center[1] < height:
                        min_dist = 150 #mm
                        max_dist = 12000 #mm
                        dist_clipped = np.clip(distance, min_dist, max_dist)
    
                        # 2. Normalize distance to a 0.0 - 1.0 range
                        # t = 0 is Red (min), t = 1 is Blue (max)
                        t = (dist_clipped - min_dist) / (max_dist - min_dist)
                        r = int((1 - t) * 255 + t * 0)
                        b = int((1 - t) * 0 + t * 255)
                        cv2.circle(raw_rgb, center, 1, (r, 0, b), -1)
            
            cv2.imshow('Camera', raw_rgb)

            key = cv2.waitKey(1) & 0xFF  
            if key == ord('q'):
                break

    finally:
        picam2.stop()
        lidar.stop()
        lidar.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()