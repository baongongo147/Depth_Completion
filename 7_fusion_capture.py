import cv2
import numpy as np
from rplidar import RPLidar
import pyrealsense2 as rs
import math
import os
import time
import json

import common

SAVE_ROOT = "dataset"
RGB_DIR = os.path.join(SAVE_ROOT, "rgb")
RGB_FUSION_DIR = os.path.join(SAVE_ROOT, "rgb_fusion")
DEPTH_DIR = os.path.join(SAVE_ROOT, "depth")
LIDAR_DIR = os.path.join(SAVE_ROOT, "lidar")

for d in [SAVE_ROOT, RGB_DIR, RGB_FUSION_DIR, DEPTH_DIR, LIDAR_DIR]:
    os.makedirs(d, exist_ok=True)

START_INDEX = 1   
MIN_DEPTH = 150      # mm
MAX_DEPTH = 12000    # mm

def depth_to_color(depth_mm):
    depth_clipped = np.clip(depth_mm, MIN_DEPTH, MAX_DEPTH)

    # Normalize giống hệt depth image
    depth_8bit = np.uint8(255 * (depth_clipped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH))

    color = cv2.applyColorMap(depth_8bit.reshape(1,1), cv2.COLORMAP_JET)[0][0]
    return int(color[0]), int(color[1]), int(color[2])  # BGR

# --- 2. LOAD CALIBRATION ---
try:
    calib = np.load("realsense_calib.npz")
    extrin = np.load("RL_extrinsics.npz")
    K, dist_coeffs = calib['mtx'], calib['dist']
    tvec, rvec = extrin['t'], extrin['r']
except FileNotFoundError:
    print("Warning: realsense_calib.npz not found.")

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

    try:
        lidar.stop()
    except:
        pass
    time.sleep(1)

    try:
        lidar.reset()
    except:
        pass
    time.sleep(2)

    lidar._serial_port.reset_input_buffer()
    lidar._serial_port.reset_output_buffer()

    print("Lidar health:", lidar.get_health())
    lidar.start_motor()
    time.sleep(1)

    print("Lidar ready!")
    return lidar

def get_next_index():
    files = os.listdir(RGB_DIR)
    indices = []

    for f in files:
        if f.endswith("_rgb.png"):
            try:
                idx = int(f.split("_")[0])
                indices.append(idx)
            except:
                pass

    if len(indices) == 0:
        return START_INDEX
    else:
        return max(indices) + 1
    
def main():
    capture_count = get_next_index()
    print(f"Dataset will start saving from index: {capture_count}")

    global rvec
    global tvec

    # ===== REALSENSE INIT (RGB + DEPTH ALIGNED) =====
    pipeline = rs.pipeline()
    config = rs.config()

    # RGB stream
    config.enable_stream(
        rs.stream.color,
        common.CAMERA_BUFFER_SIZE[0],
        common.CAMERA_BUFFER_SIZE[1],
        rs.format.bgr8,
        30
    )

    # Depth stream (16bit mm)
    config.enable_stream(
        rs.stream.depth,
        common.CAMERA_BUFFER_SIZE[0],
        common.CAMERA_BUFFER_SIZE[1],
        rs.format.z16,
        30
    )

    pipeline.start(config)

    # ⭐ CỰC KỲ QUAN TRỌNG: align depth -> color
    align = rs.align(rs.stream.color)
    

    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Lidar', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Camera', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('Camera', 1280, 720)
    
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Depth', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('Depth', 1280, 720)

    print(f"Detected Camera FOV: {cam_fov_h:.2f} degrees")
    lidar = connect_lidar()
    run_flag = True    
    try:
        while True:
            try:
                scan_generator = lidar.iter_scans(max_buf_meas=5000)
                for scan in scan_generator:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    raw_rgb = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    rgb_frame = raw_rgb.copy()       # ảnh gốc
                    fusion_frame = raw_rgb.copy()    # ảnh để vẽ lidar
                    
                    # depth_colormap = cv2.applyColorMap(
                    #     cv2.convertScaleAbs(depth_image, alpha=0.03),
                    #     cv2.COLORMAP_JET
                    # )
                    depth_clipped = np.clip(depth_image, MIN_DEPTH, MAX_DEPTH)
                    depth_8bit = ((depth_clipped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255).astype(np.uint8)
                    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                    cv2.imshow("Depth", depth_colormap)

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
                    visible_pixel_points = []
                    visible_lidar_xyz = []
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
                                # # ===== ADAPTIVE DEPTH CONSISTENCY CHECK =====
                                # depth_cam = depth_image[center[1], center[0]]  # mm
                                # depth_lidar = distance                         # mm

                                # if depth_cam == 0:
                                #     continue

                                # # Sai số RealSense tăng theo khoảng cách
                                # # Ở gần: ~10cm
                                # # Ở xa: cho phép ~8-10% khoảng cách

                                # BASE_ERROR = 120        # mm (cho gần)
                                # ERROR_RATIO = 0.08      # 8% khoảng cách

                                # adaptive_threshold = BASE_ERROR + ERROR_RATIO * depth_lidar
                                # # Chỉ giữ điểm nếu lidar và camera thấy cùng vật thể

                                # if abs(depth_cam - depth_lidar) > adaptive_threshold:
                                #     continue

                                # ========================================
                                min_dist = 150 #mm
                                max_dist = 12000 #mm
                                dist_clipped = np.clip(distance, min_dist, max_dist)
            
                                # 2. Normalize distance to a 0.0 - 1.0 range
                                # t = 0 is Red (min), t = 1 is Blue (max)
                                # t = (dist_clipped - min_dist) / (max_dist - min_dist)
                                # r = int((1 - t) * 255 + t * 0)
                                # b = int((1 - t) * 0 + t * 255)
                                # cv2.circle(raw_rgb, center, 3, (r, 0, b), -1) #cv2.circle(image, center, radius, color, thickness). De Raidus 1 qua nho nen tang thanh 3
                                color = depth_to_color(distance)
                                # Lưu pixel trên ảnh
                                visible_pixel_points.append([int(center[0]), int(center[1])])

                                # Lưu toạ độ lidar 3D (mm)
                                visible_lidar_xyz.append([
                                    float(position[0]),
                                    float(position[1]),
                                    float(position[2])
                                ])
                                cv2.circle(fusion_frame, center, 3, color, -1)
                    
                    cv2.imshow('Camera', fusion_frame)


                    key = cv2.waitKey(1) & 0xFF  
                    # ================= CAPTURE DATASET =================
                    if key == ord('s'):
                        idx = capture_count
                        # Save RGB gốc (không lidar)
                        rgb_path = os.path.join(RGB_DIR, f"{idx}_rgb.png")
                        cv2.imwrite(rgb_path, rgb_frame)

                        # RGB fusion
                        rgb_fusion_path = os.path.join(RGB_FUSION_DIR, f"{idx}_rgb_fusion.png")
                        cv2.imwrite(rgb_fusion_path, fusion_frame)

                        # Depth 16bit (ground truth)
                        depth_path_16 = os.path.join(DEPTH_DIR, f"{idx}_depth_16bit.png")
                        cv2.imwrite(depth_path_16, depth_image)

                        # Depth preview (để xem)
                        depth_preview_path = os.path.join(DEPTH_DIR, f"{idx}_depth.png")
                        cv2.imwrite(depth_preview_path, depth_colormap)

                        # Save LIDAR POINTS (chỉ điểm visible)
                        lidar_json = {
                            "labels": {
                                "image_pixel_points": visible_pixel_points,
                                "lidar_3d_points_milimeters": visible_lidar_xyz
                            }
                        }
                        lidar_path = os.path.join(LIDAR_DIR, f"{idx}_lidar.json")
                        with open(lidar_path, 'w') as f:
                            json.dump(lidar_json, f, indent=4)

                        print(f"\n📸 Saved sample #{idx}")
                        capture_count += 1
                    if key == ord('q'):
                        run_flag = False
                        break
                if not run_flag:
                    break
            except Exception as e:
                if not run_flag:
                    break
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