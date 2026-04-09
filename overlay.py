import cv2
import numpy as np
from rplidar import RPLidar
from picamera2 import Picamera2
import math

# --- 1. LOAD CALIBRATION ---
calib = np.load("picam3_calib.npz")
K, dist_coeffs = calib['mtx'], calib['dist']

# --- 2. EXTRINSIC PARAMETERS ---  
SCREEN_SIZE = (512, 256)
max_dist = 3000.0

# Calculate Camera FOV from K Matrix
fx = K[0, 0]
# Horizontal FOV in degrees
cam_fov_h = 2 * math.atan(SCREEN_SIZE[0] / (2 * fx)) * (180 / math.pi)
cam_half_fov_h = int(cam_fov_h / 2.0)

def get_depth_color(distance):
    norm = np.clip(distance, 0, max_dist) / max_dist
    val = np.array([[int(norm * 255)]], dtype=np.uint8)
    return tuple(map(int, cv2.applyColorMap(val, cv2.COLOR_RGB2BGR)[0][0]))

def nothing(x):
    pass

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": SCREEN_SIZE})
    picam2.configure(config)
    picam2.start()

    lidar = RPLidar('/dev/ttyUSB0')
    
    try:
        print(f"Detected Camera FOV: {cam_fov_h:.2f} degrees")

        # --- 2. SETUP TRACKBARS ---
        cv2.namedWindow('Control Panel')
        #cv2.resizeWindow('Control Panel', 400, 300)
        cv2.createTrackbar('Pitch (deg)', 'Control Panel', 0, 180, nothing)    # 90 is 0 deg
        cv2.createTrackbar('Yaw (deg)', 'Control Panel', 0, 360, nothing)     # 180 is 0 deg
        cv2.createTrackbar('Height (cm)', 'Control Panel', 10, 100, nothing)    # 0 to 1 meter
        
        for scan in lidar.iter_scans():
            p_val = cv2.getTrackbarPos('Pitch (deg)', 'Control Panel')
            y_val = cv2.getTrackbarPos('Yaw (deg)', 'Control Panel')
            h_val = cv2.getTrackbarPos('Height (cm)', 'Control Panel')
            
            raw_rgb = picam2.capture_array()
            projected_view = raw_rgb
            radar_view = np.zeros((SCREEN_SIZE[1], SCREEN_SIZE[0], 3), dtype=np.uint8)
            
            # --- 1. DRAW CAMERA FOV CONE ON RADAR ---
            center_x, center_y = SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2
            fov_rad = math.radians(cam_fov_h / 2)
            
            # End points for FOV lines (drawn to the edge of the radar view)
            line_len = SCREEN_SIZE[1] // 2
            # Right Boundary
            p1 = (int(center_x + line_len * math.sin(fov_rad)), int(center_y - line_len * math.cos(fov_rad)))
            p2 = (int(center_x - line_len * math.sin(fov_rad)), int(center_y - line_len * math.cos(fov_rad)))
            
            # Draw the "Vision Cone"
            cv2.line(radar_view, (center_x, center_y), p1, (100, 100, 100), 1)
            cv2.line(radar_view, (center_x, center_y), p2, (100, 100, 100), 1)
            # Add a faint "V" fill
            poly_pts = np.array([(center_x, center_y), p1, p2])
            cv2.fillPoly(radar_view, [poly_pts], (20, 20, 20))

            # --- 2. DRAW YAW ALIGNMENT LINE (TRANSPARENT) ---
            overlay = projected_view.copy()
            cv2.line(overlay, (SCREEN_SIZE[0]//2, 0), (SCREEN_SIZE[0]//2, SCREEN_SIZE[1]), (0, 255, 255), 1) 
            cv2.addWeighted(overlay, 0.3, projected_view, 0.7, 0, projected_view)

            # Rotation Matrix for Pitch
            p_rad = math.radians(p_val)
            R_pitch = np.array([
                [1, 0, 0],
                [0, math.cos(p_rad), -math.sin(p_rad)],
                [0, math.sin(p_rad),  math.cos(p_rad)]
            ])

            for (i, angle, distance) in scan:
                if distance <= 0: continue 
                # Clamp the angles
   
                color = get_depth_color(distance)
                adjusted_angle = angle + y_val
                angle_rad = math.radians(adjusted_angle)
                
                # Polar -> Cartesian
                lx = (distance / 1000.0) * math.cos(angle_rad) 
                ly = (distance / 1000.0) * math.sin(angle_rad) 
                
                # Transform to Camera Space
                p_lidar = np.array([lx, 0, ly])
                p_cam = p_lidar

                # Camera matrix
                tvec = np.array([0, h_val, 0], dtype=np.float32)
                rvec = np.array([0.0, 0.0, 0.0], dtype=np.float32) 

                # Project to Camera View
                img_pt, _ = cv2.projectPoints(p_cam, rvec, tvec, K, dist_coeffs)
                u, v = img_pt[0].ravel()
                if 0 <= u < SCREEN_SIZE[0] and 0 <= v < SCREEN_SIZE[1]:
                    cv2.circle(projected_view, (int(u), int(v)), 2, color, -1)

                # Draw to Radar View
                # Note: radar_view coordinates are rotated so 'forward' is UP
                rx = int(SCREEN_SIZE[0]//2 + (ly * (SCREEN_SIZE[1] / (max_dist/1000 * 2))))
                ry = int(SCREEN_SIZE[1]//2 - (lx * (SCREEN_SIZE[1] / (max_dist/1000 * 2))))
                if 0 <= rx < SCREEN_SIZE[0] and 0 <= ry < SCREEN_SIZE[1]:
                    cv2.circle(radar_view, (rx, ry), 1, color, -1)

            # Display Status
            cv2.putText(projected_view, f"P: {p_val} Y: {y_val} FOV: {cam_fov_h:.1f}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            combined = np.hstack((projected_view, radar_view))
            cv2.imshow('FOV & Extrinsic Alignment', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

    finally:
        picam2.stop()
        lidar.stop()
        lidar.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()