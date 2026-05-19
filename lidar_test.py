import cv2
import numpy as np
from rplidar import RPLidar
import math

# --- CONFIGURATION ---
PORT_NAME = '/dev/ttyUSB0'
WINDOW_SIZE = 800
MAX_DISTANCE = 4000  # 4 meters
SCALE = WINDOW_SIZE / (MAX_DISTANCE * 2)
CAM_FOV = 75.0
LABEL_EVERY_N_POINTS = 5  # Increase this to reduce screen clutter

def run_labeled_lidar():
    lidar = RPLidar(PORT_NAME)
    
    try:
        print("Starting... Press 'q' to quit.")
        canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

        for scan in lidar.iter_scans():
            canvas.fill(0) # Clear frame
            
            # Draw a center crosshair for the Robot/Camera position
            cv2.line(canvas, (WINDOW_SIZE//2 - 10, WINDOW_SIZE//2), (WINDOW_SIZE//2 + 10, WINDOW_SIZE//2), (255, 255, 255), 1)
            cv2.line(canvas, (WINDOW_SIZE//2, WINDOW_SIZE//2 - 10), (WINDOW_SIZE//2, WINDOW_SIZE//2 + 10), (255, 255, 255), 1)

            for i, (_, angle, distance) in enumerate(scan):
                if distance <= 0: continue
                
                # 1. Normalize Angle for Coordinate Math
                # (LiDAR 0 deg is usually 'Front')
                angle_rad = math.radians(angle)
                
                # 2. Convert to Cartesian for the Map
                lx = distance * math.cos(angle_rad)
                ly = distance * math.sin(angle_rad)

                # 3. Map to Pixel Coordinates
                px = int(WINDOW_SIZE//2 + (lx * SCALE))
                py = int(WINDOW_SIZE//2 - (ly * SCALE))

                if 0 <= px < WINDOW_SIZE and 0 <= py < WINDOW_SIZE:
                    # Draw the actual point
                    cv2.circle(canvas, (px, py), 3, (0, 255, 0), -1)

                    # 4. Add Text Label (Distance and Angle)
                    # We only label every Nth point to keep it readable
                    # if i % LABEL_EVERY_N_POINTS == 0:
                    #     label = f"{int(distance)}mm, {int(angle)}deg"
                        
                    #     # Draw small text next to the point
                    #     cv2.putText(canvas, label, (px + 5, py - 5), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            cv2.imshow('LiDAR with Text Labels', canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        lidar.stop()
        lidar.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_labeled_lidar()