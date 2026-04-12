import cv2
import numpy as np
from picamera2 import Picamera2
import os
import time

import common

# --- SETTINGS ---
CHESSBOARD_SIZE = (14, 14) 
SAVE_DIR = "cali"
CAPTURE_INTERVAL = 1.0  # Seconds between auto-captures

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": common.CAMERA_BUFFER_SIZE})
picam2.configure(config)
picam2.set_controls({"AfMode": 0, "LensPosition": 1.0}) 
picam2.start()

# --- STATE VARIABLES ---
auto_mode = False
last_capture_time = 0
count = 0

print("--- CALIBRATION AUTO-CAPTURE ---")
print("SPACE : Toggle Auto-Capture (1 image/sec)")
print("ESC   : Quit")

try:
    while True:
        frame = picam2.capture_array()
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # Check for corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        display = bgr_frame.copy()
        current_time = time.time()

        # --- AUTO-CAPTURE LOGIC ---
        if auto_mode:
            # Visual Feedback: Red 'REC' circle
            cv2.circle(display, (20, 20), 8, (0, 0, 255), -1)
            cv2.putText(display, "AUTO-REC", (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Check if 1 second has passed AND corners are visible
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                if ret:
                    count += 1
                    filename = f"{SAVE_DIR}/auto_{int(current_time)}_{count}.jpg"
                    cv2.imwrite(filename, bgr_frame)
                    last_capture_time = current_time
                    print(f"Auto-Saved: {filename}")
                else:
                    # Optional: Print warning if auto-mode is on but board is missing
                    cv2.putText(display, "NO BOARD FOUND", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw corners for feedback
        if ret:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, ret)
        
        cv2.imshow("Calibration Capture", display)
        
        key = cv2.waitKey(1)
        if key == ord(' '):
            auto_mode = not auto_mode
            print(f"Auto-Capture: {'ENABLED' if auto_mode else 'DISABLED'}")
            # Reset timer so it triggers immediately when enabled
            last_capture_time = 0 
                
        elif key == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"\nSession finished. Total images: {count}")