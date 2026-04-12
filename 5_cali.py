import cv2
import numpy as np
import json
import glob
import os

# --- CONFIGURATION ---
LABELED_DIR = "lidar_image_pair_labeled"
CALIB_FILE = "picam3_calib.npz"

# 1. Load Camera Intrinsics
calib_data = np.load(CALIB_FILE)
K = calib_data['mtx']
dist_coeffs = calib_data['dist']

all_object_pts = [] # 3D points in LiDAR space (meters)
all_image_pts = []  # 2D points in Camera space (pixels)

# 2. Collect labeled "Center Crease" points from all files
json_files = glob.glob(os.path.join(LABELED_DIR, "*.json"))

print(f"Loading {len(json_files)} labeled pairs...")

for j_path in json_files:
    with open(j_path, 'r') as f:
        data = json.load(f)

    for label in ["Left Edge", "Center Crease", "Right Edge"]:
        l_pt = data["labels"]["lidar_3d_points"][label]
        i_pt = data["labels"]["image_pixel_points"][label]
        
        all_object_pts.append([-l_pt[1], -0.005, l_pt[0]])
        all_image_pts.append([i_pt[0], i_pt[1]])

# Convert to Numpy arrays for OpenCV
obj_pts = np.array(all_object_pts, dtype=np.float32)
img_pts = np.array(all_image_pts, dtype=np.float32)

# 3. Solve PnP
# Use RANSAC if you think some of your manual clicks might be "bad"
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    obj_pts, 
    img_pts, 
    K, 
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

if success:
    # Convert rotation vector to 3x3 Matrix
    R, _ = cv2.Rodrigues(rvec)
    
    print("\n" + "="*30)
    print("CALIBRATION SUCCESSFUL")
    print("="*30)
    
    # Translation Vector (meters)
    # tvec[0] = X (Right/Left), tvec[1] = Y (Up/Down), tvec[2] = Z (Forward/Back)
    print(f"Translation (m): \n{tvec}")
    
    # Rotation Matrix
    print(f"Rotation Matrix: \n{R}")
    
    # Convert R to Euler Angles (Pitch, Roll, Yaw) for humans to read
    # This helps you check if the "tilt" looks correct
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    print(f"\nEstimated Orientation (Degrees):")
    print(f"Pitch: {np.degrees(x):.2f} | Yaw: {np.degrees(y):.2f} | Roll: {np.degrees(z):.2f}")
    
    # Save the Extrinsics
    np.savez("extrinsics.npz", R=R, t=tvec, rvec=rvec)
    print("\nSaved to extrinsics.npz")
else:
    print("PnP Solver failed. Check your point correspondences.")