import cv2
import numpy as np
import glob
import os
import math

# --- SETTINGS ---
CHESSBOARD_SIZE = (14, 14)
SQUARE_SIZE = 35.0  # mm
INPUT_DIR = "cali"
OUTPUT_FILE = "picam3_calib.npz"

def calculate_fov(mtx, width, height):
    fx, fy = mtx[0, 0], mtx[1, 1]
    fov_h = 2 * math.atan(width / (2 * fx)) * (180 / math.pi)
    fov_v = 2 * math.atan(height / (2 * fy)) * (180 / math.pi)
    return fov_h, fov_v

# Prepare object points (X, Y, 0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpoints, filenames = [], [], []
images = glob.glob(f"{INPUT_DIR}/*.jpg")

if not images:
    print(f"Error: No images found in {INPUT_DIR}!")
    exit()

print(f"Processing {len(images)} images...")
img_size = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None: img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        filenames.append(os.path.basename(fname))
        print(f" [+] Success: {filenames[-1]}")

# --- 1. RUN INTRINSIC CALIBRATION ---
if len(objpoints) > 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    # --- 2. EXTRACT HOMOGRAPHIES ---
    # We store a homography for every image so you can pick the best 'floor' reference
    homographies = {}
    
    for i in range(len(imgpoints)):
        # Homography only needs X,Y from objp (Z is 0)
        src_pts = objpoints[i][:, :2] 
        dst_pts = imgpoints[i]
        
        # Calculate H (maps mm on board -> pixels in image)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        homographies[filenames[i]] = H
    primary_H = homographies[filenames[0]]
    # Save all data
    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist, homography=primary_H, **homographies)
    
    h_fov, v_fov = calculate_fov(mtx, img_size[0], img_size[1])
    print("-" * 30)
    print(f"CALIBRATION COMPLETE (RMS: {ret:.4f})")
    print(f"FOV: {h_fov:.2f}H x {v_fov:.2f}V")
    print(f"Saved {len(homographies)} Homographies to {OUTPUT_FILE}")
else:
    print("Error: Need at least 10 valid images.")