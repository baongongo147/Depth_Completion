import cv2
import numpy as np
import glob
import os
import math
import common

INPUT_DIR = "cali"
OUTPUT_FILE = "picam3_calib.npz"


# Prepare object points (X, Y, 0)
objp = np.zeros((common.CHECKERBOARD_SIZE[0] * common.CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:common.CHECKERBOARD_SIZE[0], 0:common.CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= common.CHECKERBOARD_SQUARE_SIZE

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

    ret, corners = cv2.findChessboardCorners(gray, common.CHECKERBOARD_SIZE, None)
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
    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)
    
    h_fov, v_fov = common.calculate_fov(mtx)
    print("-" * 30)
    print(f"CALIBRATION COMPLETE (RMS: {ret:.4f})")
    print(f"FOV: {h_fov:.2f}H x {v_fov:.2f}V")
else:
    print("Error: Need at least 10 valid images.")