import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbd_dir", type=Path, default="Test_Datasets/Private_Test")
    parser.add_argument("--onnx_path", type=str, default="checkpoints/models/g2_monodepth_epoch_30.onnx")
    return parser.parse_args()

# --- KÍCH THƯỜC ẢNH THẬT VÀ PADDING ---
# Ảnh Dataset gốc: 848x480 (WxH) -> Padding lên bội số 64 gần nhất: 896x512
# ORIG_H, ORIG_W = 480, 848
# PAD_BOTTOM = 32   # 512 - 480 = 32
# PAD_RIGHT  = 48   # 896 - 848 = 48

# Ảnh Target camera gốc: 640x480 (WxH) -> Padding lên bội số 64 gần nhất: 640x512
# Ảnh Dataset gốc: 848x480 (WxH) -> Resize về Target camera 640x480 rồi padding lên 640x512
ORIG_H, ORIG_W = 480, 640
PAD_BOTTOM = 32   # 512 - 480 = 32
PAD_RIGHT  = 0    # 640 - 640 = 0

# --- REPLACEMENT FOR RGBPReader (NUMPY ONLY) ---
def prepare_input(rgb_path, raw_path):
    # Đọc ảnh RGB bằng OpenCV (Nhanh hơn PIL)
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (ORIG_W, ORIG_H), interpolation=cv2.INTER_LINEAR)
    
    # Padding viền đen theo kích thước ONNX model
    rgb = cv2.copyMakeBorder(rgb, top=0, bottom=PAD_BOTTOM, left=0, right=PAD_RIGHT,
                             borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
    rgb = rgb.astype(np.float32) / 255.0
    # [H, W, C] -> [1, C, H, W]
    rgb = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

    # Đọc ảnh Raw (Depth)
    raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    raw = cv2.resize(raw, (ORIG_W, ORIG_H), interpolation=cv2.INTER_NEAREST)
    # Padding viền đen cho ảnh Depth (giá trị 0 = không có dữ liệu LiDAR)
    raw = cv2.copyMakeBorder(raw, top=0, bottom=PAD_BOTTOM, left=0, right=PAD_RIGHT,
                             borderType=cv2.BORDER_CONSTANT, value=0)
        
    raw = (raw.astype(np.float32) / 65535.0)
    # [H, W] -> [1, 1, H, W]
    raw = raw[np.newaxis, np.newaxis, ...]

    # Tạo mask hole (nơi depth = 0)
    hole = np.where(raw == 0, 0, 1).astype(np.float32)
    
    return rgb, raw, hole

def adjust_domain_numpy(pred):
    # Giả lập hàm adjust_domain của dự án (thường là kẹp giá trị và nhân 65535)
    pred = np.clip(pred, 0, 1)
    pred = (pred * 65535.0).astype(np.uint16)
    return pred[0, 0, :, :] # Lấy [H, W]

# --- METRICS EVALUATION (NUMPY ONLY) ---
def calc_rmse(depth, ground_truth):
    residual = ((depth - ground_truth) / 256.0) ** 2
    residual[ground_truth == 0.0] = 0.0
    valid_pixels = np.count_nonzero(ground_truth)
    if valid_pixels == 0: return 0.0
    return np.sqrt(np.sum(residual) / valid_pixels)

def calc_absrel(depth, ground_truth):
    diff = depth - ground_truth
    diff[ground_truth == 0.0] = 0.0
    valid_pixels = np.count_nonzero(ground_truth)
    if valid_pixels == 0: return 0.0
    return np.sum(np.abs(diff) / (ground_truth + 1e-6)) / valid_pixels

def run_inference():
    args = parse_arguments()
    
    # 1. Cấu hình ONNX Runtime tối ưu cho Pi 5
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4 
    # Kích hoạt XNNPACK
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.use_xnnpack", "1")

    print(f"Loading Model: {args.onnx_path}")
    session = ort.InferenceSession(args.onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
    
    rgb_dir = args.rgbd_dir / "rgb"
    raw_dir = args.rgbd_dir / "raw_1%"
    gt_dir = args.rgbd_dir / "gt"
    # Lưu output ngay bên trong thư mục test đang chạy để dễ quản lý
    save_dir = args.rgbd_dir / "result_ONNX_epoch_30"
    
    total_rmse = 0.0
    total_absrel = 0.0
    valid_count = 0
    os.makedirs(save_dir, exist_ok=True)

    files = list(rgb_dir.rglob("*.png"))
    print(f"Bắt đầu xử lý {len(files)} ảnh...")

    start_total = time.time()
    
    for i, file_path in enumerate(files):
        # Đường dẫn file raw tương ứng
        rel_path = file_path.relative_to(rgb_dir)
        raw_path = raw_dir / rel_path
        save_path = save_dir / rel_path
        os.makedirs(save_path.parent, exist_ok=True)

        # A. Tiền xử lý (NumPy)
        input_rgb, input_raw, input_hole = prepare_input(file_path, raw_path)

        # B. Inference
        onnx_inputs = {
            'rgb': input_rgb,
            'raw': input_raw,
            'hole_raw': input_hole
        }
        
        t1 = time.time()
        outputs = session.run(None, onnx_inputs)
        inf_time = time.time() - t1

        # C. Hậu xử lý & Lưu (NumPy + OpenCV)
        result = adjust_domain_numpy(outputs[0])
        
        # Cắt bỏ phần viền đen đã padding (lấy đúng kích thước gốc 480x848)
        result = result[0:ORIG_H, 0:ORIG_W]
        
        cv2.imwrite(str(save_path), result)
        
        # D. Tính toán sai số nếu có file GT
        gt_path = gt_dir / rel_path
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt is not None:
                gt = cv2.resize(gt, (ORIG_W, ORIG_H), interpolation=cv2.INTER_NEAREST)
                gt = gt.astype(np.float32)
                pred_eval = result.astype(np.float32)
                
                total_rmse += calc_rmse(pred_eval, gt)
                total_absrel += calc_absrel(pred_eval, gt)
                valid_count += 1

        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(files)}] - Inf Time: {inf_time:.4f}s")

    print(f"\n--- HOÀN TẤT ---")
    print(f"Tổng thời gian: {time.time() - start_total:.2f}s")
    print(f"Trung bình: {(time.time() - start_total)/len(files):.4f}s/ảnh")
    
    if valid_count > 0:
        print(f"\n--- KẾT QUẢ ĐÁNH GIÁ (Trên {valid_count} ảnh) ---")
        print(f"RMSE   = {total_rmse/valid_count:.4f}")
        print(f"AbsRel = {total_absrel/valid_count:.4f}")

if __name__ == "__main__":
    run_inference()