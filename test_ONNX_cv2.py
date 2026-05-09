import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbd_dir", type=Path, default="Test_Datasets")
    parser.add_argument("--onnx_path", type=str, default="checkpoints/models/g2_monodepth.onnx")
    return parser.parse_args()

# --- REPLACEMENT FOR RGBPReader (NUMPY ONLY) ---
def prepare_input(rgb_path, raw_path):
    # Đọc ảnh RGB bằng OpenCV (Nhanh hơn PIL)
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    # [H, W, C] -> [1, C, H, W]
    rgb = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

    # Đọc ảnh Raw (Depth)
    raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
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
    save_dir = args.rgbd_dir.parent / "output_optimized"
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
        cv2.imwrite(str(save_path), result)

        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(files)}] - Inf Time: {inf_time:.4f}s")

    print(f"\n--- HOÀN TẤT ---")
    print(f"Tổng thời gian: {time.time() - start_total:.2f}s")
    print(f"Trung bình: {(time.time() - start_total)/len(files):.4f}s/ảnh")

if __name__ == "__main__":
    run_inference()