import os
import argparse
import numpy as np
import onnxruntime as ort
import torch # Vẫn cần torch vì RGBPReader trả về tensor
from PIL import Image
from pathlib import Path
from test_utils import RGBPReader, DepthEvaluation
import time

def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for G2-MonoDepth ONNX Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dir",
        type=lambda x: Path(x),
        default="Test_Datasets/Ibims",
        help="Path to RGBD folder",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="checkpoints/models/g2_monodepth.onnx",
        help="Path to load ONNX model",
    )
    args = parser.parse_args()
    return args

def demo_save_onnx(args):
    print(f"----------- Loading ONNX Model: {args.onnx_path} -----------")
    # Khởi tạo phiên làm việc với ONNX Runtime
    session = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
    
    raw_dirs = ["0%", "1%", "100%"]
    print("----------- Start Inferring on Dataset -----------")
    
    for raw_dir in raw_dirs:
        rgb_dir = args.rgbd_dir / "rgb"
        count = 0
        
        for file in rgb_dir.rglob("*.png"):
            count += 1
            str_file = str(file)
            
            # Sửa đường dẫn linh hoạt cho Windows/Linux
            # Tìm và thay thế cụm "\rgb\" hoặc "/rgb/" tương ứng
            rgb_pattern = f"{os.sep}rgb{os.sep}"
            raw_pattern = f"{os.sep}raw_{raw_dir}{os.sep}"
            save_pattern = f"{os.sep}result_ONNX{raw_dir}{os.sep}"
            
            raw_path = str_file.replace(rgb_pattern, raw_pattern)
            save_path = str_file.replace(rgb_pattern, save_pattern)
            
            rgbd_reader = RGBPReader()
            # 1. Đọc dữ liệu (trả về Tensor)
            rgb_t, raw_t, hole_raw_t = rgbd_reader.read_data(str_file, raw_path)
            
            # 2. Xử lý lỗi 5D -> 4D và chuyển sang NumPy
            if rgb_t.dim() == 5: rgb_t = rgb_t[:, :, :, :, 0]
            if raw_t.dim() == 5: raw_t = raw_t[:, :, :, :, 0]
            if hole_raw_t.dim() == 5: hole_raw_t = hole_raw_t[:, :, :, :, 0]
            
            # Chuyển sang NumPy float32 (ONNX yêu cầu NumPy)
            input_rgb = rgb_t.numpy().astype(np.float32)
            input_raw = raw_t.numpy().astype(np.float32)
            input_hole = hole_raw_t.numpy().astype(np.float32)

            # 3. Chuẩn bị đầu vào cho ONNX (phải khớp tên lúc Export)
            onnx_inputs = {
                'rgb': input_rgb,
                'raw': input_raw,
                'hole_raw': input_hole
            }

            # 4. Chạy mô hình
            onnx_outputs = session.run(None, onnx_inputs)
            pred_np = onnx_outputs[0]

            # 5. Hậu xử lý (Chuyển về tensor để dùng hàm adjust_domain có sẵn)
            pred_tensor = torch.from_numpy(pred_np)
            pred_final = rgbd_reader.adjust_domain(pred_tensor)
            
            # 6. Lưu kết quả
            os.makedirs(str(Path(save_path).parent), exist_ok=True)
            Image.fromarray(pred_final).save(save_path)
            
            if count % 10 == 0:
                print(f"Mode {raw_dir}: Processed {count} images...")

def demo_metric_fixed(args):
    """Tính toán lại metric từ các ảnh đã lưu ở bước trên"""
    raw_dirs = ["0%", "1%", "100%"]
    print("\n----------- Calculating Metrics -----------")
    
    for raw_dir in raw_dirs:
        srmse, ord_error, rmse, rel, count = 0.0, 0.0, 0.0, 0.0, 0.0
        rgb_dir = args.rgbd_dir / "rgb"
        
        for file in rgb_dir.rglob("*.png"):
            count += 1.0
            str_file = str(file)
            
            # Sửa lỗi replace cho Windows
            rgb_p = f"{os.sep}rgb{os.sep}"
            res_p = f"{os.sep}result_ONNX{raw_dir}{os.sep}"
            gt_p  = f"{os.sep}gt{os.sep}"
            
            pred_path = str_file.replace(rgb_p, res_p)
            gt_path = str_file.replace(rgb_p, gt_p)
            
            if not os.path.exists(pred_path): continue

            pred = np.clip(np.array(Image.open(pred_path)).astype(np.float32), 1.0, 65535.0)
            gt = np.array(Image.open(gt_path)).astype(np.float32)
            
            if raw_dir == "0%":
                srmse += DepthEvaluation.srmse(pred, gt)
                ord_error += DepthEvaluation.oe(pred, gt)
            else:
                rmse += DepthEvaluation.rmse(pred, gt)
                rel += DepthEvaluation.absRel(pred, gt)
        
        if count > 0:
            if raw_dir == "0%":
                print(f"0%: srmse={srmse/count:.4f}, oe={ord_error/count:.4f}")
            else:
                print(f"{raw_dir}: rmse={rmse/count:.4f}, Absrel={rel/count:.4f}")

if __name__ == "__main__":
    args = parse_arguments()
    
    start = time.time()
    # Chạy dự đoán và lưu ảnh
    demo_save_onnx(args)
    # Tính toán sai số
    demo_metric_fixed(args)
    
    print(f"\nTotal time: {time.time() - start:.2f} seconds")