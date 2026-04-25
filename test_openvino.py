import os
import argparse
import numpy as np
import openvino as ov
import torch
from PIL import Image
from pathlib import Path
from test_utils import RGBPReader, DepthEvaluation
import time

def parse_arguments():
    parser = argparse.ArgumentParser(
        "G2-MonoDepth OpenVINO Inference Station",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dir",
        type=lambda x: Path(x),
        default="Test_Datasets/Ibims",
        help="Path to RGBD folder",
    )
    parser.add_argument(
        "--model_xml",
        type=str,
        default="checkpoints/models/openvino_model.xml",
        help="Path to OpenVINO .xml file",
    )
    args = parser.parse_args()
    return args

def demo_save_openvino(args, compiled_model):
    print(f"----------- Bắt đầu dự đoán trên tập dữ liệu -----------")
    # raw_dirs = ["0%", "1%", "100%"]
    raw_dirs = ["1%"]
    rgbd_reader = RGBPReader()
    
    # Lấy thông tin đầu ra
    output_layer = compiled_model.output(0)

    for raw_dir in raw_dirs:
        rgb_dir = args.rgbd_dir / "rgb"
        count = 0
        
        for file in rgb_dir.rglob("*.png"):
            count += 1
            str_file = str(file)
            
            # Xử lý đường dẫn linh hoạt (Windows/Linux)
            rgb_p = f"{os.sep}rgb{os.sep}"
            raw_p = f"{os.sep}raw_{raw_dir}{os.sep}"
            # Lưu vào thư mục output riêng biệt để tránh ghi đè ảnh gốc
            save_p = f"{os.sep}output{os.sep}result_{raw_dir}{os.sep}"
            
            raw_path = str_file.replace(rgb_p, raw_p)
            save_path = str_file.replace(rgb_p, save_p)
            
            # 1. Đọc dữ liệu (trả về Tensor)
            rgb_t, raw_t, hole_raw_t = rgbd_reader.read_data(str_file, raw_path)
            
            # 2. Xử lý lỗi 5D -> 4D
            if rgb_t.dim() == 5: rgb_t = rgb_t[:, :, :, :, 0]
            if raw_t.dim() == 5: raw_t = raw_t[:, :, :, :, 0]
            if hole_raw_t.dim() == 5: hole_raw_t = hole_raw_t[:, :, :, :, 0]
            
            # 3. Chuyển sang NumPy float32 cho OpenVINO
            input_rgb = rgb_t.numpy().astype(np.float32)
            input_raw = raw_t.numpy().astype(np.float32)
            input_hole = hole_raw_t.numpy().astype(np.float32)

            # 4. Chạy mô hình (Sử dụng Dictionary để khớp tên layer)
            results = compiled_model({
                "rgb": input_rgb,
                "raw": input_raw,
                "hole_raw": input_hole
            })
            
            pred_np = results[output_layer]

            # 5. Hậu xử lý (Chuyển về tensor để dùng adjust_domain gốc)
            pred_tensor = torch.from_numpy(pred_np)
            pred_final = rgbd_reader.adjust_domain(pred_tensor)
            
            # 6. Lưu kết quả
            os.makedirs(str(Path(save_path).parent), exist_ok=True)
            Image.fromarray(pred_final).save(save_path)
            
            if count % 20 == 0:
                print(f"Mode {raw_dir}: Đã xong {count} ảnh...")

def demo_metric_openvino(args):
    """Tính toán sai số dựa trên các ảnh đã lưu ở bước trên"""
    # raw_dirs = ["0%", "1%", "100%"]
    raw_dirs = ["1%"]
    print("\n----------- Đang tính toán Metrics (RMSE/MAE) -----------")
    
    for raw_dir in raw_dirs:
        srmse, ord_error, rmse, rel, count = 0.0, 0.0, 0.0, 0.0, 0.0
        rgb_dir = args.rgbd_dir / "rgb"
        
        for file in rgb_dir.rglob("*.png"):
            str_file = str(file)
            
            rgb_p = f"{os.sep}rgb{os.sep}"
            res_p = f"{os.sep}output{os.sep}result_{raw_dir}{os.sep}"
            gt_p  = f"{os.sep}gt{os.sep}"
            
            pred_path = str_file.replace(rgb_p, res_p)
            gt_path = str_file.replace(rgb_p, gt_p)
            
            if not os.path.exists(pred_path) or not os.path.exists(gt_path): 
                continue

            count += 1.0
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
    
    # Khởi tạo OpenVINO 2026
    print(f"Loading OpenVINO Model: {args.model_xml}")
    core = ov.Core()
    model = core.read_model(model=args.model_xml)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    
    start = time.time()
    
    # 1. Dự đoán và lưu ảnh
    demo_save_openvino(args, compiled_model)
    
    # 2. Đánh giá sai số
    demo_metric_openvino(args)
    
    print(f"\nTotal time: {time.time() - start:.2f} seconds")