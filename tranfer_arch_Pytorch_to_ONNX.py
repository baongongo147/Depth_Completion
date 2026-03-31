import torch
import torch.onnx
import onnx
from onnxsim import simplify
from src.networks import UNet
from pathlib import Path
import os

def export_and_simplify(model_path, onnx_save_path):
    # --- BƯỚC 1: EXPORT TỪ PYTORCH SANG ONNX (BẢN THÔ) ---
    device = torch.device("cpu")
    network = UNet(rezero=True).to(device)
    
    # Load trọng số
    checkpoint = torch.load(model_path, map_location=device)
    network.load_state_dict(checkpoint["network"])
    network.eval()
    
    # Tạo dữ liệu giả (Input size: 320x448)
    dummy_rgb = torch.randn(1, 3, 320, 448)
    dummy_raw = torch.randn(1, 1, 320, 448)
    dummy_hole = torch.randn(1, 1, 320, 448)
    
    print(f"----------- Step 1: Exporting Raw ONNX -----------")
    
    # Export bản thô (thường bản này sẽ bị cồng kềnh và lỗi Resize)
    torch.onnx.export(
        network, 
        (dummy_rgb, dummy_raw, dummy_hole), 
        onnx_save_path, 
        export_params=True,
        opset_version=12,          # Opset 12 là bản ổn định nhất cho Resize
        do_constant_folding=True,
        input_names=['rgb', 'raw', 'hole_raw'],
        output_names=['output']
    )

    # --- BƯỚC 2: DÙNG ONNX-SIMPLIFIER ĐỂ TỐI ƯU (BẢN SẠCH) ---
    print(f"----------- Step 2: Simplifying ONNX Model -----------")
    
    # Load lại mô hình vừa tạo
    onnx_model = onnx.load(onnx_save_path)
    
    # Sử dụng simplify để dọn dẹp các nút thừa và tính toán hằng số
    # Điều này sẽ sửa lỗi "Resize" mà bạn gặp lúc trước
    model_simp, check = simplify(onnx_model)
    
    if not check:
        print("Cảnh báo: Không thể kiểm chứng mô hình sau khi Simplify!")
    
    # Lưu đè file cũ bằng bản đã tối ưu
    onnx.save(model_simp, onnx_save_path)
    
    print(f"----------- KẾT QUẢ -----------")
    # Kiểm tra lại Opset cuối cùng
    final_model = onnx.load(onnx_save_path)
    print(f"Export thành công! File: {onnx_save_path}")
    print(f"Opset version: {final_model.opset_import[0].version}")
    print(f"Kích thước file đã được tối ưu.")

if __name__ == "__main__":
    MODEL_DIR = "checkpoints/models/epoch_100.pth"
    ONNX_PATH = "checkpoints/models/g2_monodepth.onnx"
    
    export_and_simplify(MODEL_DIR, ONNX_PATH)