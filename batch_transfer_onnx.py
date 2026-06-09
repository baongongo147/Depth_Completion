import os
import sys
import importlib.util
from pathlib import Path

# Load module dynamically to support files starting with numbers
spec = importlib.util.spec_from_file_location(
    "transfer_v2", 
    Path(__file__).parent / "tranfer_arch_Pytorch_to_ONNX.py"
)
transfer_v2 = importlib.util.module_from_spec(spec)
sys.modules["transfer_v2"] = transfer_v2
spec.loader.exec_module(transfer_v2)

def main():
    models_dir = Path("checkpoints/models")
    if not models_dir.exists():
        print(f"Lỗi: Thư mục '{models_dir}' không tồn tại.")
        return

    # Danh sách các epoch muốn chuyển đổi
    target_epochs = [5, 8, 10, 29, 30, 100]
    
    print("====================================================")
    print("BẮT ĐẦU BATCH TRANSFER PYTORCH TO ONNX (512x640)")
    print("====================================================")

    for epoch in target_epochs:
        pth_name = f"epoch_{epoch}.pth"
        pth_path = models_dir / pth_name
        
        if not pth_path.exists():
            print(f"Bỏ qua: Không tìm thấy {pth_name} trong {models_dir}.")
            continue
            
        onnx_name = f"g2_monodepth_epoch_{epoch}.onnx"
        engine_name = f"g2_monodepth_epoch_{epoch}.engine"
        onnx_path = models_dir / onnx_name
        engine_path = models_dir / engine_name
        
        print(f"\n[XỬ LÝ] Chuyển đổi: {pth_name} -> {onnx_name} và {engine_name}...")
        try:
            transfer_v2.export_and_simplify(
                str(pth_path),
                str(onnx_path),
                str(engine_path)
            )
            print(f"[OK] Đã lưu {onnx_name} và {engine_name}")
        except Exception as e:
            print(f"[LỖI] Không thể chuyển đổi {pth_name}: {e}")

    print("\n====================================================")
    print("HOÀN TẤT BATCH TRANSFER!")
    print("====================================================")

if __name__ == "__main__":
    main()
