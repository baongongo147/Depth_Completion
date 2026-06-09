import torch
import torch.onnx
import onnx
from onnxsim import simplify
from src.networks import UNet
from pathlib import Path
import os

# Import TensorRT
try:
    import tensorrt as trt
except ImportError:
    trt = None
    print("[Cảnh báo] Thư viện 'tensorrt' chưa được cài đặt. Bước tạo Engine sẽ bị bỏ qua.")


def build_tensorrt_engine(onnx_path, engine_path, fp16=True):
    """
    Biên dịch file ONNX tĩnh thành TensorRT Engine (.engine)
    Tương thích tốt với cả TensorRT phiên bản 8.x và 10.x
    """
    if trt is None:
        print("[Lỗi] Không thể biên dịch TensorRT do thiếu thư viện.")
        return False

    print(f"\n----------- Step 3: Building TensorRT Engine -----------")
    print(f"ONNX Model: {onnx_path}")
    print(f"Engine Path: {engine_path}")

    # Khởi tạo logger của TRT
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Tạo Builder, Network và Parser
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Cấu hình bộ nhớ Workspace (Ví dụ: cấp 2GB để TRT lựa chọn kernel tối ưu nhất)
    # Ghi số trực tiếp vì để phép nhân gây lỗi 
    # workspace_size = 2 * (1024 ** 30)  # 2 GB
    # try:
    #     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    # except AttributeError:
    #     # Hỗ trợ các phiên bản cũ của TensorRT (< 8.4)
    #     config.max_workspace_size = workspace_size

    workspace_size = 2147483648  # 2 GB
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    except AttributeError:
        # Hỗ trợ các phiên bản cũ của TensorRT (< 8.4)
        config.max_workspace_size = workspace_size

    # Kích hoạt FP16 (nếu GPU hỗ trợ) để tối đa hiệu năng của Tensor Cores
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] Đã kích hoạt chế độ FP16 Precision.")
        else:
            print("[WARNING] GPU hiện tại không hỗ trợ FP16 tối ưu phần cứng, chuyển sang FP32.")

    # Đọc và Parse file ONNX
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("[ERROR] Thất bại khi Parse file ONNX!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Biên dịch mô hình sang Engine tuần tự hóa (Serialized Network)
    print("[INFO] Đang tối ưu hóa các lớp đồ thị và biên dịch Engine. Quá trình này có thể mất vài phút...")
    try:
        # API chuẩn cho TensorRT 8.x / 10.x
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[ERROR] Không thể khởi tạo serialized engine.")
            return False
        
        # Lưu file Engine (.engine)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"[SUCCESS] Đã tạo thành công TensorRT Engine tại: {engine_path}")
        return True
        
    except Exception as e:
        print(f"[WARNING] Gặp lỗi khi sử dụng build_serialized_network: {e}")
        # Chế độ Fallback cho một số phiên bản TensorRT cũ
        try:
            print("[INFO] Đang thử phương pháp build cũ (build_engine_with_config)...")
            engine = builder.build_engine_with_config(network, config)
            if engine is not None:
                with open(engine_path, "wb") as f:
                    f.write(engine.serialize())
                print(f"[SUCCESS] Tạo thành công TensorRT Engine (Fallback): {engine_path}")
                return True
        except Exception as fallback_err:
            print(f"[ERROR] Không thể biên dịch Engine qua phương pháp fallback: {fallback_err}")
            return False


def export_and_simplify(model_path, onnx_save_path, engine_save_path):
    # --- BƯỚC 1: EXPORT TỪ PYTORCH SANG ONNX ---
    # Chuyển sang GPU để mô hình được tối ưu hóa trực tiếp trên GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    network = UNet(rezero=True).to(device)
    
    # Load trọng số
    checkpoint = torch.load(model_path, map_location=device)
    network.load_state_dict(checkpoint["network"])
    network.eval()

    # Original size = (320, 448)
    # dummy_rgb = torch.randn(1, 3, 320, 448)
    # dummy_raw = torch.randn(1, 1, 320, 448)
    # dummy_hole = torch.randn(1, 1, 320, 448)

    # Real Dataset size = (640, 480) -> padded to (512, 896)
    # dummy_rgb = torch.randn(1, 3, 512, 896)
    # dummy_raw = torch.randn(1, 1, 512, 896)
    # dummy_hole = torch.randn(1, 1, 512, 896)

    # Target camera size = (640, 480) -> padded to (512, 640)
    dummy_rgb = torch.randn(1, 3, 512, 640, device=device)
    dummy_raw = torch.randn(1, 1, 512, 640, device=device)
    dummy_hole = torch.randn(1, 1, 512, 640, device=device)
    
    print(f"----------- Step 1: Exporting Raw ONNX -----------")
    
    # Export bản thô (thường bản này sẽ bị cồng kềnh và lỗi Resize)
    torch.onnx.export(
        network, 
        (dummy_rgb, dummy_raw, dummy_hole), 
        onnx_save_path, 
        export_params=True,
        opset_version=11,          # Opset 11 ổn định nhất cho Resize
        do_constant_folding=True,
        input_names=['rgb', 'raw', 'hole_raw'],
        output_names=['output'],
        dynamic_axes=None          # Cố định kích thước (Static Shape)
    )

    # --- BƯỚC 2: DÙNG ONNX-SIMPLIFIER ĐỂ TỐI ƯU ---
    print(f"----------- Step 2: Simplifying ONNX Model -----------")
    
    # Load lại mô hình vừa tạo
    onnx_model = onnx.load(onnx_save_path)
    
    # Sử dụng simplify để dọn dẹp các nút thừa và tính toán hằng số
    # Điều này sẽ sửa lỗi "Resize" mà bạn gặp lúc trước
    # model_simp, check = simplify(onnx_model)
    model_simp, check = simplify(
        onnx_model,
        overwrite_input_shapes={
            # 'rgb': [1, 3, 320, 448], 
            # 'raw': [1, 1, 320, 448], 
            # 'hole_raw': [1, 1, 320, 448]
            # 'rgb': [1, 3, 512, 896], 
            # 'raw': [1, 1, 512, 896], 
            # 'hole_raw': [1, 1, 512, 896]
            'rgb': [1, 3, 512, 640], 
            'raw': [1, 1, 512, 640], 
            'hole_raw': [1, 1, 512, 640]
        }
    )
    
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
    import gc

    del network
    del checkpoint

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- BƯỚC 3: BIÊN DỊCH SANG TENSORRT ENGINE ---
    build_tensorrt_engine(onnx_save_path, engine_save_path, fp16=True)

if __name__ == "__main__":
    MODEL_DIR = "checkpoints/models/epoch_30.pth"
    ONNX_PATH = "checkpoints/models/g2_monodepth_epoch_30.onnx"
    ENGINE_PATH = "checkpoints/models/g2_monodepth_epoch_30.engine"
    
    export_and_simplify(MODEL_DIR, ONNX_PATH, ENGINE_PATH)