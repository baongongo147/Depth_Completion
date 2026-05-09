import onnx
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(model_fp32_path, model_int8_path):
    if not os.path.exists(model_fp32_path):
        print(f"Lỗi: Không tìm thấy file {model_fp32_path}")
        return

    print(f"--- Đang bắt đầu lượng hóa mô hình: {model_fp32_path} ---")
    
    # Lượng hóa Dynamic (An toàn và nhanh nhất cho UNet)
    quantize_dynamic(
        model_input=model_fp32_path,
        model_output=model_int8_path,
        weight_type=QuantType.QUInt8 # Sử dụng QUInt8 để tối ưu nhất cho ARM
    )
    
    print(f"--- Hoàn tất! ---")
    print(f"Mô hình INT8 đã được lưu tại: {model_int8_path}")
    
    # Kiểm tra kích thước
    size_fp32 = os.path.getsize(model_fp32_path) / (1024 * 1024)
    size_int8 = os.path.getsize(model_int8_path) / (1024 * 1024)
    print(f"Dung lượng FP32: {size_fp32:.2f} MB")
    print(f"Dung lượng INT8: {size_int8:.2f} MB")
    print(f"Mô hình đã nhẹ hơn {(1 - size_int8/size_fp32)*100:.1f}%")
    print(f"Dung lượng INT8: {size_int8:.2f} MB (Giảm {((size_fp32-size_int8)/size_fp32)*100:.1f}%)")

if __name__ == "__main__":
    # Đảm bảo đường dẫn này đúng với máy của bạn
    fp32_path = "checkpoints/models/g2_monodepth.onnx"
    int8_path = "checkpoints/models/g2_monodepth_int8.onnx"
    
    quantize_onnx_model(fp32_path, int8_path)