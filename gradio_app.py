import json
import os
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import gradio as gr
import queue
import threading

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

try:
    from rplidar import RPLidar
except ImportError:
    RPLidar = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import openvino as ov
except ImportError:
    ov = None

try:
    import matplotlib
    import matplotlib.cm as cm
except ImportError:
    matplotlib = None
    cm = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

import base64
import io
import urllib.request

# Global variable to store latest data received from Pi 5
LATEST_PI5_DATA = {
    "rgb": None,
    "lidar_overlay": None,
    "sparse_depth": None,
    "timestamp": 0.0,
    "status": "Chưa nhận được dữ liệu từ Pi 5"
}
LATEST_PI5_LOCK = threading.Lock()


from src.networks import UNet

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "checkpoints" / "models"


def scan_models(model_dir=MODEL_DIR):
    if not model_dir.exists():
        return []
    models = []
    for path in sorted(model_dir.iterdir()):
        if path.suffix.lower() in {".pth", ".onnx", ".xml", ".engine"}:
            models.append(str(path))
    return models


DEFAULT_LIDAR_PORT = "COM3" if os.name == "nt" else "/dev/ttyUSB0"


def is_live_capture_supported():
    return Picamera2 is not None and RPLidar is not None and cv2 is not None


def load_camera_lidar_calibration(calib_path=ROOT / "picam3_calib.npz", extrin_path=ROOT / "extrinsics.npz"):
    if not calib_path.exists() or not extrin_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy file calibration Picamera2/RPLIDAR. Hãy đặt picam3_calib.npz và extrinsics.npz trong thư mục app."
        )
    calib = np.load(calib_path)
    extrin = np.load(extrin_path)
    return calib["mtx"], calib["dist"], extrin["t"], extrin["r"]


def capture_picamera2_rgb(target_size=(640, 480)):
    if Picamera2 is None:
        raise RuntimeError("Picamera2 chưa được cài đặt hoặc không tìm thấy trên hệ thống.")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": target_size}
    )
    picam2.configure(config)
    picam2.start()
    try:
        frame = picam2.capture_array()
    finally:
        picam2.stop()
        picam2.close()
    return frame

def project_lidar_to_image(scan, K, dist_coeffs, rvec, tvec, image_shape):
    if cv2 is None:
        raise RuntimeError("OpenCV chưa được cài đặt, cần để xử lý LiDAR projection.")

    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)
    
    # Ensure correct shape for rvec and tvec
    rvec = np.asarray(rvec, dtype=np.float32)
    if rvec.size == 9: # 3x3 matrix
        rvec, _ = cv2.Rodrigues(rvec)
    rvec = rvec.reshape(3, 1)
    
    tvec = np.asarray(tvec, dtype=np.float32).reshape(3, 1)

    visible_points = []
    lidar_pts = []
    distances = []
    
    for item in scan:
        if len(item) == 3:
            quality, angle, distance = item
        else:
            angle, distance = item
            quality = 15
            
        if distance <= 0 or not np.isfinite(distance):
            continue
            
        # lx, lz from demo logic: lx = distance * sin(angle_rad), lz = distance * cos(angle_rad)
        rad = math.radians(angle)
        lx = distance * math.sin(rad)
        lz = distance * math.cos(rad)
        
        # position[2] < 0.0 check from demo: lz < 0
        if lz < 0.0:
            continue
            
        lidar_pts.append([lx, 0.0, lz])
        distances.append(distance)

    if not lidar_pts:
        return visible_points

    lidar_pts_arr = np.array(lidar_pts, dtype=np.float32)
    img_pts, _ = cv2.projectPoints(lidar_pts_arr, rvec, tvec, K, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2)

    height, width = image_shape[:2]
    for i, pt in enumerate(img_pts):
        if not np.all(np.isfinite(pt)):
            continue
        x, y = pt
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            visible_points.append((ix, iy, distances[i]))
            
    return visible_points


def draw_lidar_overlay(rgb_frame, visible_points, point_radius=1):
    """Draw LiDAR points on the RGB image with distance-based coloring (Red for near, Blue for far)."""
    overlay = Image.fromarray(rgb_frame).copy()
    if not visible_points:
        return overlay

    draw = ImageDraw.Draw(overlay)
    
    min_dist = 150.0  # mm
    max_dist = 12000.0  # mm
    
    for x, y, dist in visible_points:
        dist_clipped = np.clip(dist, min_dist, max_dist)
        t = (dist_clipped - min_dist) / (max_dist - min_dist)
        
        # Close = Red, Far = Blue (from demo: r = (1-t)*255, b = t*255)
        r = int((1.0 - t) * 255)
        g = 0
        b = int(t * 255)
        color = (r, g, b)
        
        # Use point_radius = 1 to draw single pixels/dots
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=color,
        )
    return overlay

def build_sparse_depth_from_lidar_points(image_shape, visible_points):
    h, w = image_shape[:2]
    sparse = np.zeros((h, w), dtype=np.float32)
    for x, y, distance in visible_points:
        if 0 <= x < w and 0 <= y < h:
            sparse[y, x] = float(distance) / 65535.0
    return Image.fromarray((sparse * 65535.0).astype(np.uint16))


def build_sparse_depth_from_lidar(gt_depth, lidar_json_path):
    if isinstance(gt_depth, Image.Image):
        gt_depth = np.array(gt_depth)
    if gt_depth.ndim == 3 and gt_depth.shape[2] == 3:
        gt_depth = gt_depth[..., 0]

    if gt_depth.dtype != np.float32:
        gt_depth = gt_depth.astype(np.float32) / 65535.0

    sparse = np.zeros_like(gt_depth, dtype=np.float32)
    with open(lidar_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("labels", {}).get("image_pixel_points", [])
    h, w = gt_depth.shape
    for px, py in points:
        if 0 <= px < w and 0 <= py < h:
            sparse[py, px] = gt_depth[py, px]
    return sparse


def read_rgb_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.array(image)
    return np.array(Image.fromarray(image).convert("RGB"))


def read_depth_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        image = image.convert("I")
        data = np.array(image)
    else:
        data = np.array(image)
    if data.ndim == 3:
        data = data[..., 0]
    if data.dtype == np.uint16:
        return data.astype(np.float32) / 65535.0
    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0
    data = data.astype(np.float32)
    if data.max() > 1.0:
        return data / 65535.0
    return data


def resize_to_match(rgb, depth):
    if rgb.shape[:2] == depth.shape[:2]:
        return depth
    depth_img = Image.fromarray((depth * 65535.0).astype(np.uint16))
    depth_img = depth_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST)
    return np.array(depth_img).astype(np.float32) / 65535.0


def resize_for_model(rgb_img, sparse_img, target_h=512, target_w=896):
    """
    Process RGB and sparse depth to target size for ONNX/OpenVINO models.
    Pads if the input image can fit into the target size, otherwise resizes.
    Returns processed arrays, original shape, and the mode ('pad' or 'resize').
    """
    if isinstance(rgb_img, Image.Image):
        rgb_np = np.array(rgb_img.convert("RGB"))
    else:
        rgb_np = np.array(rgb_img)

    if isinstance(sparse_img, Image.Image):
        sparse_np = np.array(sparse_img.convert("I")).astype(np.float32) / 65535.0
    else:
        sparse_np = read_depth_image(sparse_img)

    original_h, original_w = rgb_np.shape[:2]

    # If the input image is smaller than or equal to the target size, pad it to preserve LiDAR coordinates
    if original_h <= target_h and original_w <= target_w:
        pad_bottom = target_h - original_h
        pad_right = target_w - original_w
        if cv2 is not None:
            rgb_processed = cv2.copyMakeBorder(rgb_np, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            sparse_processed = cv2.copyMakeBorder(sparse_np, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0.0)
        else:
            rgb_processed = np.pad(rgb_np, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant')
            sparse_processed = np.pad(sparse_np, ((0, pad_bottom), (0, pad_right)), mode='constant')
        return rgb_processed, sparse_processed, (original_h, original_w), "pad"
    else:
        # Fallback to resizing if the input image doesn't fit in the target shape
        rgb_resized = Image.fromarray(rgb_np).resize((target_w, target_h), resample=Image.BILINEAR)
        rgb_resized = np.array(rgb_resized)

        sparse_img_pil = Image.fromarray((sparse_np * 65535.0).astype(np.uint16))
        sparse_resized = sparse_img_pil.resize((target_w, target_h), resample=Image.NEAREST)
        sparse_resized = np.array(sparse_resized).astype(np.float32) / 65535.0
        return rgb_resized, sparse_resized, (original_h, original_w), "resize"


def pad_to_multiple(tensor, multiple=64):
    _, _, h, w = tensor.shape
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, None
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def adjust_domain(pred_tensor, relative):
    pred = pred_tensor.squeeze().cpu().numpy()
    if relative:
        min_v = float(np.min(pred))
        max_v = float(np.max(pred))
        if max_v - min_v > 1e-6:
            pred = (pred - min_v) / (max_v - min_v)
        else:
            pred = np.zeros_like(pred)
    pred = np.clip(pred * 65535.0, 0, 65535).astype(np.uint16)
    return pred


def depth_to_colormap(depth_uint16):
    if cm is None:
        gray = (depth_uint16 / 256).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")
    depth_norm = depth_uint16.astype(np.float32) / 65535.0
    cmap = matplotlib.colormaps.get_cmap("turbo")
    rgba = cmap(depth_norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def prepare_model_inputs(rgb_image, sparse_image):
    rgb_np = read_rgb_image(rgb_image)
    raw_np = read_depth_image(sparse_image)
    raw_np = resize_to_match(rgb_np, raw_np)
    rgb_tensor = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    raw_tensor = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0)
    hole_tensor = (raw_tensor > 0).to(torch.float32)
    relative = float(raw_tensor.sum()) == 0.0
    return rgb_tensor, raw_tensor, hole_tensor, relative


class TRTModel:
    """TensorRT inference using native PyCUDA memory management (matching test_engine.py approach).
    
    Uses the new TensorRT API (num_io_tensors, get_tensor_name, set_tensor_address, 
    execute_async_v3) with PyCUDA buffers to avoid CUDA context issues in Gradio threads.
    """
    def __init__(self, engine_path):
        if trt is None:
            raise ImportError("Thư viện tensorrt chưa được cài đặt.")
        if cuda is None:
            raise ImportError("Thư viện pycuda chưa được cài đặt.")
        
        # 1. Khởi tạo CUDA Context thủ công cho riêng mô hình này
        self.cfx = cuda.Device(0).make_context()
        
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                raise RuntimeError(f"Không thể load TensorRT engine từ {engine_path}")
            
            self.context = self.engine.create_execution_context()
            
            # Allocate buffers using PyCUDA (same approach as test_engine.py)
            self.trt_inputs = []
            self.trt_outputs = []
            self.stream = cuda.Stream()
            
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                tensor_info = {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "host": host_mem,
                    "device": device_mem,
                }
                
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.trt_inputs.append(tensor_info)
                else:
                    self.trt_outputs.append(tensor_info)
            
            # Auto-detect target shape from the first input (rgb)
            self.target_h = 512
            self.target_w = 640
            try:
                if self.trt_inputs:
                    rgb_shape = self.trt_inputs[0]['shape']
                    if len(rgb_shape) == 4:
                        h = rgb_shape[2]
                        w = rgb_shape[3]
                        if h > 0 and w > 0:
                            self.target_h = h
                            self.target_w = w
            except Exception as e:
                print(f"[TensorRT] Cảnh báo phát hiện shape: {e}")
        finally:
            # Thu hồi context ra khỏi luồng khởi tạo để tránh xung đột ngữ cảnh sau này
            self.cfx.pop()

    def infer(self, rgb_tensor, raw_tensor, hole_tensor):
        """Run inference using PyCUDA host/device memory copies (same as test_engine.py)."""
        # 2. Đẩy CUDA context vào luồng hiện tại của Gradio đang thực hiện cuộc gọi hàm này
        self.cfx.push()
        try:
            # Convert torch tensors to numpy arrays
            input_arrays = [
                rgb_tensor.numpy().astype(np.float32),
                raw_tensor.numpy().astype(np.float32),
                hole_tensor.numpy().astype(np.float32),
            ]
            
            # Copy input data to host pinned memory, then to device
            for tensor_info, array in zip(self.trt_inputs, input_arrays):
                array = np.ascontiguousarray(array)
                flat_host = np.frombuffer(tensor_info["host"], dtype=tensor_info["dtype"])
                if flat_host.size != array.size:
                    raise ValueError(
                        f"Input shape mismatch for '{tensor_info['name']}': "
                        f"expected {flat_host.size}, got {array.size}"
                    )
                flat_host[:] = array.ravel()
                cuda.memcpy_htod_async(tensor_info["device"], tensor_info["host"], self.stream)
            
            # Set tensor addresses for execution context
            for tensor_info in self.trt_inputs:
                self.context.set_tensor_address(tensor_info["name"], int(tensor_info["device"]))
            for tensor_info in self.trt_outputs:
                self.context.set_tensor_address(tensor_info["name"], int(tensor_info["device"]))
            
            # Execute inference
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # Copy output from device to host
            for out in self.trt_outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            self.stream.synchronize()
            
            # Reconstruct output as torch tensor
            results = []
            for out in self.trt_outputs:
                result = np.frombuffer(out["host"], dtype=out["dtype"]).reshape(out["shape"])
                results.append(result.copy())
            
            return torch.from_numpy(results[0])
        finally:
            # 3. Thu hồi context khỏi luồng hiện tại để dọn sạch stack ngữ cảnh CUDA
            self.cfx.pop()


class ModelManager:
    def __init__(self):
        self.model_path = None
        self.model_type = None
        self.model = None
        self.session = None
        self.compiled = None
        self.trt_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Khóa Lock để ngăn chặn xung đột đồng thời giữa các luồng Gradio
        self.lock = threading.Lock()

    def load(self, model_path):
        with self.lock:
            # Kiểm tra lại một lần nữa phòng trường hợp một luồng khác đã load xong trong lúc đợi khóa
            if self.model_path == model_path:
                return f"Loaded Model: {Path(model_path).name} (luồng khác đã thực hiện tải)"

            self.model = None
            self.session = None
            self.compiled = None
            self.trt_model = None
            self.model_type = Path(model_path).suffix.lower()
            
            try:
                if self.model_type == ".pth":
                    self.model = UNet(rezero=True)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "network" in checkpoint:
                        state_dict = checkpoint["network"]
                    else:
                        state_dict = checkpoint
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    self.model_path = model_path
                    return f"Loaded PyTorch model: {Path(model_path).name}"
                if self.model_type == ".onnx":
                    if ort is None:
                        raise RuntimeError("onnxruntime is not installed")
                    self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                    try:
                        input_shape = self.session.get_inputs()[0].shape
                        self.target_h = input_shape[2]
                        self.target_w = input_shape[3]
                    except Exception as e:
                        print(f"Không tự động phát hiện được kích thước model ONNX: {e}")
                        self.target_h = 512
                        self.target_w = 640
                    self.model_path = model_path
                    return f"Loaded ONNX model: {Path(model_path).name} with shape {self.target_w}x{self.target_h}"
                if self.model_type == ".xml":
                    if ov is None:
                        raise RuntimeError("openvino is not installed")
                    core = ov.Core()
                    model = core.read_model(model=model_path)
                    self.compiled = core.compile_model(model=model, device_name="CPU")
                    try:
                        input_shape = self.compiled.inputs[0].shape
                        self.target_h = input_shape[2]
                        self.target_w = input_shape[3]
                    except Exception as e:
                        print(f"Không tự động phát hiện được kích thước model OpenVINO: {e}")
                        self.target_h = 512
                        self.target_w = 640
                    self.model_path = model_path
                    return f"Loaded OpenVINO model: {Path(model_path).name} with shape {self.target_w}x{self.target_h}"
                if self.model_type == ".engine":
                    if trt is None:
                        raise RuntimeError("tensorrt is not installed")
                    if cuda is None:
                        raise RuntimeError("pycuda is not installed, cannot run TensorRT .engine model")
                    self.trt_model = TRTModel(model_path)
                    self.target_h = self.trt_model.target_h
                    self.target_w = self.trt_model.target_w
                    self.model_path = model_path
                    return f"Loaded TensorRT Engine: {Path(model_path).name} with shape {self.target_w}x{self.target_h}"
                raise RuntimeError("Unsupported model type")
            except Exception:
                # Load thất bại → reset model_path để hệ thống thử tải lại ở lần gọi sau
                self.model_path = None
                raise

    def infer(self, rgb_tensor, raw_tensor, hole_tensor, original_shape=None):
        with self.lock:
            if self.model_type == ".pth":
                padded_rgb, pad_info = pad_to_multiple(rgb_tensor, 64)
                padded_raw, _ = pad_to_multiple(raw_tensor, 64)
                padded_hole, _ = pad_to_multiple(hole_tensor, 64)
                padded_rgb = padded_rgb.to(self.device)
                padded_raw = padded_raw.to(self.device)
                padded_hole = padded_hole.to(self.device)        
                with torch.no_grad():
                    pred = self.model(padded_rgb, padded_raw, padded_hole)
                if pad_info is not None:
                    pad_h, pad_w = pad_info
                    if pad_h > 0:
                        pred = pred[:, :, :-pad_h, :]
                    if pad_w > 0:
                        pred = pred[:, :, :, :-pad_w]
                return pred.cpu()
            if self.model_type == ".onnx":
                if self.session is None:
                    raise RuntimeError("ONNX session is not loaded")
                inputs = {
                    input.name: np.ascontiguousarray(t.numpy().astype(np.float32))
                    for input, t in zip(self.session.get_inputs(), [rgb_tensor, raw_tensor, hole_tensor])
                }
                outputs = self.session.run(None, inputs)
                return torch.from_numpy(np.array(outputs[0]))
            if self.model_type == ".xml":
                if self.compiled is None:
                    raise RuntimeError("OpenVINO model is not loaded")
                inputs = {
                    inp.get_any_name(): np.ascontiguousarray(t.numpy().astype(np.float32))
                    for inp, t in zip(self.compiled.inputs, [rgb_tensor, raw_tensor, hole_tensor])
                }
                results = self.compiled(inputs)
                if isinstance(results, dict):
                    output = next(iter(results.values()))
                else:
                    output = results
                return torch.from_numpy(np.array(output))
            if self.model_type == ".engine":
                if self.trt_model is None:
                    raise RuntimeError("TensorRT model is not loaded")
                return self.trt_model.infer(rgb_tensor, raw_tensor, hole_tensor)
            raise RuntimeError("Unsupported model type")


model_manager = ModelManager()
MODEL_OPTIONS = scan_models()


def run_model_inference(rgb_img, sparse_img, model_path):
    if model_path is None or model_path == "":
        return None, None, "Chưa chọn model", ""
    if model_manager.model_path != model_path:
        try:
            model_manager.load(model_path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return None, None, f"Không load được model: {exc}", ""
    try:
        model_type = Path(model_path).suffix.lower()
        if model_type in {".onnx", ".xml", ".engine"}:
            target_h = getattr(model_manager, 'target_h', 512)
            target_w = getattr(model_manager, 'target_w', 640)
            rgb_np, sparse_np, original_shape, mode = resize_for_model(rgb_img, sparse_img, target_h=target_h, target_w=target_w)
            rgb_tensor, raw_tensor, hole_tensor, relative = prepare_model_inputs(
                Image.fromarray(rgb_np),
                Image.fromarray((sparse_np * 65535.0).astype(np.uint16))
            )
        else:
            rgb_tensor, raw_tensor, hole_tensor, relative = prepare_model_inputs(rgb_img, sparse_img)
            original_shape = None
            mode = None

        start = time.time()
        pred_tensor = model_manager.infer(rgb_tensor, raw_tensor, hole_tensor, original_shape)
        elapsed = time.time() - start

        if original_shape is not None and model_type in {".onnx", ".xml", ".engine"}:
            pred_np = pred_tensor.squeeze().numpy()
            orig_h, orig_w = original_shape
            if mode == "pad":
                # Crop back the padded region
                pred_np = pred_np[0:orig_h, 0:orig_w]
                pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
            else:
                pred_img = Image.fromarray((pred_np * 65535.0).astype(np.uint16))
                pred_img = pred_img.resize((orig_w, orig_h), resample=Image.NEAREST)
                pred_tensor = torch.from_numpy(np.array(pred_img).astype(np.float32) / 65535.0).unsqueeze(0).unsqueeze(0)

        depth_uint16 = adjust_domain(pred_tensor, relative)
        depth_img = Image.fromarray((depth_uint16 // 256).astype(np.uint8))
        colormap_img = depth_to_colormap(depth_uint16)
        status = f"Inference hoàn thành trong {elapsed:.2f}s"
        return depth_img, colormap_img, status, f"{elapsed:.2f} seconds"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, None, f"Lỗi inference: {exc}", ""


PI5_STREAM_URL = "http://192.168.3.2:8080/video"
PI5_STREAM_THREAD = None
PI5_STREAM_RUNNING = False

def pi5_receiver_worker(url):
    global PI5_STREAM_RUNNING
    PI5_STREAM_RUNNING = True
    print(f"[Mạng] Bắt đầu luồng nhận dữ liệu Pi 5 từ: {url}")
    
    retry_delay = 2
    while PI5_STREAM_RUNNING:
        try:
            with LATEST_PI5_LOCK:
                LATEST_PI5_DATA["status"] = f"Đang kết nối tới {url}..."
                
            stream = urllib.request.urlopen(url, timeout=5)
            
            with LATEST_PI5_LOCK:
                LATEST_PI5_DATA["status"] = "Đã kết nối, đang nhận dữ liệu..."
                
            bytes_buffer = b''
            
            while PI5_STREAM_RUNNING:
                chunk = stream.read(4096)
                if not chunk:
                    raise RuntimeError("Mất kết nối stream dữ liệu thô (empty read)")
                bytes_buffer += chunk
                
                boundary_idx = bytes_buffer.find(b'--databoundary')
                if boundary_idx != -1:
                    package_data = bytes_buffer[:boundary_idx]
                    bytes_buffer = bytes_buffer[boundary_idx + len(b'--databoundary'):]
                    
                    if b'X-Lidar-Data' in package_data:
                        lines = package_data.split(b'\r\n')
                        lidar_json = None
                        
                        for line in lines:
                            if line.startswith(b'X-Lidar-Data:'):
                                lidar_json_str = line.split(b'X-Lidar-Data:')[1].strip().decode('utf-8')
                                lidar_json = json.loads(lidar_json_str)
                                break
                        
                        header_end = package_data.find(b'\r\n\r\n')
                        if header_end != -1 and lidar_json is not None:
                            jpg_bytes = package_data[header_end + 4:]
                            
                            if len(jpg_bytes) > 0:
                                img_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
                                rgb_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                
                                if rgb_np is not None:
                                    rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
                                    rgb_img = Image.fromarray(rgb_np)
                                    
                                    scan = []
                                    for pt in lidar_json:
                                        if len(pt) == 2:
                                            scan.append((15, pt[0], pt[1]))
                                        elif len(pt) == 3:
                                            scan.append((pt[0], pt[1], pt[2]))
                                            
                                    lidar_overlay_img = rgb_img
                                    sparse_img = None
                                    if scan:
                                        try:
                                            K, dist_coeffs, tvec, rvec = load_camera_lidar_calibration()
                                            visible_points = project_lidar_to_image(scan, K, dist_coeffs, rvec, tvec, rgb_np.shape)
                                            lidar_overlay_img = draw_lidar_overlay(rgb_np, visible_points)
                                            sparse_img = build_sparse_depth_from_lidar_points(rgb_np.shape, visible_points)
                                        except Exception as proj_err:
                                            print(f"Lỗi project LiDAR trên PC: {proj_err}")
                                            
                                    with LATEST_PI5_LOCK:
                                        LATEST_PI5_DATA["rgb"] = rgb_img
                                        LATEST_PI5_DATA["lidar_overlay"] = lidar_overlay_img
                                        if sparse_img is not None:
                                            LATEST_PI5_DATA["sparse_depth"] = sparse_img
                                        LATEST_PI5_DATA["timestamp"] = time.time()
                                        LATEST_PI5_DATA["status"] = f"Đang nhận dữ liệu thời gian thực ({len(scan)} điểm LiDAR)"
                                        
        except Exception as err:
            print(f"[Cảnh báo] Lỗi kết nối / nhận dữ liệu: {err}")
            with LATEST_PI5_LOCK:
                LATEST_PI5_DATA["status"] = f"Lỗi: {err}. Đang thử kết nối lại sau {retry_delay} giây..."
            time.sleep(retry_delay)
            
    print("[Mạng] Đã dừng luồng nhận dữ liệu Pi 5.")


def start_pi5_stream_client(url):
    global PI5_STREAM_THREAD, PI5_STREAM_RUNNING, PI5_STREAM_URL
    PI5_STREAM_URL = url
    stop_pi5_stream_client()
    PI5_STREAM_RUNNING = True
    PI5_STREAM_THREAD = threading.Thread(target=pi5_receiver_worker, args=(url,), daemon=True)
    PI5_STREAM_THREAD.start()


def stop_pi5_stream_client():
    global PI5_STREAM_RUNNING, PI5_STREAM_THREAD
    PI5_STREAM_RUNNING = False
    if PI5_STREAM_THREAD is not None:
        PI5_STREAM_THREAD.join(timeout=1.0)
        PI5_STREAM_THREAD = None



def capture_realsense_frame():
    if rs is None:
        raise RuntimeError("Thư viện pyrealsense2 chưa được cài đặt.")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start pipeline
    profile = pipeline.start(config)
    
    # Align depth to color stream
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    try:
        # Chờ camera ổn định phơi sáng
        for _ in range(10):
            pipeline.wait_for_frames()
            
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("Không lấy được frame từ RealSense")
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        return color_image_rgb, depth_image
    finally:
        pipeline.stop()


def live_infer_pipeline(model_path):
    """Lấy dữ liệu mới nhất nhận được từ Pi 5, đồng thời chụp từ RealSense làm so sánh, rồi chạy inference."""
    # Trả về: Pi5_RGB, Pi5_Lidar_Overlay, Pred_Depth, Pred_Colormap, RS_RGB, RS_Depth_Colormap, Status, Elapsed
    empty = (None, None, None, None, None, None, "", "")
    if model_path is None or model_path == "":
        return (*empty[:6], "Chưa chọn model", "")
        
    with LATEST_PI5_LOCK:
        rgb = LATEST_PI5_DATA["rgb"]
        lidar_overlay = LATEST_PI5_DATA["lidar_overlay"]
        sparse = LATEST_PI5_DATA["sparse_depth"]
        status_msg = LATEST_PI5_DATA["status"]
        timestamp = LATEST_PI5_DATA["timestamp"]
        
    if rgb is None or sparse is None:
        return (*empty[:6], f"Chưa có dữ liệu từ Pi 5. Trạng thái hiện tại: {status_msg}", "")
        
    age = time.time() - timestamp
    age_msg = f" (Dữ liệu cũ từ {age:.1f}s trước)" if age > 10 else ""
    
    # Thử capture RealSense
    rs_rgb_img = None
    rs_colormap_img = None
    rs_msg = ""
    if rs is not None:
        try:
            rs_rgb_arr, rs_depth_arr = capture_realsense_frame()
            d_min, d_max = rs_depth_arr.min(), rs_depth_arr.max()
            if d_max - d_min > 0:
                depth_scaled = ((rs_depth_arr - d_min) / (d_max - d_min) * 65535.0).astype(np.uint16)
            else:
                depth_scaled = np.zeros_like(rs_depth_arr, dtype=np.uint16)
            rs_colormap_img = depth_to_colormap(depth_scaled)
            rs_rgb_img = Image.fromarray(rs_rgb_arr)
            rs_msg = " | RealSense OK"
        except Exception as e:
            rs_msg = f" | Lỗi RealSense: {e}"
    else:
        rs_msg = " | RealSense không khả dụng"

    try:
        depth_img, colormap_img, status, elapsed = run_model_inference(rgb, sparse, model_path)
        status = f"Dữ liệu từ Pi 5{age_msg}{rs_msg} | {status}"
        return rgb, lidar_overlay, depth_img, colormap_img, rs_rgb_img, rs_colormap_img, status, elapsed
    except Exception as exc:
        return (*empty[:6], f"Live capture error: {exc}", "")


def infer_pipeline(model_path, uploaded_rgb, uploaded_depth):
    if model_path is None or model_path == "":
        return None, None, "Chưa chọn model", ""
    if uploaded_rgb is None or uploaded_depth is None:
        return None, None, "Cần upload cả RGB và sparse depth", ""
    return run_model_inference(uploaded_rgb, uploaded_depth, model_path)


def update_pi5_connection(url):
    start_pi5_stream_client(url)
    return f"Đang kết nối tới {url}..."


def main():
    # Khởi chạy stream client nhận dữ liệu từ Pi 5 tự động
    start_pi5_stream_client("http://192.168.3.2:8080/video")

    # JavaScript lắng nghe phím Space để kích hoạt nút Capture
    head_html = """
    <script>
    document.addEventListener("keydown", function(e) {
        if (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA") {
            return;
        }
        if (e.code === "Space") {
            e.preventDefault();
            const buttons = Array.from(document.querySelectorAll("button"));
            const captureBtn = buttons.find(btn => btn.textContent.includes("Inference từ dữ liệu") || btn.textContent.includes("Capture"));
            if (captureBtn) {
                captureBtn.click();
            }
        }
    });
    </script>
    """

    with gr.Blocks(title="Depth Completion Gradio App (PC Server)", head=head_html) as demo:
        gr.Markdown(
            "# Depth Completion Inference Server (PC Version)\n"
            "Ứng dụng chạy trên PC nhận dữ liệu từ Raspberry Pi 5 (RGB + LiDAR) dạng MJPEG Stream và so sánh tham chiếu trực tiếp với Camera RealSense (Ground Truth)."
        )

        with gr.Row():
            model_selector = gr.Dropdown(label="Chọn model", choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0] if MODEL_OPTIONS else None)

        with gr.Tabs():
            with gr.Tab("Live capture"):
                with gr.Row():
                    pi5_url_input = gr.Textbox(label="Pi 5 Stream URL", value="http://192.168.3.2:8080/video", scale=3)
                    connect_btn = gr.Button("⚡ Kết nối / Đổi URL", scale=1)
                
                live_button = gr.Button("Capture & Inference", variant="primary")
                gr.Markdown("*Nhận ảnh RGB từ Pi Camera 2 và dữ liệu LiDAR truyền trực tiếp dạng MJPEG Stream từ Raspberry Pi 5.*")
                
                with gr.Row():
                    gr.Markdown("### 1. Đầu vào từ Raspberry Pi 5")
                with gr.Row():
                    live_captured_rgb = gr.Image(label="Captured RGB", type="pil")
                    live_lidar_overlay = gr.Image(label="RGB + LiDAR overlay", type="pil")
                
                with gr.Row():
                    gr.Markdown("### 2. Kết quả dự đoán & Tham chiếu so sánh")
                with gr.Row():
                    live_depth = gr.Image(label="Predicted depth map", type="pil")
                    live_colormap = gr.Image(label="Predicted colormap", type="pil")
                with gr.Row():
                    rs_rgb = gr.Image(label="RealSense RGB", type="pil")
                    rs_depth_colormap = gr.Image(label="RealSense Depth Colormap", type="pil")
                
                live_status = gr.Textbox(label="Trạng thái", interactive=False)
                live_elapsed = gr.Textbox(label="Inference time", interactive=False)

            with gr.Tab("Direct upload"):
                uploaded_rgb = gr.Image(label="Upload RGB image", type="pil")
                uploaded_depth = gr.Image(label="Upload sparse depth image", type="pil")
                direct_button = gr.Button("🚀 Inference từ upload")
                gr.Markdown("*Upload RGB và sparse depth map đã chuẩn bị sẵn.*")
                with gr.Row():
                    direct_depth = gr.Image(label="Predicted depth map", type="pil")
                    direct_colormap = gr.Image(label="Predicted colormap", type="pil")
                direct_status = gr.Textbox(label="Trạng thái", interactive=False)
                direct_elapsed = gr.Textbox(label="Inference time", interactive=False)

        connect_btn.click(
            fn=update_pi5_connection,
            inputs=[pi5_url_input],
            outputs=[live_status],
        )

        live_button.click(
            fn=live_infer_pipeline,
            inputs=[model_selector],
            outputs=[
                live_captured_rgb, live_lidar_overlay, 
                live_depth, live_colormap, 
                rs_rgb, rs_depth_colormap, 
                live_status, live_elapsed
            ],
        )

        direct_button.click(
            fn=infer_pipeline,
            inputs=[model_selector, uploaded_rgb, uploaded_depth],
            outputs=[direct_depth, direct_colormap, direct_status, direct_elapsed],
        )

    demo.launch(server_name="localhost", share=False, theme=gr.themes.Soft(primary_hue="teal"))


if __name__ == "__main__":
    main()
