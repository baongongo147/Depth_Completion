import json
import os
import time
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

from src.networks import UNet

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "checkpoints" / "models"
DATASET_DIR = ROOT / "Private_Test_Datasets"


def scan_models(model_dir=MODEL_DIR):
    if not model_dir.exists():
        return []
    models = []
    for path in sorted(model_dir.iterdir()):
        if path.suffix.lower() in {".pth", ".onnx", ".xml"}:
            models.append(str(path))
    return models


def scan_test_ids(dataset_dir=DATASET_DIR):
    rgb_dir = dataset_dir / "Rgb"
    if not rgb_dir.exists():
        return []
    ids = []
    for file in sorted(rgb_dir.glob("*_rgb.png")):
        ids.append(file.stem.replace("_rgb", ""))
    return ids


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


def capture_rplidar_scan(port=DEFAULT_LIDAR_PORT, timeout=10.0):
    if RPLidar is None:
        raise RuntimeError("RPLidar chưa được cài đặt hoặc không tìm thấy trên hệ thống.")
    lidar = RPLidar(port, baudrate=115200, timeout=3)
    try:
        lidar.get_info()
        lidar.get_health()

        # Start motor and wait for it to stabilize
        if hasattr(lidar, "start_motor"):
            lidar.start_motor()
        time.sleep(3.0)

        # Flush stale data from serial buffer to prevent "Wrong body size" error
        serial_obj = getattr(lidar, "_serial", None) or getattr(lidar, "_serial_port", None)
        if serial_obj and hasattr(serial_obj, "reset_input_buffer"):
            serial_obj.reset_input_buffer()
        if hasattr(lidar, "clean_input"):
            lidar.clean_input()

        scan_result = queue.Queue()

        def scan_worker():
            try:
                scan = next(lidar.iter_scans(max_buf_meas=1000))
                scan_result.put(("ok", scan))
            except Exception as exc:
                scan_result.put(("err", exc))

        worker = threading.Thread(target=scan_worker, daemon=True)
        worker.start()

        try:
            status, payload = scan_result.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"RPLidar scan timeout after {timeout:.1f}s")

        if status == "err":
            raise payload

        return payload
    finally:
        try:
            lidar.stop()
        except Exception:
            pass
        if hasattr(lidar, "stop_motor"):
            try:
                lidar.stop_motor()
            except Exception:
                pass
        try:
            lidar.disconnect()
        except Exception:
            pass


def angle_distance_to_lidar_xz(angle_deg, distance_mm):
    rad = np.deg2rad(angle_deg)
    x = np.sin(rad) * distance_mm
    z = np.cos(rad) * distance_mm
    return x, z


def project_lidar_to_image(scan, K, dist_coeffs, rvec, tvec, image_shape):
    if cv2 is None:
        raise RuntimeError("OpenCV chưa được cài đặt, cần để xử lý LiDAR projection.")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(-1, 1)

    rvec = np.asarray(rvec, dtype=np.float64)
    if rvec.shape == (3, 3):
        rvec, _ = cv2.Rodrigues(rvec)
    if rvec.ndim == 1:
        rvec = rvec.reshape(3, 1)

    tvec = np.asarray(tvec, dtype=np.float64)
    if tvec.ndim == 1:
        tvec = tvec.reshape(3, 1)

    visible_points = []
    for quality, angle, distance in scan:
        if distance <= 0 or not np.isfinite(distance):
            continue
        x, z = angle_distance_to_lidar_xz(angle, distance)
        lidar_pos = np.array([[x, 0.0, z]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(lidar_pos, rvec, tvec, K, dist_coeffs)
        u, v = img_pts.ravel()
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        ix = int(round(u))
        iy = int(round(v))
        if 0 <= ix < image_shape[1] and 0 <= iy < image_shape[0]:
            visible_points.append((ix, iy, distance))
    return visible_points


def draw_lidar_overlay(rgb_frame, visible_points, point_radius=4):
    """Draw LiDAR points on the RGB image with distance-based coloring."""
    overlay = Image.fromarray(rgb_frame).copy()
    if not visible_points:
        return overlay

    distances = [d for _, _, d in visible_points]
    d_min, d_max = min(distances), max(distances)
    d_range = d_max - d_min if d_max - d_min > 1e-3 else 1.0

    draw = ImageDraw.Draw(overlay)
    for x, y, dist in visible_points:
        # Normalize distance: close=red, far=blue
        t = (dist - d_min) / d_range
        r = int(255 * (1.0 - t))
        g = int(255 * max(0.0, 1.0 - abs(t - 0.5) * 2))
        b = int(255 * t)
        color = (r, g, b)
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=color,
            outline=(255, 255, 255),
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
    Resize RGB and sparse depth to target size for ONNX/OpenVINO models.
    Returns resized arrays and original shape for later resizing back.
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
    rgb_resized = Image.fromarray(rgb_np).resize((target_w, target_h), resample=Image.BILINEAR)
    rgb_resized = np.array(rgb_resized)

    sparse_img_pil = Image.fromarray((sparse_np * 65535.0).astype(np.uint16))
    sparse_resized = sparse_img_pil.resize((target_w, target_h), resample=Image.NEAREST)
    sparse_resized = np.array(sparse_resized).astype(np.float32) / 65535.0

    return rgb_resized, sparse_resized, (original_h, original_w)


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


def load_dataset_sample(sample_id, use_fusion=False):
    rgb_path = DATASET_DIR / "Rgb" / f"{sample_id}_rgb.png"
    lidar_path = DATASET_DIR / "Lidar" / f"{sample_id}_lidar.json"
    depth_path = DATASET_DIR / "Depth" / f"{sample_id}_depth_16bit.png"
    if use_fusion:
        fusion_path = DATASET_DIR / "Rgb - fusion" / f"{sample_id}_rgb_fusion.png"
        if fusion_path.exists():
            rgb_path = fusion_path
    if not rgb_path.exists() or not lidar_path.exists() or not depth_path.exists():
        raise FileNotFoundError("Sample data missing from Private_Test_Datasets")
    rgb = Image.open(rgb_path).convert("RGB")
    gt_depth = Image.open(depth_path).convert("I")
    sparse = build_sparse_depth_from_lidar(np.array(gt_depth).astype(np.float32) / 65535.0, lidar_path)
    sparse_img = Image.fromarray((sparse * 65535.0).astype(np.uint16))
    return rgb, sparse_img


def prepare_model_inputs(rgb_image, sparse_image):
    rgb_np = read_rgb_image(rgb_image)
    raw_np = read_depth_image(sparse_image)
    raw_np = resize_to_match(rgb_np, raw_np)
    rgb_tensor = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    raw_tensor = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0)
    hole_tensor = (raw_tensor > 0).to(torch.float32)
    relative = float(raw_tensor.sum()) == 0.0
    return rgb_tensor, raw_tensor, hole_tensor, relative


class ModelManager:
    def __init__(self):
        self.model_path = None
        self.model_type = None
        self.model = None
        self.session = None
        self.compiled = None

    def load(self, model_path):
        self.model_path = model_path
        self.model_type = Path(model_path).suffix.lower()
        self.model = None
        self.session = None
        self.compiled = None
        if self.model_type == ".pth":
            self.model = UNet(rezero=True)
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "network" in checkpoint:
                state_dict = checkpoint["network"]
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            return f"Loaded PyTorch model: {Path(model_path).name}"
        if self.model_type == ".onnx":
            if ort is None:
                raise RuntimeError("onnxruntime is not installed")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            return f"Loaded ONNX model: {Path(model_path).name}"
        if self.model_type == ".xml":
            if ov is None:
                raise RuntimeError("openvino is not installed")
            core = ov.Core()
            model = core.read_model(model=model_path)
            self.compiled = core.compile_model(model=model, device_name="CPU")
            return f"Loaded OpenVINO model: {Path(model_path).name}"
        raise RuntimeError("Unsupported model type")

    def infer(self, rgb_tensor, raw_tensor, hole_tensor, original_shape=None):
        if self.model_type == ".pth":
            padded_rgb, pad_info = pad_to_multiple(rgb_tensor, 64)
            padded_raw, _ = pad_to_multiple(raw_tensor, 64)
            padded_hole, _ = pad_to_multiple(hole_tensor, 64)
            with torch.no_grad():
                pred = self.model(padded_rgb, padded_raw, padded_hole)
            if pad_info is not None:
                pad_h, pad_w = pad_info
                if pad_h > 0:
                    pred = pred[:, :, :-pad_h, :]
                if pad_w > 0:
                    pred = pred[:, :, :, :-pad_w]
            return pred
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
        raise RuntimeError("Unsupported model type")


model_manager = ModelManager()
MODEL_OPTIONS = scan_models()
TEST_IDS = scan_test_ids()


def run_model_inference(rgb_img, sparse_img, model_path):
    if model_path is None or model_path == "":
        return None, None, "Chưa chọn model", ""
    if model_manager.model_path != model_path:
        try:
            model_manager.load(model_path)
        except Exception as exc:
            return None, None, f"Không load được model: {exc}", ""
    try:
        model_type = Path(model_path).suffix.lower()
        if model_type in {".onnx", ".xml"}:
            rgb_np, sparse_np, original_shape = resize_for_model(rgb_img, sparse_img, target_h=512, target_w=896)
            rgb_tensor, raw_tensor, hole_tensor, relative = prepare_model_inputs(
                Image.fromarray(rgb_np),
                Image.fromarray((sparse_np * 65535.0).astype(np.uint16))
            )
        else:
            rgb_tensor, raw_tensor, hole_tensor, relative = prepare_model_inputs(rgb_img, sparse_img)
            original_shape = None

        start = time.time()
        pred_tensor = model_manager.infer(rgb_tensor, raw_tensor, hole_tensor, original_shape)
        elapsed = time.time() - start

        if original_shape is not None and model_type in {".onnx", ".xml"}:
            pred_np = pred_tensor.squeeze().numpy()
            orig_h, orig_w = original_shape
            pred_img = Image.fromarray((pred_np * 65535.0).astype(np.uint16))
            pred_img = pred_img.resize((orig_w, orig_h), resample=Image.NEAREST)
            pred_tensor = torch.from_numpy(np.array(pred_img).astype(np.float32) / 65535.0).unsqueeze(0).unsqueeze(0)

        depth_uint16 = adjust_domain(pred_tensor, relative)
        depth_img = Image.fromarray((depth_uint16 // 256).astype(np.uint8))
        colormap_img = depth_to_colormap(depth_uint16)
        status = f"Inference hoàn thành trong {elapsed:.2f}s"
        return depth_img, colormap_img, status, f"{elapsed:.2f} seconds"
    except Exception as exc:
        return None, None, f"Lỗi inference: {exc}", ""


def live_infer_pipeline(model_path):
    """Capture RGB + LiDAR, show preview, then run inference."""
    # Returns: (captured_rgb, lidar_overlay, depth_map, colormap, status, elapsed)
    empty = (None, None, None, None, "", "")

    if model_path is None or model_path == "":
        return (*empty[:4], "Chưa chọn model", "")
    if not is_live_capture_supported():
        missing = []
        if Picamera2 is None:
            missing.append("Picamera2")
        if RPLidar is None:
            missing.append("RPLidar")
        if cv2 is None:
            missing.append("OpenCV")
        return (*empty[:4], f"Live capture không khả dụng: thiếu {', '.join(missing)}", "")

    try:
        K, dist_coeffs, tvec, rvec = load_camera_lidar_calibration()
    except Exception as exc:
        return (*empty[:4], f"Không load calibration: {exc}", "")

    try:
        # Capture RGB frame
        rgb_frame = capture_picamera2_rgb()
        captured_rgb = Image.fromarray(rgb_frame)

        # Capture LiDAR scan
        lidar_scan = capture_rplidar_scan()

        # Project LiDAR points onto image
        visible_points = project_lidar_to_image(lidar_scan, K, dist_coeffs, rvec, tvec, rgb_frame.shape)

        # Draw LiDAR overlay
        lidar_overlay = draw_lidar_overlay(rgb_frame, visible_points)

        if len(visible_points) == 0:
            return captured_rgb, lidar_overlay, None, None, "Không phát hiện điểm LiDAR tương ứng trên ảnh", ""

        # Build sparse depth and run inference
        sparse_img = build_sparse_depth_from_lidar_points(rgb_frame.shape, visible_points)
        depth_img, colormap_img, status, elapsed = run_model_inference(captured_rgb, sparse_img, model_path)
        n_pts = len(visible_points)
        status = f"Capture: {rgb_frame.shape[1]}x{rgb_frame.shape[0]}, {n_pts} LiDAR points | {status}"
        return captured_rgb, lidar_overlay, depth_img, colormap_img, status, elapsed

    except Exception as exc:
        return (*empty[:4], f"Live capture error: {exc}", "")


def infer_pipeline(model_path, uploaded_rgb, uploaded_depth):
    if model_path is None or model_path == "":
        return None, None, "Chưa chọn model", ""
    if uploaded_rgb is None or uploaded_depth is None:
        return None, None, "Cần upload cả RGB và sparse depth", ""
    return run_model_inference(uploaded_rgb, uploaded_depth, model_path)


def main():
    with gr.Blocks(title="Depth Completion Gradio App") as demo:
        gr.Markdown(
            "# Depth Completion Inference App\n"
            "Sử dụng RGB + dữ liệu sparse depth từ LiDAR để tạo depth map hoàn chỉnh."
        )

        with gr.Row():
            model_selector = gr.Dropdown(label="Chọn model", choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0] if MODEL_OPTIONS else None)

        with gr.Tabs():
            with gr.Tab("Live capture"):
                live_button = gr.Button("🚀 Capture & Inference")
                gr.Markdown("*Chụp ảnh trực tiếp từ Pi Camera và ánh xạ LiDAR bằng RPLIDAR.*")
                with gr.Row():
                    live_captured_rgb = gr.Image(label="Captured RGB", type="pil")
                    live_lidar_overlay = gr.Image(label="RGB + LiDAR overlay", type="pil")
                with gr.Row():
                    live_depth = gr.Image(label="Predicted depth map", type="pil")
                    live_colormap = gr.Image(label="Predicted colormap", type="pil")
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

        live_button.click(
            fn=live_infer_pipeline,
            inputs=[model_selector],
            outputs=[live_captured_rgb, live_lidar_overlay, live_depth, live_colormap, live_status, live_elapsed],
        )

        direct_button.click(
            fn=infer_pipeline,
            inputs=[model_selector, uploaded_rgb, uploaded_depth],
            outputs=[direct_depth, direct_colormap, direct_status, direct_elapsed],
        )

    demo.launch(server_name="localhost", share=False, theme=gr.themes.Soft(primary_hue="teal"))


if __name__ == "__main__":
    main()
